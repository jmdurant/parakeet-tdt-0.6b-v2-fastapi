from contextlib import asynccontextmanager
import contextlib
import gc
import torch, asyncio

from .config import MODEL_NAME, MODEL_BACKEND, logger
from .backends import load_backend

from parakeet_service.batchworker import batch_worker


def _to_builtin(obj):
    """torch/NumPy → pure-Python (JSON-safe)."""
    import numpy as np
    import torch as th

    if isinstance(obj, (th.Tensor, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    return obj


@asynccontextmanager
async def lifespan(app):
    """Load model once per process; free GPU on shutdown."""
    logger.info("Loading %s with %s backend...", MODEL_NAME, MODEL_BACKEND)

    # Load backend (NeMo or ONNX based on config)
    backend = load_backend()
    backend.load()

    # For backward compatibility, expose the underlying model
    # NeMo backend has .model property, ONNX backend returns itself
    if hasattr(backend, 'model') and backend.model is not None:
        app.state.asr_model = backend.model
    else:
        app.state.asr_model = backend

    app.state.asr_backend = backend
    logger.info("Backend ready")

    # Start batch worker (only for NeMo backend with actual model)
    if MODEL_BACKEND == "nemo" and hasattr(backend.model, 'transcribe'):
        app.state.worker = asyncio.create_task(batch_worker(backend.model), name="batch_worker")
        logger.info("batch_worker scheduled")
    else:
        app.state.worker = None

    try:
        yield
    finally:
        if app.state.worker:
            app.state.worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await app.state.worker

        logger.info("Releasing memory and shutting down")
        backend.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def reset_fast_path(model):
    """Restore low-latency decoding flags."""
    # Handle both direct model and backend wrapper
    if hasattr(model, 'reset_fast_path'):
        model.reset_fast_path()
        return

    # Direct NeMo model access
    from omegaconf import open_dict
    if hasattr(model, 'cfg') and hasattr(model.cfg, 'decoding'):
        with open_dict(model.cfg.decoding):
            if getattr(model.cfg.decoding, "compute_timestamps", False):
                model.cfg.decoding.compute_timestamps = False
            if getattr(model.cfg.decoding, "preserve_alignments", False):
                model.cfg.decoding.preserve_alignments = False
        model.change_decoding_strategy(model.cfg.decoding)
