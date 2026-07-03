"""
ASR Backend adapters for Parakeet service.

Provides a unified interface for different ASR backends:
- NeMo (default): Full-featured, PyTorch-based
- ONNX: Lightweight, faster inference with INT8 quantization

Usage:
    from parakeet_service.backends import load_backend

    backend = load_backend()  # Uses MODEL_BACKEND from config
    results = backend.transcribe(["audio.wav"], timestamps=True)
"""

from .base import ASRBackend
from .nemo_backend import NemoBackend

__all__ = ["ASRBackend", "NemoBackend", "load_backend"]


def load_backend(backend_type: str = None) -> ASRBackend:
    """
    Load the specified ASR backend.

    Args:
        backend_type: "nemo" or "onnx". Defaults to MODEL_BACKEND from config.

    Returns:
        ASRBackend instance
    """
    from ..config import (MODEL_BACKEND, ONNX_MODEL_NAME, ONNX_QUANTIZATION,
                          MODEL_NAME, MODEL_PRECISION, DEVICE)

    backend_type = backend_type or MODEL_BACKEND

    if backend_type == "onnx":
        from .onnx_backend import OnnxBackend
        return OnnxBackend(model_name=ONNX_MODEL_NAME, quantization=ONNX_QUANTIZATION)
    else:
        return NemoBackend(
            model_name=MODEL_NAME,
            precision=MODEL_PRECISION,
            device=DEVICE
        )
