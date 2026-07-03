"""ONNX ASR backend using onnx-asr library."""

from typing import Any
from dataclasses import dataclass, field

from ..config import logger


@dataclass
class OnnxConfig:
    """Mock config object for ONNX backend (NeMo compatibility shim)."""

    @dataclass
    class Decoding:
        compute_timestamps: bool = False
        preserve_alignments: bool = False

    decoding: Decoding = field(default_factory=Decoding)


@dataclass
class OnnxResult:
    """Result object mimicking NeMo's output format."""
    text: str
    timestamp: dict = field(default_factory=dict)


class OnnxBackend:
    """
    ONNX-based ASR backend using the onnx-asr library.

    Faster, lighter inference than NeMo (no PyTorch/NeMo dependency tree),
    with optional INT8 quantization and GPU execution via onnxruntime.

    Requires: pip install onnx-asr 'onnxruntime-gpu==1.22.0' (or onnxruntime for CPU).
    Pin note: newer onnxruntime-gpu wheels link CUDA 13 (libcudart.so.13); on a
    CUDA-12 image that crashes at import, so pin to a CUDA-12 build (1.22.0 verified).
    """

    def __init__(self, model_name: str = "nemo-parakeet-tdt-0.6b-v3",
                 quantization: str = "int8"):
        """
        Initialize ONNX backend.

        Args:
            model_name: onnx-asr registry name, e.g. "nemo-parakeet-tdt-0.6b-v3"
                (must be a real onnx-asr model id — "base.int8" is NOT one).
            quantization: "int8" for the fast quantized weights, "" for fp32.
        """
        self.model_name = model_name
        self.quantization = quantization or None
        self._model = None
        self._cfg = OnnxConfig()

    def load(self) -> "OnnxBackend":
        """Load the ONNX model onto the GPU when one is available."""
        try:
            import onnx_asr
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnx-asr/onnxruntime not installed. "
                "Install with: pip install onnx-asr 'onnxruntime-gpu==1.22.0'"
            ) from e

        # Route inference onto CUDA explicitly — without providers, onnxruntime-gpu
        # silently falls back to CPU.
        available = ort.get_available_providers()
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"])

        logger.info("Loading %s with ONNX backend (quant=%s, providers=%s)...",
                    self.model_name, self.quantization, providers)
        self._model = onnx_asr.load_model(
            self.model_name,
            quantization=self.quantization,
            providers=providers,
        )
        logger.info("ONNX backend ready")
        return self

    def transcribe(
        self,
        paths: list[str],
        batch_size: int = 2,
        timestamps: bool = False
    ) -> list[Any]:
        """
        Transcribe audio files using ONNX.

        Note: onnx-asr doesn't support batch processing, so files are
        processed sequentially. The batch_size parameter is ignored.

        Args:
            paths: List of paths to audio files
            batch_size: Ignored (kept for API compatibility)
            timestamps: Whether to include word/segment timestamps

        Returns:
            List of OnnxResult objects with text and timestamps
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = []
        for path in paths:
            try:
                result = self._model.recognize(path)

                if timestamps:
                    # Get timestamps if requested
                    result_with_ts = result.with_timestamps()
                    ts_data = {
                        "segments": result_with_ts.get("segments", []),
                        "words": result_with_ts.get("words", []),
                    }
                    results.append(OnnxResult(
                        text=result_with_ts.get("text", str(result)),
                        timestamp=ts_data
                    ))
                else:
                    # Just get text
                    text = result.text if hasattr(result, "text") else str(result)
                    results.append(OnnxResult(text=text))

            except Exception as e:
                logger.error(f"ONNX transcription failed for {path}: {e}")
                results.append(OnnxResult(text="", timestamp={}))

        return results

    @property
    def cfg(self) -> OnnxConfig:
        """Return mock config object (for NeMo compatibility)."""
        return self._cfg

    @property
    def model(self) -> Any:
        """Return the underlying onnx-asr model."""
        return self._model

    def reset_fast_path(self):
        """No-op for ONNX backend (always fast)."""
        pass

    def unload(self):
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("ONNX backend unloaded")
