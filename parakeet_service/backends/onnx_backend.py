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
    ONNX-based ASR backend using onnx-asr library.

    This backend provides faster inference with INT8 quantization.
    Performance: ~181x realtime on T4 GPU with TensorRT.

    Requires: pip install onnx-asr onnxruntime-gpu (or onnxruntime for CPU)
    """

    def __init__(self, model_name: str = "base.int8"):
        """
        Initialize ONNX backend.

        Args:
            model_name: onnx-asr model variant
                - "tiny", "tiny.int8"
                - "base", "base.int8"
                - "small", "small.int8"
        """
        self.model_name = model_name
        self._model = None
        self._cfg = OnnxConfig()

    def load(self) -> "OnnxBackend":
        """Load the ONNX model."""
        try:
            import onnx_asr
        except ImportError as e:
            raise ImportError(
                "onnx-asr not installed. Install with: pip install onnx-asr onnxruntime-gpu"
            ) from e

        logger.info("Loading %s with ONNX backend...", self.model_name)
        self._model = onnx_asr.load_model(self.model_name)
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
