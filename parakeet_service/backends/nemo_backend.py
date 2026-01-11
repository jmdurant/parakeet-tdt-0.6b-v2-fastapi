"""NeMo ASR backend wrapper."""

from typing import Any
import gc
import torch
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict

from ..config import logger


class NemoBackend:
    """NeMo-based ASR backend (default)."""

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        precision: str = "fp16",
        device: str = "cuda"
    ):
        """
        Initialize NeMo backend.

        Args:
            model_name: HuggingFace model name or local path
            precision: "fp16" or "fp32"
            device: "cuda" or "cpu"
        """
        self.model_name = model_name
        self.precision = precision
        self.device = device
        self._model = None

    def load(self) -> "NemoBackend":
        """Load the model into memory."""
        logger.info("Loading %s with NeMo backend...", self.model_name)

        with torch.inference_mode():
            dtype = torch.float16 if self.precision == "fp16" else torch.float32

            self._model = nemo_asr.models.ASRModel.from_pretrained(
                self.model_name,
                map_location=self.device
            ).to(dtype=dtype)

            logger.info("Loaded model with %s weights on %s", self.precision.upper(), self.device)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("NeMo backend ready on %s", next(self._model.parameters()).device)
        return self

    def transcribe(
        self,
        paths: list[str],
        batch_size: int = 2,
        timestamps: bool = False
    ) -> list[Any]:
        """
        Transcribe audio files using NeMo.

        Args:
            paths: List of paths to audio files
            batch_size: Number of files to process in parallel
            timestamps: Whether to include word/segment timestamps

        Returns:
            List of transcription results from NeMo
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self._model.transcribe(
            paths,
            batch_size=batch_size,
            timestamps=timestamps
        )

    @property
    def cfg(self) -> Any:
        """Return NeMo model configuration."""
        if self._model is None:
            return None
        return self._model.cfg

    @property
    def model(self) -> Any:
        """Return the underlying NeMo model (for direct access if needed)."""
        return self._model

    def reset_fast_path(self):
        """Restore low-latency decoding flags."""
        if self._model is None:
            return

        with open_dict(self._model.cfg.decoding):
            if getattr(self._model.cfg.decoding, "compute_timestamps", False):
                self._model.cfg.decoding.compute_timestamps = False
            if getattr(self._model.cfg.decoding, "preserve_alignments", False):
                self._model.cfg.decoding.preserve_alignments = False
        self._model.change_decoding_strategy(self._model.cfg.decoding)

    def unload(self):
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("NeMo backend unloaded")
