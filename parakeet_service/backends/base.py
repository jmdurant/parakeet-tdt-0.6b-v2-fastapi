"""Base protocol for ASR backends."""

from typing import Protocol, Any


class ASRBackend(Protocol):
    """Protocol defining the interface for ASR backends."""

    def transcribe(
        self,
        paths: list[str],
        batch_size: int = 2,
        timestamps: bool = False
    ) -> list[Any]:
        """
        Transcribe audio files.

        Args:
            paths: List of paths to audio files
            batch_size: Number of files to process in parallel
            timestamps: Whether to include word/segment timestamps

        Returns:
            List of transcription results. Each result should have:
            - text: str - The transcribed text
            - timestamp: dict (optional) - Timestamp information if requested
        """
        ...

    @property
    def cfg(self) -> Any:
        """Return model configuration (for NeMo compatibility)."""
        ...
