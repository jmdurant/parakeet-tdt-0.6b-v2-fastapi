from __future__ import annotations
import io, wave, tempfile, numpy as np, torch, torchaudio
from typing import List
from torch.hub import load as torch_hub_load

# Use cached repo from Docker build (see Dockerfile pre-download step)
SILERO_VAD_PATH = "/root/.cache/torch/hub/snakers4_silero-vad_master"
vad_model, vad_utils = torch_hub_load(SILERO_VAD_PATH, "silero_vad", source='local', trust_repo=True)
(_, _, _, VADIterator, _) = vad_utils

# TODO: Update to read from .env
SAMPLE_RATE              = 16_000         # model is trained for 16 kHz
WINDOW_SAMPLES           = 512            # 32 ms frame
THRESHOLD                = 0.60           # voice prob ≥ 0.60 → speech
MIN_SILENCE_MS           = 250            # flush after ≥250 ms quiet
SPEECH_PAD_MS            = 120            # keep 120 ms context before/after
MAX_SPEECH_MS            = 8_000          # hard stop at 8 s

# Helper: float32 → int16 PCM bytes
def _f32_to_pcm16(frames: np.ndarray) -> bytes:
    return np.clip(frames * 32768, -32768, 32767).astype(np.int16).tobytes()

class StreamingVAD:
    """
    Feed successive 20–40 ms PCM frames (16 kHz, int16 mono).
    Emits temp-file *paths* when a full utterance is detected.
    """

    def __init__(self, input_sample_rate: int = SAMPLE_RATE):
        if input_sample_rate <= 0:
            raise ValueError("input_sample_rate must be positive")
        self.input_sample_rate = input_sample_rate
        self.vad = VADIterator(
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=THRESHOLD,
            min_silence_duration_ms=MIN_SILENCE_MS,
            speech_pad_ms=SPEECH_PAD_MS,
        )
        self.buffer = bytearray()
        self.pending = np.empty(0, dtype=np.float32)
        self.speech_ms = 0


    def _flush(self) -> List[str]:
        if not self.buffer:
            return []
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(self.buffer)
        self.buffer.clear()
        self.speech_ms = 0
        self.vad.reset_states()
        return [tmp.name]

    def feed(self, frame_bytes: bytes) -> List[str]:
        out: List[str] = []

        pcm_f32 = np.frombuffer(frame_bytes, np.int16).astype("float32") / 32768
        if self.input_sample_rate != SAMPLE_RATE and pcm_f32.size:
            tensor = torch.from_numpy(pcm_f32)
            pcm_f32 = torchaudio.functional.resample(
                tensor, self.input_sample_rate, SAMPLE_RATE
            ).numpy()
        if self.pending.size:
            pcm_f32 = np.concatenate((self.pending, pcm_f32))

        complete = (len(pcm_f32) // WINDOW_SAMPLES) * WINDOW_SAMPLES
        self.pending = pcm_f32[complete:].copy()
        for start in range(0, complete, WINDOW_SAMPLES):
            window = pcm_f32[start:start + WINDOW_SAMPLES]

            voice_event = self.vad(window, return_seconds=False)
            self.buffer.extend(_f32_to_pcm16(window))
            self.speech_ms += 32

            # Flush on trailing-silence event or max-length guard
            if voice_event and voice_event.get("end"):
                out.extend(self._flush())
            elif self.speech_ms >= MAX_SPEECH_MS:
                out.extend(self._flush())

        return out

    def flush(self) -> List[str]:
        """Flush residual audio when a websocket sends Vosk's EOF message."""
        if self.pending.size:
            self.buffer.extend(_f32_to_pcm16(self.pending))
            self.pending = np.empty(0, dtype=np.float32)
        return self._flush()
