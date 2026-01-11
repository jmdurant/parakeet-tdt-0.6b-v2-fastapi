"""
Offline VAD-aware splitters
───────────────────────────
* vad_chunk_lowmem     – Low-memory chunker for non-streaming processing
  - target 60-s chunks (±10 s)
  - never cut mid-utterance (trailing silence ≥ 300 ms)
  - processes audio incrementally to minimize memory usage
  - returns List[pathlib.Path] of temp .wav files

* vad_chunk_streaming  – Low-RAM streamer for streaming use cases
  - processes audio in small stripes (2 seconds at a time)
  - uses VADIterator to split on speech boundaries
  - returns List[pathlib.Path] of temp .wav files
"""

from __future__ import annotations
import tempfile, wave, pathlib, numpy as np, subprocess, re
from typing import List
import soundfile as sf

from .config import CHUNKING_METHOD, SILENCE_THRESHOLD, SILENCE_DURATION, logger


from torch.hub import load as torch_hub_load

# Use cached repo from Docker build (see Dockerfile pre-download step)
SILERO_VAD_PATH = "/root/.cache/torch/hub/snakers4_silero-vad_master"
vad_model, vad_utils = torch_hub_load(SILERO_VAD_PATH, "silero_vad", source='local', trust_repo=True)
get_speech_ts, _, _, VADIterator, _ = vad_utils 

SAMPLE_RATE        = 16_000
TARGET_SEC         = 60
MAX_SEC            = 70          # never exceed this in one chunk
TRAIL_SIL_MS       = 300         # keep ≥300 ms silence at cut point
THRESH             = 0.60        # stricter prob threshold

def vad_chunk_lowmem(path: pathlib.Path) -> List[pathlib.Path]:
    """Low-memory VAD chunking for non-streaming processing"""
    import librosa
    
    # Get audio file info
    with sf.SoundFile(path) as snd:
        file_sr = snd.samplerate
        duration = len(snd) / file_sr
        
    # Initialize VAD iterator
    vad_iter = VADIterator(
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=THRESH,
        min_silence_duration_ms=TRAIL_SIL_MS,
        speech_pad_ms=SPEECH_PAD_MS,
    )
    
    # Buffer for current chunk
    current_chunk = bytearray()
    chunks = []
    speech_ms = 0
    
    # Process audio in chunks
    for chunk_start in range(0, int(duration * SAMPLE_RATE), STRIPE_FRAMES):
        # Load audio segment with librosa for format conversion
        y, _ = librosa.load(
            path, 
            sr=SAMPLE_RATE, 
            offset=chunk_start/SAMPLE_RATE,
            duration=STRIPE_SEC,
            mono=True
        )
        
        # Convert to int16
        audio_int16 = (y * 32767).astype(np.int16)
        
        # Process in 512-sample windows
        for i in range(0, len(audio_int16), 512):
            window = audio_int16[i:i+512]
            if len(window) < 512:
                break
                
            # Convert to float for VAD
            window_f32 = window.astype(np.float32) / 32768
            evt = vad_iter(window_f32)
            
            # Add to current chunk
            current_chunk.extend(window.tobytes())
            speech_ms += 32
            
            # Check if we should finalize chunk
            if (evt and evt.get("end")) or speech_ms >= MAX_CHUNK_MS:
                if current_chunk:
                    chunks.append(_flush(current_chunk))
                    current_chunk.clear()
                    speech_ms = 0
    
    # Finalize last chunk
    if current_chunk:
        chunks.append(_flush(current_chunk))
    
    return chunks

# Constants for both chunkers
STRIPE_SEC        = 2                         # read 2-second stripes
STRIPE_FRAMES     = SAMPLE_RATE * STRIPE_SEC
MAX_CHUNK_MS      = 60_000                    # hard 60s cap
SPEECH_PAD_MS     = 120                       # same as live VAD
TARGET_SEC        = 60
MAX_SEC           = 70
TRAIL_SIL_MS      = 300
THRESH            = 0.60

def _flush(buf: bytearray) -> pathlib.Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(bytes(buf))
    return pathlib.Path(tmp.name)

def vad_chunk_streaming(path: pathlib.Path) -> List[pathlib.Path]:
    """
    Stream the file in small stripes and split on VADIterator boundaries.
    Uses the SAME PyTorch Silero model, but keeps only a few seconds in RAM.
    """
    vad_iter = VADIterator(
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=THRESH,
        min_silence_duration_ms=TRAIL_SIL_MS,
        speech_pad_ms=SPEECH_PAD_MS,
    )

    paths, buf = [], bytearray()
    speech_ms  = 0

    with sf.SoundFile(path) as snd:
        while True:
            audio = snd.read(frames=STRIPE_FRAMES, dtype="int16", always_2d=False)
            if not len(audio):
                break

            # Normalise to float32 [-1,1] for VADIterator
            audio_f32 = audio.astype("float32") / 32768

            # Feed 512-sample windows
            for start in range(0, len(audio_f32), 512):
                window = audio_f32[start:start+512]
                if len(window) < 512:
                    break
                evt = vad_iter(window)
                buf.extend(audio[start:start+512].tobytes())
                speech_ms += 32

                if (evt and evt.get("end")) or speech_ms >= MAX_CHUNK_MS:
                    paths.append(_flush(buf))
                    buf.clear()
                    speech_ms = 0

    if buf:
        paths.append(_flush(buf))
    return paths


def ffmpeg_silence_chunk(
    path: pathlib.Path,
    max_chunk_sec: float = 90,
    silence_thresh: str = None,
    silence_duration: float = None
) -> List[pathlib.Path]:
    """
    Use FFmpeg silencedetect to find chunk boundaries.

    This is faster than VAD since FFmpeg uses native C code with no Python/PyTorch overhead.
    Good for realtime use cases where latency matters.

    Args:
        path: Path to audio file
        max_chunk_sec: Maximum chunk duration (default 90s)
        silence_thresh: Silence threshold in dB (default from config)
        silence_duration: Minimum silence duration in seconds (default from config)

    Returns:
        List of paths to temporary WAV chunk files
    """
    silence_thresh = silence_thresh or SILENCE_THRESHOLD
    silence_duration = silence_duration or SILENCE_DURATION

    # Get audio duration first
    probe_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        total_duration = float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"ffprobe failed, falling back to VAD: {e}")
        return vad_chunk_lowmem(path)

    # Short audio - no chunking needed
    if total_duration <= max_chunk_sec:
        return [path]

    # Run FFmpeg silencedetect
    detect_cmd = [
        "ffmpeg", "-v", "warning", "-i", str(path), "-af",
        f"silencedetect=n={silence_thresh}:d={silence_duration}",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(detect_cmd, capture_output=True, text=True)
        stderr = result.stderr
    except subprocess.CalledProcessError as e:
        logger.warning(f"silencedetect failed, falling back to VAD: {e}")
        return vad_chunk_lowmem(path)

    # Parse silence boundaries from stderr
    # Format: [silencedetect @ ...] silence_start: 1.234
    # Format: [silencedetect @ ...] silence_end: 2.345 | silence_duration: 1.111
    silence_starts = [float(m) for m in re.findall(r'silence_start:\s*([\d.]+)', stderr)]
    silence_ends = [float(m) for m in re.findall(r'silence_end:\s*([\d.]+)', stderr)]

    # Build chunk boundaries at silence midpoints
    # Prefer cutting at silence regions, respecting max_chunk_sec
    cut_points = [0.0]
    last_cut = 0.0

    for start, end in zip(silence_starts, silence_ends):
        midpoint = (start + end) / 2
        time_since_cut = midpoint - last_cut

        # Cut if we're past target duration and at a silence boundary
        if time_since_cut >= max_chunk_sec * 0.7:  # Allow cuts at ~70% of max
            cut_points.append(midpoint)
            last_cut = midpoint
        # Force cut if we've exceeded max duration
        elif time_since_cut >= max_chunk_sec:
            cut_points.append(midpoint)
            last_cut = midpoint

    cut_points.append(total_duration)

    # If no good cut points found, fall back to fixed-duration chunks
    if len(cut_points) <= 2:
        num_chunks = int(total_duration / max_chunk_sec) + 1
        cut_points = [i * max_chunk_sec for i in range(num_chunks)]
        cut_points.append(total_duration)

    # Extract chunks using FFmpeg
    chunks = []
    for i in range(len(cut_points) - 1):
        start = cut_points[i]
        duration = cut_points[i + 1] - start

        if duration < 0.1:  # Skip very short chunks
            continue

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()

        extract_cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-ss", str(start), "-i", str(path),
            "-t", str(duration),
            "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE),
            tmp.name
        ]
        try:
            subprocess.run(extract_cmd, check=True, capture_output=True)
            chunks.append(pathlib.Path(tmp.name))
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to extract chunk {i}: {e}")
            # Clean up failed temp file
            pathlib.Path(tmp.name).unlink(missing_ok=True)

    logger.info(f"ffmpeg_silence_chunk: split {total_duration:.1f}s audio into {len(chunks)} chunks")
    return chunks if chunks else [path]


def get_chunker(method: str = None):
    """
    Get the chunking function based on method.

    Args:
        method: "vad" or "silence". Defaults to CHUNKING_METHOD from config.

    Returns:
        Chunking function (vad_chunk_lowmem or ffmpeg_silence_chunk)
    """
    method = method or CHUNKING_METHOD
    if method == "silence":
        return ffmpeg_silence_chunk
    return vad_chunk_lowmem
