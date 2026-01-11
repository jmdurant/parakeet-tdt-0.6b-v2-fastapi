import logging, os, sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model name - configurable via PARAKEET_MODEL env var
# To change models: update .env, rebuild with --no-cache
MODEL_NAME = os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3")

# Configuration from environment variables
TARGET_SR = int(os.getenv("TARGET_SR", "16000"))          # model’s native sample-rate
MODEL_PRECISION = os.getenv("MODEL_PRECISION", "fp16")
DEVICE = os.getenv("DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "30"))   # seconds
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", "60"))    # seconds

# Backend selection: "nemo" (default) or "onnx"
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "nemo")
ONNX_MODEL_NAME = os.getenv("ONNX_MODEL_NAME", "base.int8")

# Chunking method: "vad" (default, Silero VAD) or "silence" (FFmpeg silencedetect)
CHUNKING_METHOD = os.getenv("CHUNKING_METHOD", "vad")
SILENCE_THRESHOLD = os.getenv("SILENCE_THRESHOLD", "-30dB")
SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", "0.5"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    stream=sys.stdout,
    force=True
)

logger = logging.getLogger("parakeet_service")
