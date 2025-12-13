# Stage 1: Builder stage for installing dependencies
FROM python:3.10.7-slim AS builder

# Install system dependencies including ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir \
 && pip install cuda-python==12.6.0 \
 && pip install nemo_toolkit["asr"] \
 && pip install 'uvicorn[standard]' --no-cache-dir \
 && pip install --no-cache-dir -r requirements.txt \
 && pip cache purge

# Pre-download Silero VAD model so it's available at runtime without internet
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)"

# Pre-download Parakeet model from HuggingFace so it's available at runtime without internet
# Note: Change version here when upgrading, then rebuild with --no-cache
RUN python -c "import nemo.collections.asr as nemo_asr; nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2')"

# Stage 2: Runtime stage
FROM python:3.10.7-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./parakeet_service ./parakeet_service
COPY .env.example .env
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/torch /root/.cache/torch
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "parakeet_service.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
