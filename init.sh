#!/bin/bash
# Initialize model cache on first run
# This copies pre-downloaded models from the image to the mounted volume if needed

set -e

# Check if Silero VAD cache exists
if [ ! -d "/root/.cache/torch/hub/snakers4_silero-vad_master" ]; then
    echo "Initializing Silero VAD cache from image..."
    mkdir -p /root/.cache/torch/hub
    cp -r /opt/model-cache/torch/hub/* /root/.cache/torch/hub/
    echo "Silero VAD cache initialized."
fi

# Check if HuggingFace cache exists (look for any parakeet model)
if ! ls /root/.cache/huggingface/hub/models--nvidia--parakeet* 1>/dev/null 2>&1; then
    echo "Initializing HuggingFace cache from image..."
    mkdir -p /root/.cache/huggingface
    cp -r /opt/model-cache/huggingface/* /root/.cache/huggingface/
    echo "HuggingFace cache initialized."
fi

echo "Model cache ready. Starting service..."
exec "$@"
