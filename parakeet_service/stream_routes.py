from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from parakeet_service.streaming_vad import StreamingVAD
from parakeet_service.batchworker import transcribe_blob
import json
router = APIRouter()


async def _send_transcriptions(ws: WebSocket, chunks: list[str], queued: bool = False):
    for chunk in chunks:
        if queued:
            await ws.send_json({"status": "queued"})
        text = await transcribe_blob(chunk)
        await ws.send_json({"text": text})


async def _ws_asr_handler(ws: WebSocket):
    """Core WebSocket ASR handler - shared by both /ws and /ws/transcribe endpoints."""
    await ws.accept()
    vad = StreamingVAD()

    try:
        while True:
            frame = await ws.receive_bytes()
            await _send_transcriptions(ws, vad.feed(frame), queued=True)
    except WebSocketDisconnect:
        pass


@router.websocket("/ws")
async def ws_asr(ws: WebSocket):
    """WebSocket endpoint for streaming ASR."""
    await _ws_asr_handler(ws)


@router.websocket("/ws/transcribe")
async def ws_asr_transcribe(ws: WebSocket):
    """WebSocket endpoint for streaming ASR (alias for pipeline compatibility)."""
    await _ws_asr_handler(ws)


@router.websocket("/vosk")
async def ws_vosk(ws: WebSocket):
    """Jigasi/Vosk-compatible streaming endpoint.

    Jigasi first sends ``{"config":{"sample_rate":...}}``, then signed
    little-endian PCM16 frames, and finally ``{"eof":1}``. Vosk final
    responses use the ``text`` key; omitting ``partial`` marks them final.
    """
    await ws.accept()
    vad = None
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break

            if message.get("text") is not None:
                command = json.loads(message["text"])
                if "config" in command:
                    sample_rate = int(command["config"].get("sample_rate", 16000))
                    vad = StreamingVAD(input_sample_rate=sample_rate)
                    continue
                if command.get("eof") == 1:
                    if vad is not None:
                        await _send_transcriptions(ws, vad.flush())
                    await ws.close(code=1000)
                    break
                continue

            if message.get("bytes") is not None:
                if vad is None:
                    vad = StreamingVAD()
                await _send_transcriptions(ws, vad.feed(message["bytes"]))
    except (WebSocketDisconnect, RuntimeError):
        pass
