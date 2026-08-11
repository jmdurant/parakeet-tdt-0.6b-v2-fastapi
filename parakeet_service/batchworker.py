import asyncio, contextlib, logging, tempfile, pathlib, time, torch
from dataclasses import dataclass
from typing import Union, List

from parakeet_service import model as mdl

logger = logging.getLogger("batcher")
logger.setLevel(logging.DEBUG)

# -------- shared state -------------------------------------------------------
@dataclass
class TranscriptionJob:
    payload: str | bytes
    future: asyncio.Future[str]


transcription_queue: asyncio.Queue[TranscriptionJob] = asyncio.Queue()


async def transcribe_blob(blob: str | bytes) -> str:
    """Queue one utterance and return only its result to the submitting client."""
    future = asyncio.get_running_loop().create_future()
    await transcription_queue.put(TranscriptionJob(blob, future))
    return await future

# -------- helper -------------------------------------------------------------
def _as_path(blob: Union[str, bytes]) -> str:
    """Ensures we always hand a *file path* to NeMo."""
    if isinstance(blob, str):
        return blob
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(blob)
    tmp.close()
    return tmp.name

# -------- main worker --------------------------------------------------------
async def batch_worker(model, batch_ms: float = 15.0, max_batch: int = 4):
    """Forever drain `transcription_queue` → ASR → `results`."""
    logger.info("worker started (batch ≤%d, window %.0f ms)", max_batch, batch_ms)
    logger.info("worker started with model id=%s", id(model))
    

    while True:
        job = await transcription_queue.get()      # blocks until 1st item
        jobs: List[TranscriptionJob] = [job]

        # ---------- micro-batch gathering with timeout ----------
        deadline = time.monotonic() + batch_ms / 1000
        while len(jobs) < max_batch:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                jobs.append(await asyncio.wait_for(transcription_queue.get(), timeout))
            except asyncio.TimeoutError:
                break

        batch = [_as_path(item.payload) for item in jobs]
        logger.debug("processing %d-file batch", len(batch))

        # ---------- inference ----------
        try:
            with torch.inference_mode():
                outs = model.transcribe(
                    batch, batch_size=len(batch)
                )                                       # NeMo API
        except Exception as exc:
            logger.exception("ASR failed: %s", exc)
            for item in jobs:
                if not item.future.done():
                    item.future.set_exception(exc)
                transcription_queue.task_done()
            continue

        # ---------- store results & notify ----------
        for item, h in zip(jobs, outs):
            if not item.future.done():
                item.future.set_result(getattr(h, "text", str(h)))
            transcription_queue.task_done()            # mark done

        # ---------- cleanup ----------
        for p in batch:
            with contextlib.suppress(FileNotFoundError):
                pathlib.Path(p).unlink(missing_ok=True)
