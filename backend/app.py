"""
FastAPI layer: exposes the pipeline (preprocess -> transcribe -> format ->
summarize) as HTTP endpoints.

WHY THIS FILE STAYS THIN:
app.py contains no business logic of its own — it only saves the upload,
calls into utils/transcribe/formatter/summarize, and shapes HTTP
responses/errors. This is deliberate: every module it calls also works as a
plain Python script/CLI without FastAPI at all, which matters for local
debugging (you can call backend.transcribe.pipeline.transcribe(...)
directly in a REPL without spinning up a server). If business logic lived
here, that would no longer be true.

WHY MODELS ARE NOT PRELOADED AT STARTUP:
Whisper large-v3 + the diarization pipeline take real time and several GB
of memory to load. Loading them eagerly at startup would slow down `uvicorn
backend.app:app` boot even for requests that never come (e.g. health
checks in a container orchestrator). Instead, transcribe.py's
WhisperXPipeline loads lazily on first use and stays cached afterward —
first request pays the cost, every request after is fast.
"""

from __future__ import annotations

import shutil
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import settings
from backend.formatter import format_transcript, transcript_to_dict, write_transcript_outputs
from backend.summarize import SummarizationError, summarizer, write_summary_outputs
from backend.transcribe import TranscriptionError, pipeline
from backend.utils import (
    AudioProcessingError,
    cleanup_temp_file,
    convert_to_wav,
    get_logger,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at process startup — ensures uploads/outputs/temp/logs exist
    # before the first request arrives, rather than each module creating
    # directories ad hoc on first write.
    settings.ensure_directories()
    logger.info("meeting-ai API started")
    yield
    logger.info("meeting-ai API shutting down")


app = FastAPI(title="meeting-ai", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness check. Deliberately does not touch WhisperX/Ollama, so it
    stays fast and doesn't trigger model loading."""
    return {"status": "ok"}





@app.post("/api/transcribe")
async def transcribe_meeting(file: UploadFile = File(...)) -> JSONResponse:
    """
    Transcription pipeline: upload -> preprocess -> WhisperX -> format.
    """
    upload_path = settings.upload_dir / f"{uuid.uuid4().hex}_{file.filename}"
    wav_path = None

    try:
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        wav_path = convert_to_wav(upload_path)

        transcription = pipeline.transcribe(wav_path)
        transcript = format_transcript(transcription.segments)
        transcript_json_path, transcript_txt_path = write_transcript_outputs(
            transcript, settings.output_dir
        )

    except AudioProcessingError as exc:
        logger.error("Audio processing failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TranscriptionError as exc:
        logger.error("Transcription failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        # The original upload is kept (uploads/ is meant to persist raw
        # input for audit/reprocessing, per the project's storage lifecycle
        # design) — only the intermediate WAV in temp/ is scratch space.
        if wav_path is not None:
            cleanup_temp_file(wav_path)

    return JSONResponse(
        {
            "language": transcription.language,
            "duration_seconds": transcription.duration_seconds,
            "transcript": transcript_to_dict(transcript),
            "output_files": {
                "transcript_json": str(transcript_json_path),
                "transcript_txt": str(transcript_txt_path),
            },
        }
    )


class SummarizeRequest(BaseModel):
    transcript_dict: dict


@app.post("/api/summarize")
def summarize_meeting_api(req: SummarizeRequest) -> JSONResponse:
    from backend.formatter import MeetingTranscript, TranscriptBlock
    
    try:
        blocks = [TranscriptBlock(**b) for b in req.transcript_dict.get("blocks", [])]
        transcript = MeetingTranscript(
            blocks=blocks,
            speaker_count=req.transcript_dict.get("speaker_count", 0),
            duration_seconds=req.transcript_dict.get("duration_seconds", 0.0)
        )
        summary = summarizer.summarize(transcript)
        summary_json_path, summary_txt_path = write_summary_outputs(summary, settings.output_dir)
        
        return JSONResponse(
            {
                "executive_summary": summary.executive_summary,
                "key_topics": summary.key_topics,
                "action_items": summary.action_items,
                "decisions": summary.decisions,
                "risks": summary.risks,
                "next_steps": summary.next_steps,
                "output_files": {
                    "summary_json": str(summary_json_path),
                    "summary_txt": str(summary_txt_path),
                },
            }
        )
    except SummarizationError as exc:
        logger.error("Summarization failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


app.mount("/", StaticFiles(directory="static", html=True), name="static")
