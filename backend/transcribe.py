"""
WhisperX integration: transcription + word alignment + speaker diarization.

WHY THIS FILE OWNS ALL OF WHISPERX/PYANNOTE:
This is the only module that imports whisperx (and, transitively,
pyannote.audio). Every other module talks to this one through the plain
TranscriptionResult/RawSegment data shapes, never through WhisperX's own
types. That means:
- If WhisperX changes its API (it has, across versions), only this file
  needs to change.
- If we ever swap the engine entirely (e.g. a hosted transcription API),
  only this file changes — formatter.py, summarize.py, and app.py are
  unaffected because they only depend on RawSegment/TranscriptionResult.

WHY MODELS ARE LOADED LAZILY AND CACHED ON THE INSTANCE:
Whisper large-v3 and the diarization pipeline are multi-GB models that take
real time to load from disk. In a FastAPI server handling multiple
requests, we load them once (at first use) and reuse the same instance for
every subsequent request, rather than reloading per-request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.formatter import RawSegment
from backend.utils import get_logger

logger = get_logger(__name__)


class TranscriptionError(Exception):
    """Raised when audio loading, transcription, alignment, or diarization fails."""


@dataclass
class TranscriptionResult:
    """Everything downstream code (formatter.py) needs from a transcription run."""

    segments: list[RawSegment]
    language: str
    duration_seconds: float


class WhisperXPipeline:
    """
    Thin, stateful wrapper around WhisperX's three stages: transcribe,
    align, diarize. One instance is created and reused for the lifetime of
    the process (see app.py's startup hook).
    """

    def __init__(self) -> None:
        self._whisper_model: Any = None
        self._align_model: Any = None
        self._align_metadata: Any = None
        self._align_language: str | None = None
        self._diarize_pipeline: Any = None

    def _load_whisper_model(self) -> Any:
        if self._whisper_model is None:
            # Imported lazily, not at module load time, so importing
            # backend.transcribe (e.g. for type checking or from app.py's
            # module graph) doesn't force torch/whisperx to load immediately.
            import whisperx

            logger.info(
                "Loading Whisper model '%s' on device=%s compute_type=%s",
                settings.whisper_model,
                settings.device,
                settings.compute_type,
            )
            self._whisper_model = whisperx.load_model(
                settings.whisper_model,
                device=settings.device,
                compute_type=settings.compute_type,
            )
        return self._whisper_model

    def _load_align_model(self, language_code: str) -> tuple[Any, Any]:
        # Alignment models are language-specific, so we reload only if the
        # detected language changes between requests (rare, but possible
        # with multilingual meetings processed back to back).
        if self._align_model is None or self._align_language != language_code:
            import whisperx

            logger.info("Loading alignment model for language='%s'", language_code)
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language_code, device=settings.device
            )
            self._align_language = language_code
        return self._align_model, self._align_metadata

    def _load_diarize_pipeline(self) -> Any:
        if self._diarize_pipeline is None:
            if not settings.hf_token:
                raise TranscriptionError(
                    "HF_TOKEN is not set. Diarization requires a Hugging Face "
                    "token with access to pyannote/speaker-diarization-3.1 "
                    "(see README setup steps)."
                )

            import whisperx

            logger.info("Loading pyannote diarization pipeline")
            self._diarize_pipeline = whisperx.diarize.DiarizationPipeline(
                model_name=settings.diarization_model,
                token=settings.hf_token,
                device=settings.device,
            )
        return self._diarize_pipeline

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """
        Run the full WhisperX pipeline on a preprocessed (mono, 16kHz WAV)
        audio file: transcribe -> align -> diarize -> assign speakers.
        """
        import whisperx

        try:
            audio = whisperx.load_audio(str(audio_path))
        except Exception as exc:  # whisperx/ffmpeg-backed loader raises broadly
            raise TranscriptionError(f"Failed to load audio {audio_path}: {exc}") from exc

        try:
            whisper_model = self._load_whisper_model()
            transcription = whisper_model.transcribe(
                audio,
                batch_size=settings.batch_size,
                language=settings.language,
            )
            detected_language = transcription["language"]
            # Whisper large-v3 sometimes returns variants like 'en_US'
            # but the alignment model expects strict 2-letter ISO codes.
            if len(detected_language) > 2:
                detected_language = detected_language[:2].lower()
            logger.info("Detected language (normalized): %s", detected_language)

            align_model, align_metadata = self._load_align_model(detected_language)
            aligned = whisperx.align(
                transcription["segments"],
                align_model,
                align_metadata,
                audio,
                settings.device,
                return_char_alignments=False,
            )

            diarize_pipeline = self._load_diarize_pipeline()
            diarization = diarize_pipeline(
                audio,
                min_speakers=settings.min_speakers,
                max_speakers=settings.max_speakers,
            )
            result = whisperx.assign_word_speakers(diarization, aligned)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"WhisperX pipeline failed on {audio_path}: {exc}") from exc

        segments = self._to_raw_segments(result["segments"])
        duration = max((s.end for s in segments), default=0.0)

        return TranscriptionResult(
            segments=segments,
            language=detected_language,
            duration_seconds=duration,
        )

    @staticmethod
    def _to_raw_segments(whisperx_segments: list[dict]) -> list[RawSegment]:
        """
        Convert WhisperX's raw segment dicts into our typed RawSegment
        contract. A segment can be missing a "speaker" key if diarization
        couldn't confidently assign one (e.g. cross-talk) — we label those
        "SPEAKER_UNKNOWN" rather than dropping the text, so no speech is
        silently lost from the transcript.
        """
        segments: list[RawSegment] = []
        for seg in whisperx_segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                RawSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=text,
                    speaker=seg.get("speaker", "SPEAKER_UNKNOWN"),
                )
            )
        return segments


# Single shared instance, analogous to `settings` in config.py — app.py
# imports this rather than constructing a new WhisperXPipeline per request,
# so multi-GB models are loaded once per process, not once per upload.
pipeline = WhisperXPipeline()
