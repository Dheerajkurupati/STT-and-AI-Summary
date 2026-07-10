"""
Upload pipeline orchestration: VAD -> STT -> punctuation -> word alignment ->
diarization -> speaker assignment -> post-processing.

WHY THIS FILE ORCHESTRATES RATHER THAN HARDCODES ONE ENGINE (version2):
Client feedback on speaker-label accuracy led to adding benchmark
alternatives for VAD/STT/diarization (see backend/engines/) alongside the
original WhisperX/pyannote pipeline. This file no longer owns those model
calls directly — each stage's actual model logic lives in its own
backend/engines/*.py module (same isolation principle as before, just one
level more granular), and WhisperXPipeline.transcribe() wires the selected
engines (backend.config.settings.vad_engine / stt_engine /
diarization_engine) together. The two stages the client's table marked
"keep" — WhisperX's wav2vec2 word alignment and whisperx.assign_word_speakers
— are NOT pluggable and still called directly here verbatim, because they
were verified (against the installed whisperx source) to work generically
with any engine's output: alignment only needs segment start/end/text, and
assign_word_speakers only needs a DataFrame with start/end/speaker columns.

WHY TranscriptionResult / RawSegment DON'T CHANGE:
Every other module (formatter.py, app.py, cli.py) talks to this file only
through TranscriptionResult/RawSegment. Regardless of which VAD/STT/
diarization engine is selected, transcribe() still returns the exact same
shape — downstream code needs zero changes to support new engines.

WHY MODELS/ENGINES ARE LOADED LAZILY AND CACHED ON THE INSTANCE:
Whisper large-v3, the diarization pipeline, and every FunASR model are
large and take real time to load from disk. In a FastAPI server handling
multiple requests, we load them once (at first use) and reuse the same
instance for every subsequent request, rather than reloading per-request.
Which engine gets constructed is decided once, from settings, on first
access — settings are loaded once at process startup, so this is
equivalent to a per-process engine choice (this is also why cli.py's
per-run --stt/--vad/--diarization overrides must be set before the first
call to pipeline.transcribe() in that process).
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
    Orchestrates the upload pipeline over pluggable engines (see
    backend/engines/): VAD -> STT -> [punctuation] -> align -> diarize ->
    assign speakers -> post-process. One instance is created and reused for
    the lifetime of the process (see app.py's startup hook); each engine is
    constructed lazily on first use and cached, same as before.
    """

    def __init__(self) -> None:
        self._align_model: Any = None
        self._align_metadata: Any = None
        self._align_language: str | None = None
        self._vad_engine: Any = None
        self._stt_engine: Any = None
        self._diarization_engine: Any = None
        self._punctuator: Any = None

    def _get_vad_engine(self) -> Any:
        if self._vad_engine is None:
            from backend.engines.vad import get_vad_engine

            logger.info("Selected VAD engine: %s", settings.vad_engine)
            self._vad_engine = get_vad_engine(settings.vad_engine)
        return self._vad_engine

    def _get_stt_engine(self) -> Any:
        if self._stt_engine is None:
            from backend.engines.stt import get_stt_engine

            logger.info("Selected STT engine: %s", settings.stt_engine)
            self._stt_engine = get_stt_engine(settings.stt_engine)
        return self._stt_engine

    def _get_diarization_engine(self) -> Any:
        if self._diarization_engine is None:
            from backend.engines.diarization import get_diarization_engine

            logger.info("Selected diarization engine: %s", settings.diarization_engine)
            self._diarization_engine = get_diarization_engine(settings.diarization_engine)
        return self._diarization_engine

    def _get_punctuator(self) -> Any:
        if self._punctuator is None:
            from backend.engines.punctuation import get_punctuator

            self._punctuator = get_punctuator(settings.enable_punctuation_restoration)
        return self._punctuator

    def _load_align_model(self, language_code: str) -> tuple[Any, Any]:
        # Alignment models are language-specific, so we reload only if the
        # detected language changes between requests (rare, but possible
        # with multilingual meetings processed back to back).
        # NOT pluggable: verified against the installed whisperx source that
        # align() only needs segment start/end/text, so it works identically
        # regardless of which STT engine produced those segments.
        if self._align_model is None or self._align_language != language_code:
            import whisperx

            logger.info("Loading alignment model for language='%s'", language_code)
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language_code, device=settings.device
            )
            self._align_language = language_code
        return self._align_model, self._align_metadata

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """
        Run the full upload pipeline on a preprocessed (mono, 16kHz WAV)
        audio file: VAD -> STT -> [punctuation] -> align -> diarize -> assign
        speakers -> post-process.
        """
        import whisperx

        try:
            audio = whisperx.load_audio(str(audio_path))
        except Exception as exc:  # whisperx/ffmpeg-backed loader raises broadly
            raise TranscriptionError(f"Failed to load audio {audio_path}: {exc}") from exc

        from whisperx.audio import SAMPLE_RATE as sample_rate  # 16000; matches utils.py's ffmpeg output

        try:
            vad_spans = self._get_vad_engine().detect(audio, sample_rate)

            raw_segments, detected_language = self._get_stt_engine().transcribe(
                audio, sample_rate, vad_spans, settings.language
            )
            # Whisper large-v3 sometimes returns variants like 'en_US'
            # but the alignment model expects strict 2-letter ISO codes.
            if len(detected_language) > 2:
                detected_language = detected_language[:2].lower()
            logger.info("Detected language (normalized): %s", detected_language)

            if settings.enable_punctuation_restoration:
                raw_segments = self._get_punctuator().restore(raw_segments)

            if not raw_segments:
                return TranscriptionResult(segments=[], language=detected_language, duration_seconds=0.0)

            align_model, align_metadata = self._load_align_model(detected_language)
            aligned = whisperx.align(
                raw_segments,
                align_model,
                align_metadata,
                audio,
                settings.device,
                return_char_alignments=False,
            )

            diarize_df = self._get_diarization_engine().diarize(audio, sample_rate)
            result = whisperx.assign_word_speakers(diarize_df, aligned)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Pipeline failed on {audio_path}: {exc}") from exc

        segments = self._to_raw_segments(result["segments"])
        duration = max((s.end for s in segments), default=0.0)

        return TranscriptionResult(
            segments=segments,
            language=detected_language,
            duration_seconds=duration,
        )

    @staticmethod
    def _force_sentence_speakers(words: list[dict]) -> None:
        """
        Groups words into sentences using punctuation (.!?), counts the speaker
        frequencies within each sentence, and mathematically forces the ENTIRE
        sentence to belong to the majority speaker. This perfectly eliminates
        mid-sentence boundary bleeds caused by Pyannote without using an LLM.
        """
        if not words:
            return

        sentence_start_idx = 0
        for i, word_obj in enumerate(words):
            text = word_obj.get("word", "").strip()
            if not text:
                continue
                
            # If this word ends with sentence-ending punctuation, or it's the very last word
            if text[-1] in ".!?" or i == len(words) - 1:
                # We found a full sentence from sentence_start_idx to i
                sentence_words = words[sentence_start_idx:i + 1]
                
                # Count speakers
                speaker_counts = {}
                for w in sentence_words:
                    spk = w.get("speaker")
                    if spk:
                        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
                
                # Find majority speaker
                if speaker_counts:
                    majority_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]
                    # Force all words in this sentence to the majority speaker
                    for w in sentence_words:
                        w["speaker"] = majority_speaker
                
                sentence_start_idx = i + 1

    @classmethod
    def _to_raw_segments(cls, whisperx_segments: list[dict]) -> list[RawSegment]:
        segments: list[RawSegment] = []
        for seg in whisperx_segments:
            if "words" not in seg or not seg["words"]:
                text = (seg.get("text") or "").strip()
                if text:
                    segments.append(RawSegment(
                        start=float(seg.get("start", 0.0)),
                        end=float(seg.get("end", 0.0)),
                        text=text,
                        speaker=seg.get("speaker", "SPEAKER_UNKNOWN"),
                    ))
                continue

            # Apply sentence-level majority voting to fix Pyannote mid-sentence bleeds
            cls._force_sentence_speakers(seg["words"])

            current_speaker: str | None = None
            current_words: list[str] = []
            current_start = 0.0

            for word_obj in seg["words"]:
                word_speaker = word_obj.get("speaker", seg.get("speaker", "SPEAKER_UNKNOWN"))
                word_text = word_obj.get("word", "").strip()
                if not word_text:
                    continue

                if current_speaker is None:
                    current_speaker = word_speaker
                    current_start = word_obj.get("start", 0.0)

                # Split on speaker change
                if word_speaker != current_speaker and current_words:
                    segments.append(RawSegment(
                        start=float(current_start),
                        end=float(word_obj.get("start", 0.0)),
                        text=" ".join(current_words),
                        speaker=current_speaker,
                    ))
                    current_words = []
                    current_speaker = word_speaker
                    current_start = word_obj.get("start", 0.0)

                current_words.append(word_text)

            if current_words:
                last_word = seg["words"][-1]
                segments.append(RawSegment(
                    start=float(current_start),
                    end=float(last_word.get("end", 0.0)),
                    text=" ".join(current_words),
                    speaker=current_speaker or "SPEAKER_UNKNOWN",
                ))

        # Perform semantic boundary snapping to fix minor word bleeds
        return cls._semantic_boundary_snap(segments)

    @classmethod
    def _semantic_boundary_snap(cls, segments: list[RawSegment]) -> list[RawSegment]:
        """
        Fixes punctuation drift. 
        If a sentence ends with punctuation, but the next speaker's block starts 
        with a lowercase continuation word, we mathematically pull that word back.
        """
        if not segments:
            return segments

        for i in range(len(segments) - 1):
            curr = segments[i]
            nxt = segments[i + 1]

            if curr.speaker == nxt.speaker:
                continue

            curr_words = curr.text.split()
            nxt_words = nxt.text.split()

            if not curr_words or not nxt_words:
                continue

            # Bleed type 1: Pyannote pushed the LAST word of Speaker A into Speaker B
            if nxt_words[0][-1] in ".!?" and len(nxt_words) > 1 and nxt_words[1][0].isupper():
                bleeding_word = nxt_words.pop(0)
                curr_words.append(bleeding_word)
                curr.text = " ".join(curr_words)
                nxt.text = " ".join(nxt_words)

            # Bleed type 2: Pyannote pushed the FIRST word of Speaker B into Speaker A
            elif curr_words[-1][-1] not in ".!?" and curr_words[-1].islower() and len(curr_words) > 1 and curr_words[-2][-1] in ".!?":
                bleeding_word = curr_words.pop()
                nxt_words.insert(0, bleeding_word)
                curr.text = " ".join(curr_words)
                nxt.text = " ".join(nxt_words)

        # Remove any segments that became empty after snapping
        return [s for s in segments if s.text.strip()]


# Single shared instance, analogous to `settings` in config.py — app.py
# imports this rather than constructing a new WhisperXPipeline per request,
# so multi-GB models are loaded once per process, not once per upload.
pipeline = WhisperXPipeline()
