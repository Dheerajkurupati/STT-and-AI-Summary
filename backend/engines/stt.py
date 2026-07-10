"""
Speech-to-text engines for the upload pipeline.

WHY BOTH ENGINES SHARE THE SAME OUTPUT SHAPE:
Every engine returns (segments, detected_language) where each segment is a
plain {"start": float, "end": float, "text": str} dict in absolute seconds —
exactly the shape whisperx.align() expects (verified against the installed
whisperx package: align() only reads segment["text"]/"start"/"end", nothing
whisper-specific). This is what makes SenseVoice-Small a drop-in alternative
to Whisper without touching alignment, diarization, or anything downstream.

WHY TRANSCRIBE() TAKES SPANS INSTEAD OF THE WHOLE AUDIO:
Both engines are called once per VAD span (see backend/engines/vad.py). With
VAD_ENGINE=none there is exactly one span covering the whole file, so this
reproduces today's single-call behavior exactly. With FSMN-VAD enabled, each
span gets its own STT call — smaller, silence-free clips reduce the audio
Whisper/SenseVoice have to consider per call.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from backend.config import settings
from backend.engines.vad import SpeechSpan
from backend.utils import get_logger

logger = get_logger(__name__)


class SttEngine(Protocol):
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        spans: list[SpeechSpan],
        language: str | None,
    ) -> tuple[list[dict], str]: ...


class WhisperSttEngine:
    """
    Wraps WhisperX's Whisper large-v3 model. Extracted verbatim from the
    pre-version2 transcribe.py so WhisperXPipeline can orchestrate it through
    the same VAD-span loop every engine uses, without changing what actually
    gets sent to whisper_model.transcribe().
    """

    def __init__(self) -> None:
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            import whisperx

            logger.info(
                "Loading Whisper model '%s' on device=%s compute_type=%s",
                settings.whisper_model,
                settings.device,
                settings.compute_type,
            )
            self._model = whisperx.load_model(
                settings.whisper_model,
                device=settings.device,
                compute_type=settings.compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        spans: list[SpeechSpan],
        language: str | None,
    ) -> tuple[list[dict], str]:
        model = self._load()
        segments: list[dict] = []
        detected_language: str | None = None

        for span in spans:
            start_sample = int(span.start * sample_rate)
            end_sample = int(span.end * sample_rate)
            clip = audio[start_sample:end_sample]
            if len(clip) == 0:
                continue

            result = model.transcribe(clip, batch_size=settings.batch_size, language=language)
            if detected_language is None:
                detected_language = result["language"]

            for seg in result["segments"]:
                text = (seg.get("text") or "").strip()
                if not text:
                    continue
                segments.append(
                    {
                        "start": span.start + float(seg.get("start", 0.0)),
                        "end": span.start + float(seg.get("end", 0.0)),
                        "text": text,
                    }
                )

        return segments, (detected_language or language or "en")


class SenseVoiceSttEngine:
    """
    FunASR SenseVoice-Small, loaded lazily and cached on the instance.

    WHY EACH SPAN BECOMES ONE SEGMENT:
    Unlike Whisper, SenseVoice doesn't emit sub-segment timestamps — it
    returns one text block per call. Since transcribe() is already called once
    per VAD span, the span's own (start, end) bounds become the segment's
    timestamps. Word-level timing is still recovered afterward by
    whisperx.align(), same as the Whisper path — alignment doesn't care which
    engine produced the segment text.

    KNOWN LIMITATION (confirmed on uploads/test_meeting.wav during the
    version2 benchmark): because a segment can never be split smaller than
    one VAD span, back-to-back conversational turns with no silence gap
    between speakers collapse into a single segment/single speaker label —
    even with VAD_ENGINE=fsmn enabled, since FSMN-VAD only splits on actual
    silence, not on speaker changes. Whisper avoids this because it emits its
    own internal sub-segment timestamps regardless of VAD. This makes
    SenseVoice a weaker choice specifically for fast back-and-forth dialogue
    without pauses — worth weighing against its multilingual claims when
    deciding whether to switch the default.
    """

    def __init__(self) -> None:
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from funasr import AutoModel

            logger.info("Loading SenseVoice-Small (FunASR, CPU)")
            # Verified against the actual FunASR install: "iic/SenseVoice-small"
            # 404s on ModelScope — the real ID is "iic/SenseVoiceSmall", and it's
            # natively registered in this funasr version (SenseVoiceSmall in
            # AutoModel's registry), so trust_remote_code isn't needed either.
            self._model = AutoModel(
                model="iic/SenseVoiceSmall",
                device="cpu",
                disable_update=True,
            )
        return self._model

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        spans: list[SpeechSpan],
        language: str | None,
    ) -> tuple[list[dict], str]:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        model = self._load()
        segments: list[dict] = []
        # SenseVoice's own language codes; "auto" lets it detect per clip.
        sense_language = language or "auto"

        for span in spans:
            start_sample = int(span.start * sample_rate)
            end_sample = int(span.end * sample_rate)
            clip = audio[start_sample:end_sample]
            if len(clip) == 0:
                continue

            result = model.generate(
                input=clip,
                fs=sample_rate,
                language=sense_language,
                use_itn=True,  # inverse text normalization: adds punctuation/numerals
            )
            if not result:
                continue

            text = rich_transcription_postprocess(result[0].get("text", "")).strip()
            if text:
                segments.append({"start": span.start, "end": span.end, "text": text})

        return segments, (language or "en")


def get_stt_engine(name: str) -> SttEngine:
    if name == "sensevoice":
        return SenseVoiceSttEngine()
    return WhisperSttEngine()
