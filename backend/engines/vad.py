"""
Voice Activity Detection engines for the upload pipeline.

WHY THIS EXISTS:
The current upload pipeline (pre-version2) has no dedicated VAD stage — WhisperX
transcribes the whole normalized file, silence included. FSMN-VAD adds an
optional stage that trims silence before STT even runs, which can reduce
hallucinations on long quiet stretches. It is off by default (VAD_ENGINE=none)
so the current pipeline's exact behavior is unchanged unless explicitly
enabled — see backend/config.py's vad_engine setting.

WHY DETECT() RETURNS SPANS INSTEAD OF TRIMMED AUDIO:
Concatenating only the speech portions of the audio would desync every
downstream timestamp from the original file, requiring a fragile time-remapping
layer for the transcript. Returning (start, end) spans in the ORIGINAL file's
timeline instead lets every downstream consumer (STT engines, whisperx.align)
keep working in absolute, original-file time natively — a span is just a
window to slice the full audio array with, not a rewrite of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from backend.utils import get_logger

logger = get_logger(__name__)


@dataclass
class SpeechSpan:
    """One speech region, in seconds, absolute within the source audio."""

    start: float
    end: float


class VadEngine(Protocol):
    def detect(self, audio: np.ndarray, sample_rate: int) -> list[SpeechSpan]: ...


class NoVad:
    """
    Default engine: treats the entire audio as one speech span.

    This is what makes VAD_ENGINE=none reproduce today's exact pipeline
    behavior — downstream STT engines slice audio[start:end] per span, and
    slicing the one span [0, duration] is a no-op equal to using the full
    array, exactly like the current code does today.
    """

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[SpeechSpan]:
        duration = len(audio) / sample_rate
        return [SpeechSpan(start=0.0, end=duration)]


class FsmnVad:
    """
    FunASR FSMN-VAD, loaded lazily and cached on the instance (same pattern as
    WhisperXPipeline's model loading in transcribe.py). Runs once per upload
    over the full preprocessed WAV.
    """

    def __init__(self) -> None:
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from funasr import AutoModel

            logger.info("Loading FSMN-VAD (FunASR, CPU)")
            self._model = AutoModel(model="fsmn-vad", device="cpu", disable_update=True)
        return self._model

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[SpeechSpan]:
        model = self._load()
        result = model.generate(input=audio, fs=sample_rate)

        spans: list[SpeechSpan] = []
        if result and result[0].get("value"):
            for start_ms, end_ms in result[0]["value"]:
                spans.append(SpeechSpan(start=start_ms / 1000.0, end=end_ms / 1000.0))

        if not spans:
            # No speech detected at all (or FSMN-VAD found nothing) — fall back
            # to one span covering the whole file rather than silently
            # returning zero spans, which would skip STT entirely.
            duration = len(audio) / sample_rate
            logger.warning("FSMN-VAD found no speech spans; falling back to whole-file span")
            spans = [SpeechSpan(start=0.0, end=duration)]

        return spans

    def has_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
        """
        Cheap yes/no speech check, used by the live path (stream.py) as a
        pre-filter before calling faster-whisper. Deliberately does NOT use
        detect()'s whole-file fallback: for the live path, "no speech found"
        should mean "skip this round and wait for more buffered audio," not
        "assume the whole window is speech" — the buffer keeps accumulating
        either way, so a false negative here only costs one skipped round,
        never lost audio.
        """
        model = self._load()
        result = model.generate(input=audio, fs=sample_rate)
        return bool(result and result[0].get("value"))


def get_vad_engine(name: str) -> VadEngine:
    if name == "fsmn":
        return FsmnVad()
    return NoVad()
