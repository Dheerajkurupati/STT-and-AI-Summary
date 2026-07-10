"""
Punctuation restoration for the upload pipeline.

WHY THIS EXISTS:
Client's table calls for adding punctuation restoration as an explicit new
stage (not a benchmark) — useful when an STT engine's output is weakly
punctuated. Runs after STT, before alignment: safe because wav2vec2 CTC
alignment (backend/transcribe.py's _load_align_model/whisperx.align) ignores
punctuation tokens when doing phoneme matching, it only uses them in the
final word text. Off by default (ENABLE_PUNCTUATION_RESTORATION=false in
backend/config.py) so the current Whisper/SenseVoice baseline is unaffected
unless explicitly enabled for comparison.

VERIFICATION STATUS — READ BEFORE ENABLING:
FSMN-VAD, SenseVoice-Small, and CAM++ (backend/engines/vad.py, stt.py,
diarization.py) were each smoke-tested against this exact FunASR install to
confirm their real output shape before this code was written. FunASR's
`ct-punc` model download stalled on this network during implementation and
did not finish in time to do the same here. Every FunASR text-restoration
model observed in this codebase's other engines returns a list of dicts with
a "text" key (SenseVoice does), so restore() reads that key — but this one
specific assumption is NOT independently confirmed the way the others are.
_extract_text() below is defensive (checks a couple of plausible key names
and falls back to the original, unpunctuated text with a logged warning
rather than crashing or silently corrupting output) specifically because of
that gap. Before relying on this in a real comparison run: enable it on one
short transcript and visually confirm the output actually gained punctuation
rather than silently falling back.
"""

from __future__ import annotations

from typing import Any, Protocol

from backend.utils import get_logger

logger = get_logger(__name__)


class PunctuationRestorer(Protocol):
    def restore(self, segments: list[dict]) -> list[dict]: ...


class NoOpPunctuator:
    """Default: returns segments unchanged. Matches today's behavior exactly."""

    def restore(self, segments: list[dict]) -> list[dict]:
        return segments


class CtTransformerPunctuator:
    """FunASR CT-Transformer punctuation restoration, loaded lazily and cached."""

    def __init__(self) -> None:
        self._model: Any = None
        self._warned_fallback = False

    def _load(self) -> Any:
        if self._model is None:
            from funasr import AutoModel

            logger.info("Loading CT-Transformer punctuation model (FunASR, CPU)")
            self._model = AutoModel(model="ct-punc", device="cpu", disable_update=True)
        return self._model

    def _extract_text(self, result: Any, original: str) -> str:
        if result and isinstance(result, list) and isinstance(result[0], dict):
            for key in ("text", "punc_text", "value"):
                candidate = result[0].get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()

        if not self._warned_fallback:
            logger.warning(
                "CT-Transformer output shape unrecognized (%r) — falling back to "
                "unpunctuated text for this and future segments. See "
                "backend/engines/punctuation.py's verification-status note.",
                result,
            )
            self._warned_fallback = True
        return original

    def restore(self, segments: list[dict]) -> list[dict]:
        if not segments:
            return segments

        model = self._load()
        restored: list[dict] = []
        for seg in segments:
            original_text = seg.get("text", "")
            if not original_text.strip():
                restored.append(seg)
                continue

            try:
                result = model.generate(input=original_text)
                punctuated = self._extract_text(result, original_text)
            except Exception as exc:
                logger.warning("CT-Transformer failed on a segment, keeping original text: %s", exc)
                punctuated = original_text

            restored.append({**seg, "text": punctuated})

        return restored


def get_punctuator(enabled: bool) -> PunctuationRestorer:
    if enabled:
        return CtTransformerPunctuator()
    return NoOpPunctuator()
