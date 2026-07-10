"""
Speaker diarization engines for the upload pipeline.

WHY BOTH ENGINES RETURN A PLAIN pandas.DataFrame:
Verified directly against the installed whisperx package
(.venv/.../whisperx/diarize.py): whisperx.assign_word_speakers(diarize_df, ...)
only ever reads diarize_df['start'], ['end'], ['speaker'] via .iterrows() — it
never touches pyannote-specific columns or types. That means any diarizer
that can produce a DataFrame with those three columns is a drop-in
replacement, and transcribe.py's call to assign_word_speakers() never needs
to change regardless of which engine produced the DataFrame.

WHY CamPlusPlusDiarizer IS ASSEMBLED FROM SCRATCH:
Unlike pyannote (a full segmentation + embedding + clustering pipeline behind
one call), CAM++ (verified via FunASR AutoModel) is only a speaker-embedding
extractor: model.generate(...) returns a single spk_embedding tensor per
clip, nothing else. Turning that into a diarizer means doing the
segmentation + clustering ourselves — this class is a best-effort wrapper for
benchmarking against pyannote, not a mature pipeline. It windows the audio
(reusing FsmnVad to skip silence), extracts one CAM++ embedding per window,
and clusters windows with cosine-distance agglomerative clustering — the
batch counterpart to the running cosine-similarity speaker matching stream.py
already does incrementally for the live path.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from backend.config import settings
from backend.utils import get_logger

logger = get_logger(__name__)


class DiarizationError(Exception):
    """Raised when a diarization engine can't run (missing token, model load failure, etc.)."""


class DiarizationEngine(Protocol):
    def diarize(self, audio: np.ndarray, sample_rate: int) -> pd.DataFrame: ...


class PyannoteDiarizer:
    """
    Wraps whisperx's DiarizationPipeline (pyannote 3.1 under the hood).
    Extracted verbatim from the pre-version2 transcribe.py.
    """

    def __init__(self) -> None:
        self._pipeline: Any = None

    def _load(self) -> Any:
        if self._pipeline is None:
            if not settings.hf_token:
                raise DiarizationError(
                    "HF_TOKEN is not set. Diarization requires a Hugging Face "
                    "token with access to pyannote/speaker-diarization-3.1 "
                    "(see README setup steps)."
                )
            # Explicit submodule import, not just `import whisperx`: whisperx's
            # top-level package doesn't unconditionally expose `.diarize` as an
            # attribute, so `whisperx.diarize.DiarizationPipeline` only works if
            # something else happened to import it first transitively. Verified
            # this breaks when the STT engine is SenseVoice instead of
            # WhisperX (which incidentally triggers that transitive import) —
            # explicit import makes this engine's loading independent of which
            # STT engine ran first.
            import whisperx.diarize

            logger.info("Loading pyannote diarization pipeline")
            self._pipeline = whisperx.diarize.DiarizationPipeline(
                model_name=settings.diarization_model,
                token=settings.hf_token,
                device=settings.device,
            )
        return self._pipeline

    def diarize(self, audio: np.ndarray, sample_rate: int) -> pd.DataFrame:
        pipeline = self._load()
        return pipeline(
            audio,
            min_speakers=settings.min_speakers,
            max_speakers=settings.max_speakers,
        )


class CamPlusPlusDiarizer:
    """
    FunASR CAM++ embeddings + sklearn agglomerative clustering.

    See module docstring for why this exists as a hand-assembled wrapper
    rather than a single pipeline call. Tuning constants below are a starting
    point for the pyannote-vs-CAM++ benchmark called for in the implementation
    plan, not tuned production values.
    """

    EMBEDDING_MODEL = "iic/speech_campplus_sv_zh_en_16k-common_advanced"
    WINDOW_SECONDS = 1.5
    HOP_SECONDS = 0.75
    MIN_WINDOW_SECONDS = 0.5  # shorter slivers produce unreliable embeddings, skip them
    # Cosine-distance threshold for merging two windows into the same speaker
    # when the exact speaker count isn't known via settings.min/max_speakers.
    DISTANCE_THRESHOLD = 0.4

    def __init__(self) -> None:
        self._embedding_model: Any = None
        self._vad: Any = None

    def _load_embedding_model(self) -> Any:
        if self._embedding_model is None:
            from funasr import AutoModel

            logger.info("Loading CAM++ speaker embedding model (FunASR, CPU)")
            self._embedding_model = AutoModel(
                model=self.EMBEDDING_MODEL, device="cpu", disable_update=True
            )
        return self._embedding_model

    def _load_vad(self) -> Any:
        if self._vad is None:
            from backend.engines.vad import FsmnVad

            self._vad = FsmnVad()
        return self._vad

    def _build_windows(self, audio: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
        spans = self._load_vad().detect(audio, sample_rate)
        windows: list[tuple[float, float]] = []
        for span in spans:
            t = span.start
            while t < span.end:
                w_end = min(t + self.WINDOW_SECONDS, span.end)
                if w_end - t >= self.MIN_WINDOW_SECONDS:
                    windows.append((t, w_end))
                t += self.HOP_SECONDS
        return windows

    def diarize(self, audio: np.ndarray, sample_rate: int) -> pd.DataFrame:
        from sklearn.cluster import AgglomerativeClustering

        windows = self._build_windows(audio, sample_rate)
        if not windows:
            return pd.DataFrame(columns=["start", "end", "speaker"])

        model = self._load_embedding_model()
        embeddings: list[np.ndarray] = []
        for start, end in windows:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            clip = audio[start_sample:end_sample]
            result = model.generate(input=clip, fs=sample_rate)
            embedding = result[0]["spk_embedding"].squeeze(0).numpy()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embeddings.append(embedding)

        embeddings_arr = np.stack(embeddings)

        # Known exact speaker count -> cluster to that count directly.
        # Otherwise -> cluster by distance threshold and let the count emerge.
        if settings.min_speakers and settings.min_speakers == settings.max_speakers:
            clustering = AgglomerativeClustering(
                n_clusters=settings.min_speakers, metric="cosine", linkage="average"
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.DISTANCE_THRESHOLD,
                metric="cosine",
                linkage="average",
            )
        labels = clustering.fit_predict(embeddings_arr)

        rows = [
            {"start": start, "end": end, "speaker": f"SPEAKER_{label:02d}"}
            for (start, end), label in zip(windows, labels)
        ]
        return pd.DataFrame(rows)


def get_diarization_engine(name: str) -> DiarizationEngine:
    if name == "campplusplus":
        return CamPlusPlusDiarizer()
    return PyannoteDiarizer()
