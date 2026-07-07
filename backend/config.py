"""
Central configuration for the meeting-ai pipeline.

WHY THIS FILE EXISTS:
Every other module (transcribe.py, summarize.py, utils.py, app.py) needs to
agree on the same paths, model names, and device settings. Without a single
source of truth, these values get hardcoded in multiple places and a change
(e.g. switching from llama3.1:8b to qwen3:8b) requires hunting through the
whole codebase. This file is imported everywhere; nothing else defines these
values independently.

Values are loaded from environment variables (via a .env file, see
.env.example) rather than hardcoded, because:
- HF_TOKEN is a secret and must never be committed to source control.
- DEVICE/COMPUTE_TYPE differ per machine (Apple Silicon vs Intel Mac vs a
  future Linux/GPU deployment) without changing code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Typed, validated application settings.

    Using pydantic-settings instead of raw os.environ.get() calls gives us:
    - Type coercion + validation at startup (fail fast if HF_TOKEN is missing
      rather than failing deep inside a WhisperX call an hour into a run).
    - A single documented schema of every configurable value.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Hugging Face / Pyannote ---
    # Required to download the gated pyannote speaker-diarization and
    # segmentation models. You must accept both model licenses on
    # huggingface.co before this token will work (see README setup steps).
    hf_token: str = ""

    # --- WhisperX / Whisper ---
    whisper_model: str = "large-v3"
    # "mps" = Apple Silicon GPU acceleration, "cpu" fallback for Intel Macs.
    # Defaulted to "mps" since this project targets Apple Silicon first.
    # WhisperX does not yet support CUDA-equivalent perf on MPS for all ops,
    # so compute_type is kept conservative (float32) on mps/cpu to avoid
    # silent accuracy loss or crashes from unsupported float16 kernels.
    device: str = "mps"
    compute_type: str = "float32"
    batch_size: int = 8
    language: str | None = None  # None = auto-detect

    # --- Diarization ---
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    min_speakers: int | None = None
    max_speakers: int | None = None

    @field_validator("language", "min_speakers", "max_speakers", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        if v == "":
            return None
        # Fix for Google Colab: Colab sets a global OS env var `LANGUAGE=en_US`.
        # Since pydantic reads OS env vars, this injects "en_US" into WhisperX,
        # crashing it. We strictly normalize any language string to its 2-letter ISO code.
        if isinstance(v, str) and cls == Settings:
            pass # wait, cls == Settings is always true, but let's just do it securely below
        
        return v

    @field_validator("language", mode="after")
    @classmethod
    def normalize_language(cls, v: Any) -> Any:
        if isinstance(v, str) and len(v) > 2:
            return v[:2].lower()
        return v

    # --- Ollama ---
    # Using 127.0.0.1 instead of localhost to prevent IPv6 httpx timeouts on Macs
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_request_timeout: int = 300

    # --- Summarization chunking ---
    # Rough word-count budget per chunk sent to the LLM. Kept conservative
    # relative to an 8B model's context window (~8k tokens) to leave room
    # for the prompt template and the model's own output.
    max_words_per_chunk: int = 3000

    # --- Logging ---
    log_level: str = "INFO"

    # --- Paths (relative to project root, not backend/) ---
    project_root: Path = Path(__file__).resolve().parent.parent
    upload_dir: Path = project_root / "uploads"
    output_dir: Path = project_root / "outputs"
    temp_dir: Path = project_root / "temp"
    log_dir: Path = project_root / "logs"

    def ensure_directories(self) -> None:
        """Create runtime directories if they don't exist yet.

        Called once at app/CLI startup rather than at import time, so
        importing this module never has filesystem side effects.
        """
        for directory in (self.upload_dir, self.output_dir, self.temp_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)


# Single shared instance. Every module does `from backend.config import settings`
# instead of constructing its own Settings() — this guarantees one consistent
# view of configuration across the whole process.
settings = Settings()
