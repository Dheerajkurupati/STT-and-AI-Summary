"""
Shared, model-agnostic utilities: logging setup and audio preprocessing.

WHY THIS FILE EXISTS:
transcribe.py should only know about WhisperX, and summarize.py should only
know about Ollama. Neither should know *how* an arbitrary MP3/M4A/MP4 gets
turned into the clean WAV format Whisper expects. That responsibility lives
here, isolated, so:
- We can swap the audio backend (ffmpeg subprocess vs pydub vs torchaudio)
  without touching transcription logic.
- We can unit test "does this reject a corrupt file" without loading any ML
  model.

WHY FFMPEG SPECIFICALLY (not pydub/torchaudio alone):
Whisper/WhisperX expects mono, 16kHz, 16-bit PCM WAV input. Input files
arrive in wildly inconsistent formats (stereo, 44.1/48kHz, variable bitrate
MP4 audio tracks). FFmpeg is the industry-standard, most format-tolerant
tool for this conversion and is what WhisperX's own examples rely on.
We shell out to the ffmpeg binary (installed via Homebrew) rather than using
a pure-Python decoder because ffmpeg has far broader codec support.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from pathlib import Path

from backend.config import settings

# Formats we explicitly support at the API boundary. Rejecting early with a
# clear error is better than letting ffmpeg fail deep inside a subprocess
# call with a cryptic exit code.
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4"}

# Whisper's expected input format. 16kHz mono is the sample rate the model
# was trained on; anything else works but wastes compute / can degrade
# alignment accuracy.
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1


class AudioProcessingError(Exception):
    """Raised when an input file is invalid or ffmpeg conversion fails."""


_logging_configured = False


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger writing to both console and logs/pipeline.log.

    Configured once per process (guarded by _logging_configured) so importing
    this from multiple modules doesn't attach duplicate handlers, which would
    otherwise print every log line multiple times.
    """
    global _logging_configured

    if not _logging_configured:
        settings.ensure_directories()
        log_file = settings.log_dir / "pipeline.log"

        root_logger = logging.getLogger()
        root_logger.setLevel(settings.log_level)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        _logging_configured = True

    return logging.getLogger(name)


logger = get_logger(__name__)


def validate_audio_file(file_path: Path) -> None:
    """
    Reject unsupported or missing files before any expensive processing runs.

    Why fail here instead of inside transcribe.py: this check has nothing to
    do with transcription — it's a generic input contract that every caller
    (FastAPI upload, CLI script, future batch job) should enforce identically.
    """
    if not file_path.exists():
        raise AudioProcessingError(f"Input file does not exist: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioProcessingError(
            f"Unsupported file type '{file_path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if file_path.stat().st_size == 0:
        raise AudioProcessingError(f"Input file is empty: {file_path}")


def convert_to_wav(input_path: Path, temp_dir: Path | None = None) -> Path:
    """
    Convert an arbitrary audio/video file into mono 16kHz WAV via ffmpeg.

    Returns the path to the converted file in temp/. Callers are responsible
    for cleanup (see cleanup_temp_file) since temp/ is meant to be safely
    purgeable scratch space, not permanent storage.
    """
    validate_audio_file(input_path)

    temp_dir = temp_dir or settings.temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)

    output_path = temp_dir / f"{uuid.uuid4().hex}.wav"

    command = [
        "ffmpeg",
        "-y",  # overwrite output if it somehow already exists
        "-i", str(input_path),
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-vn",  # drop any video stream (relevant for .mp4 input)
        str(output_path),
    ]

    logger.info("Converting %s -> %s (16kHz mono WAV)", input_path.name, output_path.name)

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise AudioProcessingError(
            "ffmpeg binary not found. Install it with `brew install ffmpeg`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise AudioProcessingError(
            f"ffmpeg failed to convert {input_path.name}: {exc.stderr.strip()}"
        ) from exc

    return output_path


def cleanup_temp_file(file_path: Path) -> None:
    """Best-effort removal of an intermediate temp/ file.

    Failures are logged, not raised — a leftover temp file should never
    fail an otherwise-successful transcription request.
    """
    try:
        file_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete temp file %s: %s", file_path, exc)
