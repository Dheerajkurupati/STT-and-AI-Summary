"""
Real-time live transcription pipeline.

TWO MODES (selected automatically based on GROQ_API_KEY in .env):

1. GROQ API MODE (recommended, no local models needed):
   Buffers audio -> saves temp WAV -> sends to Groq Whisper API -> returns text.
   Speaker tracking is basic (labels by chunk order) since Groq doesn't provide
   speaker embeddings. Works on any machine (Windows/Mac/Linux) with no GPU or
   special dependencies.

2. LOCAL MODE (original, requires faster-whisper + resemblyzer):
   Buffers audio -> faster-whisper transcription -> resemblyzer speaker embeddings.
   Requires local model downloads and compilation of native extensions.

WHY THIS IS SEPARATE FROM transcribe.py:
transcribe.py owns the batch path (file uploads). This file owns the live
streaming path (WebSocket). They share no state — a live session creates its
own audio buffer and state. This means:
- The batch pipeline is completely unaffected by live feature additions.
- Live sessions are fully isolated per WebSocket connection (concurrent calls safe)
- If we ever swap the live engine, only this file changes.
"""

from __future__ import annotations

import io
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from backend.config import settings
from backend.utils import get_logger

logger = get_logger(__name__)

SAMPLE_RATE = 16000  # Hz - what Whisper expects (matches AudioContext in browser)


@dataclass
class LiveChunk:
    """One transcribed + speaker-labelled chunk from the live stream."""

    speaker: str       # e.g. "Speaker 1"
    text: str
    timestamp: str     # session-relative, e.g. "01:24"
    start_seconds: float


def _float32_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert a float32 numpy array to an in-memory WAV file (PCM 16-bit)."""
    int16_audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    num_samples = len(int16_audio)
    data_size = num_samples * 2  # 2 bytes per int16 sample
    # Write WAV header
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))         # chunk size
    buf.write(struct.pack('<H', 1))          # PCM format
    buf.write(struct.pack('<H', 1))          # mono
    buf.write(struct.pack('<I', sample_rate))
    buf.write(struct.pack('<I', sample_rate * 2))  # byte rate
    buf.write(struct.pack('<H', 2))          # block align
    buf.write(struct.pack('<H', 16))         # bits per sample
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(int16_audio.tobytes())
    return buf.getvalue()


class LiveTranscriptionSession:
    """
    Stateful, per-WebSocket-connection live transcription + speaker tracking.

    Lifecycle:
      1. Created when a WebSocket connects  (app.py: /ws/live)
      2. process_chunk() called for every audio packet from the browser
      3. cleanup() called on disconnect

    Thread safety: not thread-safe - each connection gets its own instance
    and FastAPI runs each WebSocket handler in its own async context.
    """

    def __init__(self) -> None:
        self._use_groq = bool(settings.groq_api_key)
        self._whisper = None          # faster_whisper.WhisperModel (local mode only)
        self._voice_encoder = None    # resemblyzer.VoiceEncoder (local mode only)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_samples = int(SAMPLE_RATE * settings.live_buffer_seconds)
        self._overlap_samples = 0
        self._speaker_profiles: dict[str, np.ndarray] = {}
        self._speaker_counts: dict[str, int] = {}
        self._next_num = 1
        self._session_start = time.time()
        # Track previous text to avoid Groq returning the same text twice
        self._previous_texts: list[str] = []

        if self._use_groq:
            logger.info("Live session using Groq API mode (Model: %s)", settings.groq_whisper_model)
        else:
            logger.info("Live session using local mode (faster-whisper + resemblyzer)")

    # ------------------------------------------------------------------ #
    #  Lazy model loading (LOCAL MODE ONLY)                                #
    # ------------------------------------------------------------------ #

    def _load_whisper(self) -> None:
        if self._whisper is not None:
            return
        from faster_whisper import WhisperModel

        logger.info(
            "Loading faster-whisper model '%s' (live path)",
            settings.live_whisper_model,
        )
        self._whisper = WhisperModel(
            settings.live_whisper_model,
            device="cpu",        # CTranslate2 does not support MPS yet
            compute_type="int8", # fastest on CPU, negligible accuracy loss
        )

    def _load_encoder(self) -> None:
        if self._voice_encoder is not None:
            return
        from resemblyzer import VoiceEncoder

        logger.info("Loading resemblyzer VoiceEncoder (live speaker tracking)")
        self._voice_encoder = VoiceEncoder()

    # ------------------------------------------------------------------ #
    #  Text post-processing                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deduplicate(text: str) -> str:
        """
        Remove repeated phrases that Whisper sometimes hallucinates within a
        single transcription chunk.
        """
        words = text.split()
        if len(words) < 6:
            return text

        result: list[str] = []
        i = 0
        while i < len(words):
            found_repeat = False
            for n in range(6, 2, -1):
                if i + n * 2 <= len(words) + n:
                    window = words[i:i + n]
                    next_window = words[i + n:i + n * 2]
                    w1 = [w.lower().strip('.,!?;:\'"') for w in window]
                    w2 = [w.lower().strip('.,!?;:\'"') for w in next_window]
                    if len(w2) == n and w1 == w2:
                        result.extend(window)
                        i += n * 2
                        found_repeat = True
                        break
            if not found_repeat:
                result.append(words[i])
                i += 1

        return " ".join(result)

    # ------------------------------------------------------------------ #
    #  Speaker identification (LOCAL MODE ONLY)                            #
    # ------------------------------------------------------------------ #

    def _identify_speaker(self, audio: np.ndarray) -> str:
        """Return a consistent "Speaker N" label for the audio window."""
        try:
            from resemblyzer import preprocess_wav

            wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)
            embedding = self._voice_encoder.embed_utterance(wav)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            best_label: Optional[str] = None
            best_sim = -1.0

            for label, profile in self._speaker_profiles.items():
                sim = float(np.dot(embedding, profile))
                if sim > best_sim:
                    best_sim = sim
                    best_label = label

            if best_label and best_sim >= settings.live_similarity_threshold:
                n = self._speaker_counts[best_label]
                self._speaker_profiles[best_label] = (
                    (self._speaker_profiles[best_label] * n + embedding) / (n + 1)
                )
                self._speaker_counts[best_label] = n + 1
                return best_label

            label = f"Speaker {self._next_num}"
            self._next_num += 1
            self._speaker_profiles[label] = embedding
            self._speaker_counts[label] = 1
            logger.info("New speaker detected: %s (best_sim=%.3f)", label, best_sim)
            return label

        except Exception as exc:
            logger.warning("Speaker embedding failed: %s - labelling as Speaker 1", exc)
            return "Speaker 1"

    # ------------------------------------------------------------------ #
    #  GROQ API MODE — transcribe via Groq Whisper API                     #
    # ------------------------------------------------------------------ #

    def _transcribe_groq(self, window: np.ndarray) -> list[LiveChunk]:
        """Send buffered audio to Groq Whisper API and return LiveChunks."""
        from openai import OpenAI

        wav_bytes = _float32_to_wav_bytes(window)

        client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=settings.groq_request_timeout,
        )

        try:
            response = client.audio.transcriptions.create(
                file=("live_chunk.wav", wav_bytes, "audio/wav"),
                model=settings.groq_whisper_model,
                response_format="verbose_json",
                language=settings.language or None,  # None = auto-detect language
            )
        except Exception as exc:
            logger.error("Groq live transcription error: %s", exc)
            return []

        # Parse the response
        response_dict = getattr(response, "model_dump", lambda: None)()
        if response_dict is None:
            import json
            response_dict = json.loads(response) if isinstance(response, str) else {}

        text = response_dict.get("text", "").strip()
        if not text or len(text) < 3:
            return []

        text = self._deduplicate(text)



        # Skip if this is a duplicate of the previous chunk
        text_lower = text.lower().strip('.,!?;:\'" ')
        if text_lower in self._previous_texts:
            return []
        self._previous_texts.append(text_lower)
        # Keep only the last 5 texts for dedup
        if len(self._previous_texts) > 5:
            self._previous_texts = self._previous_texts[-5:]

        elapsed = time.time() - self._session_start
        mins, secs = divmod(int(elapsed), 60)

        # Use Resemblyzer voice embeddings to identify the speaker for this audio window
        self._load_encoder()
        speaker = self._identify_speaker(window)

        return [LiveChunk(
            speaker=speaker,
            text=text,
            timestamp=f"{mins:02d}:{secs:02d}",
            start_seconds=elapsed,
        )]

    # ------------------------------------------------------------------ #
    #  LOCAL MODE — transcribe via faster-whisper + resemblyzer             #
    # ------------------------------------------------------------------ #

    def _transcribe_local(self, window: np.ndarray) -> list[LiveChunk]:
        """Transcribe with local faster-whisper and identify speakers with resemblyzer."""
        try:
            live_language = settings.language if settings.language else None
            segments, _info = self._whisper.transcribe(
                window,
                language=live_language,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "threshold": 0.45,
                },
                condition_on_previous_text=False,
                repetition_penalty=1.3,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                beam_size=5,
            )
            segments = list(segments)
        except Exception as exc:
            logger.error("faster-whisper error: %s", exc)
            return []

        chunks: list[LiveChunk] = []
        elapsed_base = time.time() - self._session_start

        for seg in segments:
            seg_text = self._deduplicate(seg.text.strip())
            if not seg_text or len(seg_text) < 3:
                continue

            start_sample = int(seg.start * SAMPLE_RATE)
            end_sample = int(seg.end * SAMPLE_RATE)
            seg_audio = window[start_sample:min(end_sample, len(window))]

            if len(seg_audio) < SAMPLE_RATE:
                seg_audio = window

            speaker = self._identify_speaker(seg_audio)

            seg_elapsed = elapsed_base + seg.start
            mins, secs = divmod(int(seg_elapsed), 60)

            chunks.append(LiveChunk(
                speaker=speaker,
                text=seg_text,
                timestamp=f"{mins:02d}:{secs:02d}",
                start_seconds=seg_elapsed,
            ))

        return chunks

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def process_chunk(self, raw_bytes: bytes) -> list[LiveChunk]:
        """
        Accept raw int16 LE PCM bytes (16 kHz, mono) from the browser WebSocket.

        Returns a list of LiveChunks. Returns an empty list while still
        buffering or during pure silence.
        """
        if not self._use_groq:
            self._load_whisper()
        self._load_encoder()

        # 1. Decode bytes -> float32
        int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        float32 = int16.astype(np.float32) / 32768.0
        self._buffer = np.concatenate([self._buffer, float32])

        # 2. Wait until buffer is full
        if len(self._buffer) < self._buffer_samples:
            return []

        # 3. Extract window
        window = self._buffer[:self._buffer_samples].copy()
        self._buffer = self._buffer[self._buffer_samples - self._overlap_samples:]

        # 4. Route to the appropriate backend
        if self._use_groq:
            return self._transcribe_groq(window)
        else:
            return self._transcribe_local(window)

    # ------------------------------------------------------------------ #
    #  Cleanup                                                             #
    # ------------------------------------------------------------------ #

    def cleanup(self) -> None:
        """Release per-session state on WebSocket disconnect."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._speaker_profiles.clear()
        self._speaker_counts.clear()
        self._previous_texts.clear()
        logger.info(
            "Live session ended - %d speaker(s) tracked over %.0fs",
            self._next_num - 1,
            time.time() - self._session_start,
        )
