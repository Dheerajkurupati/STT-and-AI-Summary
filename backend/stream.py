"""
Real-time live transcription pipeline: faster-whisper + resemblyzer speaker embeddings.

WHY THIS IS SEPARATE FROM transcribe.py:
transcribe.py owns the batch WhisperX path (file uploads). This file owns the
live streaming path (WebSocket). They share no state — a live session creates its
own model instances, its own audio buffer, its own speaker memory. This means:
- The batch pipeline is completely unaffected by live feature additions.
- Live sessions are fully isolated per WebSocket connection (concurrent calls safe)
- If we ever swap the live engine, only this file changes.

HOW SPEAKER TRACKING WORKS:
resemblyzer computes a 256-dimensional "d-vector" (voice embedding) for each audio
chunk. We compare each new embedding against stored profiles using cosine similarity.
If similarity >= threshold -> same speaker (update running average). Otherwise -> new
speaker, assign "Speaker N" and store embedding. This gives consistent labels
throughout the session without requiring the full audio up front.

WHY faster-whisper INSTEAD OF whisperx HERE:
WhisperX is great for batch (it adds alignment + diarization on top of Whisper), but
for live chunks we need raw speed. faster-whisper (CTranslate2 backend) is
4x faster than original Whisper on CPU.

KEY IMPROVEMENTS OVER V1:
- Buffer increased from 3s -> 5s for more context.
- Pyannote Wespeaker replaces Resemblyzer for robust per-segment identification (handles interruptions properly).
- repetition_penalty + no_repeat_ngram_size added to kill Whisper hallucinations.
- language locked to "en" per chunk (was auto-detecting, causing Korean/Dutch hallucinations).
- log_prob_threshold + compression_ratio_threshold filter out low-confidence garbage.
- condition_on_previous_text=False prevents Whisper from copying its own previous output.
- Minimum text length guard (< 3 chars) to skip single-character phantom outputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
        self._voice_encoder: Any = None
        self._groq_client: Any = None
        
        # Audio buffer (raw float32 samples at 16kHz)
        self._buffer = np.zeros(0, dtype=np.float32)
        
        # We need ~5 seconds of audio to get a reliable chunk for Groq & Wespeaker
        self._buffer_samples = 5 * SAMPLE_RATE
        # Keep 1 second overlap so words at chunk boundaries aren't lost
        self._overlap_samples = 1 * SAMPLE_RATE

        # Speaker tracking
        self._speaker_profiles: dict[str, np.ndarray] = {}
        self._speaker_counts: dict[str, int] = {}
        self._next_num = 1
        self._session_start = time.time()

    # ------------------------------------------------------------------ #
    #  Model lazy-loading                                                  #
    # ------------------------------------------------------------------ #

    def _load_encoder(self) -> None:
        """Pyannote Wespeaker for robust 256-d voice embeddings."""
        if self._voice_encoder is not None:
            return
        import torch
        from pyannote.audio import Model, Inference

        logger.info("Loading Pyannote Wespeaker (live speaker tracking)")
        model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
        self._voice_encoder = Inference(model, window="whole", device=torch.device("cpu"))

    # ------------------------------------------------------------------ #
    #  Text post-processing                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deduplicate(text: str) -> str:
        """
        Remove repeated phrases that Whisper sometimes hallucinates within a
        single transcription chunk, e.g.:
          "You're going to have You're going to have a lot" -> "You're going to have a lot"
          "incorporate that. of how you incorporate that." -> "incorporate that."

        Algorithm: scan over the text word-by-word. For each position, test
        whether the next N words repeat the previous N words (for N=3..6).
        If they do, skip the duplicated block.
        """
        words = text.split()
        if len(words) < 6:
            return text

        result: list[str] = []
        i = 0
        while i < len(words):
            # Try to detect repeats of lengths 6, 5, 4, 3 (longest first)
            found_repeat = False
            for n in range(6, 2, -1):
                if i + n * 2 <= len(words) + n:  # enough words ahead to check
                    window = words[i:i + n]
                    next_window = words[i + n:i + n * 2]
                    # Case-insensitive comparison, strip punctuation for matching
                    w1 = [w.lower().strip('.,!?;:\'"') for w in window]
                    w2 = [w.lower().strip('.,!?;:\'"') for w in next_window]
                    if len(w2) == n and w1 == w2:
                        # Duplicate detected: emit first occurrence, skip second
                        result.extend(window)
                        i += n * 2  # jump past both copies
                        found_repeat = True
                        break
            if not found_repeat:
                result.append(words[i])
                i += 1

        return " ".join(result)

    # ------------------------------------------------------------------ #
    #  Speaker identification                                              #
    # ------------------------------------------------------------------ #

    def _identify_speaker(self, audio: np.ndarray) -> str:
        """
        Return a consistent "Speaker N" label for the audio window.

        Steps:
          1. Compute 256-d voice embedding with Pyannote Wespeaker.
          2. Cosine-distance compare against stored profiles.
          3. Below threshold -> known speaker, update running average.
          4. Above threshold -> new speaker, register profile.
        """
        import torch

        try:
            # Pyannote expects a dict with waveform tensor (channel, time) and sample rate
            tensor = torch.from_numpy(audio).unsqueeze(0)
            
            # Extract embedding - returns a numpy array of shape (256,)
            embedding = self._voice_encoder({"waveform": tensor, "sample_rate": SAMPLE_RATE})
            
            # Normalize embedding just in case to make cosine logic simple
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            best_label: Optional[str] = None
            best_dist = float('inf')

            for label, profile in self._speaker_profiles.items():
                # Cosine distance = 1 - cosine similarity
                sim = float(np.dot(embedding, profile))
                dist = 1.0 - sim
                if dist < best_dist:
                    best_dist = dist
                    best_label = label

            if best_label and best_dist <= settings.live_wespeaker_threshold:
                # Known speaker - update running average so the profile adapts
                n = self._speaker_counts[best_label]
                self._speaker_profiles[best_label] = (
                    (self._speaker_profiles[best_label] * n + embedding) / (n + 1)
                )
                self._speaker_counts[best_label] = n + 1
                return best_label

            # New speaker
            label = f"Speaker {self._next_num}"
            self._next_num += 1
            self._speaker_profiles[label] = embedding
            self._speaker_counts[label] = 1
            logger.info("New speaker detected: %s (best_dist=%.3f)", label, best_dist)
            return label

        except Exception as exc:
            # If embedding fails (very short audio, silence, etc.), fall back gracefully
            logger.warning("Speaker embedding failed: %s - labelling as Speaker 1", exc)
            return "Speaker 1"

    def _get_groq_client(self) -> Any:
        if getattr(self, "_groq_client", None) is None:
            if not settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set in your .env file.")
            from groq import Groq
            self._groq_client = Groq(api_key=settings.groq_api_key)
        return self._groq_client

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def process_chunk(self, raw_bytes: bytes) -> list[LiveChunk]:
        import io
        import soundfile as sf
        
        self._load_encoder()
        client = self._get_groq_client()

        # 1. Decode bytes -> float32
        int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        float32 = int16.astype(np.float32) / 32768.0
        self._buffer = np.concatenate([self._buffer, float32])

        # 2. Wait until buffer is full
        if len(self._buffer) < self._buffer_samples:
            return []

        # 3. Extract window, keep overlap
        window = self._buffer[: self._buffer_samples].copy()
        # Keep last 1s so a sentence spanning a chunk boundary is not cut
        self._buffer = self._buffer[self._buffer_samples - self._overlap_samples :]

        # 4. Transcribe with Groq API
        try:
            live_language = settings.language if settings.language else "en"
            
            # Write 5s audio chunk to memory
            buffer = io.BytesIO()
            sf.write(buffer, window, SAMPLE_RATE, format="WAV")
            buffer.seek(0)
            
            # Call Groq API
            api_params = {
                "model": settings.groq_stt_model,
                "response_format": "verbose_json",
                # Low temperature reduces hallucinations in live chunks
                "temperature": 0.0,
                "prompt": "The audio will ONLY contain English, Telugu, or Hindi (or a mix of them). Transcribe exactly what is spoken in the native script. Do not translate."
            }
            if getattr(settings, "language", None):
                api_params["language"] = settings.language
            
            transcription = client.audio.transcriptions.create(
                file=("live_chunk.wav", buffer.read()),
                **api_params
            )
            
            # Extract segments
            segments = list(getattr(transcription, "segments", None) or [])
        except Exception as exc:
            logger.error("Groq API error during live stream: %s", exc)
            return []

        # Build one LiveChunk per segment, each with its own speaker ID.
        chunks: list[LiveChunk] = []
        elapsed_base = time.time() - self._session_start

        for seg in segments:
            # Groq segments are dicts
            start_sec = float(seg.get("start", 0.0))
            end_sec = float(seg.get("end", 0.0))
            seg_text = self._deduplicate(seg.get("text", "").strip())

            if not seg_text or len(seg_text) < 3:
                continue

            # Extract the audio slice for this specific VAD segment
            start_sample = int(start_sec * SAMPLE_RATE)
            end_sample   = int(end_sec * SAMPLE_RATE)
            seg_audio = window[start_sample : min(end_sample, len(window))]

            # resemblyzer needs at least ~1s of audio for a reliable embedding.
            # If this VAD segment is shorter, fall back to the full window
            # (still better than no identification).
            if len(seg_audio) < SAMPLE_RATE:
                seg_audio = window

            speaker = self._identify_speaker(seg_audio)

            seg_elapsed = elapsed_base + start_sec
            mins, secs = divmod(int(seg_elapsed), 60)

            chunks.append(LiveChunk(
                speaker=speaker,
                text=seg_text,
                timestamp=f"{mins:02d}:{secs:02d}",
                start_seconds=seg_elapsed,
            ))

        return chunks

    # ------------------------------------------------------------------ #
    #  Cleanup                                                             #
    # ------------------------------------------------------------------ #

    def cleanup(self) -> None:
        """Release per-session state on WebSocket disconnect."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._speaker_profiles.clear()
        self._speaker_counts.clear()
        logger.info(
            "Live session ended - %d speaker(s) tracked over %.0fs",
            self._next_num - 1,
            time.time() - self._session_start,
        )
