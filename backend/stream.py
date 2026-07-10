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

LIVE VAD BENCHMARK OPTION (version2):
LIVE_VAD_ENGINE=silero (default) is unchanged — faster-whisper's own
vad_filter=True handles silence internally, exactly as before this setting
existed. LIVE_VAD_ENGINE=fsmn adds a pre-filter: FunASR's FSMN-VAD (see
backend/engines/vad.py) checks the buffered window for speech BEFORE calling
faster-whisper at all, skipping the call entirely on silence. This is a
benchmark option for comparing VAD quality/latency, not a default change.
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
        self._whisper = None          # faster_whisper.WhisperModel, loaded lazily
        self._voice_encoder = None    # resemblyzer.VoiceEncoder, loaded lazily
        self._fsmn_vad = None         # backend.engines.vad.FsmnVad, loaded lazily (LIVE_VAD_ENGINE=fsmn only)
        self._buffer = np.zeros(0, dtype=np.float32)
        # Sliding Window Config
        self._emit_interval_samples = int(SAMPLE_RATE * settings.live_buffer_seconds)
        self._max_window_samples = int(SAMPLE_RATE * 15.0)
        self._unprocessed_samples = 0
        self._last_emitted_end_time = 0.0

        self._speaker_profiles: dict[str, np.ndarray] = {}   # label -> avg embedding
        self._speaker_counts: dict[str, int] = {}             # label -> sample count
        self._next_num = 1
        self._session_start = time.time()
        self._last_transcript = ""

    # ------------------------------------------------------------------ #
    #  Lazy model loading                                                  #
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

    def _load_fsmn_vad(self):
        if self._fsmn_vad is None:
            from backend.engines.vad import FsmnVad

            logger.info("Loading FSMN-VAD for live pre-filtering (LIVE_VAD_ENGINE=fsmn)")
            self._fsmn_vad = FsmnVad()
        return self._fsmn_vad

    def _load_encoder(self) -> None:
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

            # Force a minimum leniency of 0.75 to prevent 9-speaker fragmentation
            # during live cross-talk, while respecting user config if they set it even higher.
            threshold = max(settings.live_wespeaker_threshold, 0.75)
            
            if best_label and best_dist <= threshold:
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

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def process_chunk(self, raw_bytes: bytes) -> list[LiveChunk]:
        """
        Accept raw int16 LE PCM bytes (16 kHz, mono) from the browser WebSocket.

        Returns a list of LiveChunks — one per detected speech segment within
        the buffered window. Each segment gets its own speaker identification,
        so two speakers within the same 5-second buffer are correctly labelled.
        Returns an empty list while still buffering or during pure silence.

        The browser sends Int16Array buffers (from ScriptProcessorNode) that are
        already at 16 kHz because the AudioContext is created at that rate.
        """
        self._load_whisper()
        self._load_encoder()

        # 1. Decode bytes -> float32
        int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        float32 = int16.astype(np.float32) / 32768.0
        self._buffer = np.concatenate([self._buffer, float32])
        self._unprocessed_samples += len(float32)

        # 2. Wait until we have enough new audio (e.g. 5s)
        if self._unprocessed_samples < self._emit_interval_samples:
            return []
            
        self._unprocessed_samples = 0

        # 3. Extract up to 15 seconds of history for deep context
        window = self._buffer[-self._max_window_samples :]
        
        # Calculate exactly where this 15s window starts in the overall stream
        total_audio_seconds = len(self._buffer) / SAMPLE_RATE
        window_duration = len(window) / SAMPLE_RATE
        window_start_time = total_audio_seconds - window_duration

        # Benchmark pre-filter: when LIVE_VAD_ENGINE=fsmn, skip the faster-whisper
        # call entirely if FSMN-VAD finds no speech in this window. Default
        # ("silero") skips this and relies on faster-whisper's own vad_filter=True
        # below, exactly like before this option existed.
        if settings.live_vad_engine == "fsmn":
            try:
                if not self._load_fsmn_vad().has_speech(window, SAMPLE_RATE):
                    return []
            except Exception as exc:
                logger.warning("FSMN-VAD pre-filter failed, falling back to faster-whisper: %s", exc)

        try:
            live_language = settings.language if settings.language else "en"

            segments, _info = self._whisper.transcribe(
                window,
                language=live_language,
                vad_filter=True,
                vad_parameters={
                    # Relax VAD from 500ms to 2000ms. If a user takes a breath, we DO NOT 
                    # want to slice the sentence in half, otherwise large-v3 hallucinates!
                    "min_silence_duration_ms": 2000,
                    "threshold": 0.45,
                },
                # We turn off condition_on_previous_text because the 15s window
                # already contains massive context natively! This prevents hallucination loops.
                condition_on_previous_text=False,
                repetition_penalty=1.3,
                
                # --- Anti-Hallucination Enforcements ---
                # Force temperature to 0 to stop Whisper from "guessing" words
                temperature=0.0,
                # Reduce beam size to stop it from creatively building fake sentences
                beam_size=2,
                
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                word_timestamps=True,  # Crucial for sub-segment Diarization
            )
            segments = list(segments)
        except Exception as exc:
            logger.error("faster-whisper error: %s", exc)
            return []

        chunks: list[LiveChunk] = []

        for seg in segments:
            if not seg.words:
                continue

            # Group words into sentences by punctuation (.!?)
            sentence_words = []
            for i, w in enumerate(seg.words):
                sentence_words.append(w)
                text = w.word.strip()
                
                # If this word ends with a sentence terminator, or it's the last word in the segment
                if (text and text[-1] in ".!?") or i == len(seg.words) - 1:
                    if not sentence_words:
                        continue
                        
                    sentence_start = sentence_words[0].start
                    sentence_end = sentence_words[-1].end
                    sentence_text = "".join(w.word for w in sentence_words).strip()
                    
                    sentence_absolute_start = window_start_time + sentence_start
                    sentence_absolute_end = window_start_time + sentence_end
                    
                    # Deduplication: Only emit if this sentence started AFTER our last emitted sentence ENDED.
                    if sentence_absolute_start <= self._last_emitted_end_time - 0.3:
                        sentence_words = []
                        continue
                        
                    sentence_text = self._deduplicate(sentence_text)
                    if not sentence_text or len(sentence_text) < 3:
                        sentence_words = []
                        continue

                    # Extract audio strictly for this sentence for perfectly targeted Diarization
                    start_sample = int(sentence_start * SAMPLE_RATE)
                    end_sample   = int(sentence_end   * SAMPLE_RATE)
                    seg_audio = window[start_sample : min(end_sample, len(window))]

                    # --- Diarization Padding ---
                    # If the sentence is too short, expand bounds symmetrically to 1.5s
                    min_samples = int(1.5 * SAMPLE_RATE)
                    if len(seg_audio) < min_samples:
                        pad_needed = min_samples - len(seg_audio)
                        pad_left = pad_needed // 2
                        pad_right = pad_needed - pad_left
                        
                        s_idx = max(0, start_sample - pad_left)
                        e_idx = min(len(window), end_sample + pad_right)
                        seg_audio = window[s_idx:e_idx]

                    speaker = self._identify_speaker(seg_audio)
                    
                    mins, secs = divmod(int(sentence_absolute_start), 60)

                    chunks.append(LiveChunk(
                        speaker=speaker,
                        text=sentence_text,
                        timestamp=f"{mins:02d}:{secs:02d}",
                        start_seconds=sentence_absolute_start,
                    ))
                    
                    self._last_emitted_end_time = sentence_absolute_end
                    sentence_words = []

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
