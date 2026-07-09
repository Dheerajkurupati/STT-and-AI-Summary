"""
Hybrid API + Local integration: Groq API for STT + Pyannote for speaker diarization.

ARCHITECTURE:
  1. Audio Pre-Processing  – noisereduce removes background hiss; pyloudnorm
     normalises loudness so quiet speakers are transcribed accurately.
  2. Groq Whisper API      – Full-file transcription with word+segment timestamps.
  3. Pyannote diarization  – Identifies exactly who is speaking when.
  4. WhisperX merge        – Snaps every word to the correct speaker using the
     word-level timestamps from Groq and the boundaries from Pyannote.
  5. Speaker Smoothing     – Multi-pass smoothing fixes stray words that bled
     into the wrong speaker due to Groq's sloppy cross-talk timestamps.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import settings
from backend.formatter import RawSegment
from backend.utils import get_logger

logger = get_logger(__name__)

SAMPLE_RATE = 16000  # Hz – Whisper's native rate


class TranscriptionError(Exception):
    """Raised when audio loading, transcription, alignment, or diarization fails."""


@dataclass
class TranscriptionResult:
    """Everything downstream code (formatter.py) needs from a transcription run."""

    segments: list[RawSegment]
    language: str
    duration_seconds: float


class HybridPipeline:
    """
    Thin, stateful wrapper that handles:
    1. Audio Pre-Processing (noise reduction + normalization).
    2. Calling Groq API for text (STT) on the full file.
    3. Calling local Pyannote for speaker labels.
    4. Merging them using WhisperX's bounding-box logic.
    5. Multi-pass smoothing to fix cross-talk bleeding.
    """

    def __init__(self) -> None:
        self._diarize_pipeline: Any = None
        self._groq_client: Any = None

    def _get_groq_client(self) -> Any:
        if self._groq_client is None:
            if not settings.groq_api_key:
                raise TranscriptionError("GROQ_API_KEY is not set in your .env file.")
            from groq import Groq
            self._groq_client = Groq(api_key=settings.groq_api_key)
        return self._groq_client

    def _load_align_model(self, language_code: str) -> tuple[Any, Any]:
        # Only use Wav2Vec2 for lightweight languages supported by default.
        # This prevents crashing the server by downloading massive 4GB MMS models for Telugu/Hindi.
        supported_light_languages = {
            "en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt", "ar", 
            "ru", "pl", "hu", "fi", "fa", "el", "tr", "da", "he", "vi", "ko", "ur"
        }
        if language_code not in supported_light_languages:
            logger.info("Language '%s' is not lightweight. Skipping Wav2Vec2 alignment to save RAM.", language_code)
            return None, None

        if getattr(self, "_align_model", None) is None or getattr(self, "_align_language", None) != language_code:
            import whisperx

            logger.info("Loading lightweight alignment model for language='%s'", language_code)
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language_code, device=settings.device
            )
            self._align_language = language_code
        return self._align_model, self._align_metadata

    def _load_diarize_pipeline(self) -> Any:
        if self._diarize_pipeline is None:
            if not settings.hf_token:
                raise TranscriptionError(
                    "HF_TOKEN is not set. Diarization requires a Hugging Face "
                    "token with access to pyannote/speaker-diarization-3.1"
                )

            from whisperx.diarize import DiarizationPipeline

            logger.info("Loading local Pyannote diarization pipeline on %s...", settings.device)
            self._diarize_pipeline = DiarizationPipeline(
                model_name=settings.diarization_model,
                token=settings.hf_token,
                device=settings.device,
            )
        return self._diarize_pipeline

    @staticmethod
    def _preprocess_audio(audio: np.ndarray) -> np.ndarray:
        """
        Audio Pre-Processing pipeline:
        - Noise Reduction: noisereduce removes background hiss/hum/noise.
          Cleaner audio = fewer Whisper hallucinations and more accurate
          word-level timestamps during cross-talk.
        - Loudness Normalization: pyloudnorm normalises to -16 LUFS
          (broadcast standard). Ensures quiet speakers are not
          under-transcribed by Whisper.
        """
        try:
            import noisereduce as nr
            logger.info("Applying noise reduction...")
            noise_sample = audio[: SAMPLE_RATE // 2] if len(audio) > SAMPLE_RATE // 2 else audio
            audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, y_noise=noise_sample, prop_decrease=0.75)
            # noisereduce can return float64 — cast back to float32 which Pyannote requires
            audio = audio.astype(np.float32)
            logger.info("Noise reduction complete.")
        except Exception as exc:
            logger.warning("noisereduce failed (skipping): %s", exc)

        try:
            import pyloudnorm as pyln
            logger.info("Normalizing loudness to -16 LUFS...")
            meter = pyln.Meter(SAMPLE_RATE)
            loudness = meter.integrated_loudness(audio)
            if loudness > -70:
                audio = pyln.normalize.loudness(audio, loudness, -16.0)
            # Ensure float32 after normalization too
            audio = audio.astype(np.float32)
            logger.info("Loudness normalization complete.")
        except Exception as exc:
            logger.warning("pyloudnorm failed (skipping): %s", exc)

        return audio

    @staticmethod
    def _smooth_speakers(words: list[dict]) -> None:
        """
        Single-pass speaker smoothing (V1 logic).
        Only fixes 1-word bleeds (orphan words where left and right neighbors match).
        Does not steamroll longer 2-3 word interjections!
        """
        for i in range(1, len(words) - 1):
            curr = words[i]
            prev_word = words[i - 1]
            next_word = words[i + 1]

            prev_speaker = prev_word.get("speaker")
            next_speaker = next_word.get("speaker")

            if prev_speaker and next_speaker and prev_speaker == next_speaker:
                if curr.get("speaker") != prev_speaker:
                    gap_before = curr.get("start", 0) - prev_word.get("end", 0)
                    gap_after = next_word.get("start", 0) - curr.get("end", 0)
                    if gap_before < 0.6 and gap_after < 0.6:
                        curr["speaker"] = prev_speaker

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

            # Apply multi-pass smoothing before grouping
            cls._smooth_speakers(seg["words"])

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
                    current_start = word_obj.get("start", seg.get("start", 0.0))
                    current_words.append(word_text)
                elif current_speaker == word_speaker:
                    current_words.append(word_text)
                else:
                    segments.append(RawSegment(
                        start=float(current_start),
                        end=float(word_obj.get("start", seg.get("end", 0.0))),
                        text=" ".join(current_words),
                        speaker=current_speaker,
                    ))
                    current_speaker = word_speaker
                    current_start = word_obj.get("start", seg.get("start", 0.0))
                    current_words = [word_text]

            if current_words and current_speaker is not None:
                segments.append(RawSegment(
                    start=float(current_start),
                    end=float(seg.get("end", 0.0)),
                    text=" ".join(current_words),
                    speaker=current_speaker,
                ))

        return segments

    def _correct_diarization_with_llm(self, segments: list[RawSegment]) -> list[RawSegment]:
        """
        Uses Groq's Llama model to correct severe diarization failures (e.g. 15-second
        blocks of 4 people yelling that Pyannote merged into a single speaker).
        The LLM adds missing punctuation and re-assigns the speakers semantically.
        """
        client = self._get_groq_client()
        
        # Build prompt payload
        transcript_lines = []
        for s in segments:
            transcript_lines.append(f"[{s.start:.1f} - {s.end:.1f}] {s.speaker}: {s.text}")
            
        transcript_text = "\n".join(transcript_lines)
        
        system_prompt = (
            "You are an expert transcript editor. The provided transcript has blocks of rapid cross-talk "
            "where multiple people arguing were incorrectly merged into a single speaker turn without punctuation.\n"
            "Your task is to fix this by breaking large, unpunctuated blocks into distinct sentences with proper "
            "punctuation, and assigning them to alternating speakers (e.g. SPEAKER_00, SPEAKER_01) if they are clearly arguing.\n"
            "CRITICAL: The audio will ONLY contain English, Telugu, or Hindi (or a mix of them). DO NOT TRANSLATE. "
            "Preserve all native scripts (తెలుగు, हिंदी, etc.) exactly as they were provided.\n"
            "Keep the exact same timestamp prefix for any new lines you create from a block.\n"
            "DO NOT change, add, or remove any spoken words. ONLY adjust punctuation and speaker tags.\n"
            "Output ONLY the corrected transcript in the exact same format, nothing else."
        )
        
        logger.info("Sending transcript to Groq %s for Diarization Correction...", settings.groq_llm_model)
        try:
            response = client.chat.completions.create(
                model=settings.groq_llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript_text}
                ],
                temperature=0.0
            )
            corrected_text = response.choices[0].message.content.strip()
            
            corrected_segments = []
            import re
            pattern = re.compile(r"^\[([\d\.]+) - ([\d\.]+)\]\s+([^:]+):\s+(.*)$")
            
            for line in corrected_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                match = pattern.match(line)
                if match:
                    start = float(match.group(1))
                    end = float(match.group(2))
                    speaker = match.group(3).strip()
                    text = match.group(4).strip()
                    corrected_segments.append(RawSegment(start=start, end=end, text=text, speaker=speaker))
                else:
                    logger.warning("LLM output line did not match format: %s", line)
                    
            if corrected_segments:
                return corrected_segments
        except Exception as exc:
            logger.error("LLM Diarization correction failed (falling back to original): %s", exc)
            
        return segments

    @staticmethod
    def _semantic_boundary_snap(words: list[dict]) -> None:
        """
        Fixes millisecond-level speaker bleeding by mathematically snapping Pyannote 
        boundaries to the nearest terminal punctuation (sentences/clauses).
        """
        terminals = {'.', '?', '!'}
        for w in words:
            text = w.get('word', '').strip()
            w['is_terminal'] = any(text.endswith(t) for t in terminals)

        i = 1
        while i < len(words):
            prev_speaker = words[i-1].get('speaker')
            curr_speaker = words[i].get('speaker')
            
            if prev_speaker and curr_speaker and prev_speaker != curr_speaker:
                # Search left for terminal
                left_dist = None
                for j in range(1, 10):
                    if i - j >= 0 and words[i - j].get('is_terminal'):
                        left_dist = j
                        break
                
                # Search right for terminal
                right_dist = None
                for j in range(0, 10):
                    if i + j < len(words) and words[i + j].get('is_terminal'):
                        right_dist = j
                        break

                if left_dist is not None and right_dist is not None:
                    # Snap to whichever is closer
                    if left_dist <= right_dist + 1:
                        for k in range(i - left_dist + 1, i):
                            words[k]['speaker'] = curr_speaker
                    else:
                        for k in range(i, i + right_dist + 1):
                            words[k]['speaker'] = prev_speaker
                    i += max(left_dist, right_dist) + 1
                    continue
                elif left_dist is not None and left_dist <= 8:
                    for k in range(i - left_dist + 1, i):
                        words[k]['speaker'] = curr_speaker
                    i += 1
                    continue
                elif right_dist is not None and right_dist <= 8:
                    for k in range(i, i + right_dist + 1):
                        words[k]['speaker'] = prev_speaker
                    i += right_dist + 1
                    continue
            i += 1

    @staticmethod
    def _rebuild_segments_from_words(words: list[dict]) -> list[RawSegment]:
        """
        Reconstructs the RawSegment objects from the flat list of words after 
        speaker assignments and semantic snapping have been finalized.
        """
        segments = []
        if not words:
            return segments
            
        current_speaker = words[0].get('speaker', 'SPEAKER_UNKNOWN')
        current_words = [words[0]]
        
        for w in words[1:]:
            speaker = w.get('speaker', 'SPEAKER_UNKNOWN')
            gap = w.get('start', 0.0) - current_words[-1].get('end', 0.0)
            
            if speaker != current_speaker or gap > 2.0:
                text = " ".join(cw.get('word', '').strip() for cw in current_words)
                segments.append(RawSegment(
                    start=float(current_words[0].get('start', 0.0)),
                    end=float(current_words[-1].get('end', 0.0)),
                    text=text.strip(),
                    speaker=current_speaker
                ))
                current_speaker = speaker
                current_words = [w]
            else:
                current_words.append(w)
                
        if current_words:
            text = " ".join(cw.get('word', '').strip() for cw in current_words)
            segments.append(RawSegment(
                start=float(current_words[0].get('start', 0.0)),
                end=float(current_words[-1].get('end', 0.0)),
                text=text.strip(),
                speaker=current_speaker
            ))
            
        return segments
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        import soundfile as sf
        import whisperx

        client = self._get_groq_client()

        # 1. Load Audio
        logger.info("Loading audio...")
        try:
            audio = whisperx.load_audio(str(audio_path))
        except Exception as exc:
            raise TranscriptionError(f"Failed to load audio {audio_path}: {exc}") from exc

        # 2. Audio Pre-Processing (noise reduction + loudness normalization)
        audio = self._preprocess_audio(audio)

        # 3. Send the FULL pre-processed file to Groq
        # Full-file context prevents Whisper from hallucinating during cross-talk.
        logger.info("Sending pre-processed audio to Groq API (%s)...", settings.groq_stt_model)
        try:
            buffer = io.BytesIO()
            sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
            buffer.seek(0)

            api_params: dict[str, Any] = {
                "model": settings.groq_stt_model,
                "response_format": "verbose_json",
                "timestamp_granularities": ["word", "segment"],
                "prompt": "The audio will ONLY contain English, Telugu, or Hindi (or a mix of them). Transcribe exactly what is spoken in the native script. Do not translate."
            }
            if getattr(settings, "language", None):
                api_params["language"] = settings.language

            transcription = client.audio.transcriptions.create(
                file=("audio.wav", buffer.read()),
                **api_params
            )
        except Exception as exc:
            raise TranscriptionError(f"Groq API failed: {exc}") from exc

        # Extract data from Groq response
        detected_language = getattr(transcription, "language", "en")
        if len(detected_language) > 2:
            detected_language = detected_language[:2].lower()

        duration = getattr(transcription, "duration", 0.0)
        
        # We don't need Groq's sloppy word timestamps anymore.
        # Just take the raw segments and pass them to Wav2Vec2 for perfect alignment!
        groq_segments = [
            {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": str(s.get("text", ""))}
            for s in list(getattr(transcription, "segments", None) or [])
        ]

        # 4. Align timestamps perfectly using Wav2Vec2
        logger.info("Aligning timestamps using Wav2Vec2 (fixes Groq's sloppy boundaries)...")
        align_model, align_metadata = self._load_align_model(detected_language)
        
        if align_model is not None:
            try:
                # whisperx.align returns a dict with "segments" and "word_segments"
                aligned = whisperx.align(
                    groq_segments, align_model, align_metadata, audio, settings.device, return_char_alignments=False
                )
            except Exception as exc:
                logger.error("Wav2Vec2 alignment failed (falling back to rough Groq timestamps): %s", exc)
                aligned = {"segments": groq_segments}
        else:
            aligned = {"segments": groq_segments}

        # 5. Run Pyannote Diarization on the pre-processed audio
        logger.info("Running local Pyannote diarization...")
        diarize_pipeline = self._load_diarize_pipeline()
        diarization = diarize_pipeline(
            audio,
            min_speakers=settings.min_speakers,
            max_speakers=settings.max_speakers,
        )

        # 6. Assign speakers to words using WhisperX bounding-box logic
        logger.info("Assigning speakers to words...")
        result = whisperx.assign_word_speakers(diarization, aligned)

        # 7. Apply Semantic Snapping & Rebuild
        logger.info("Applying semantic punctuation snapping to fix boundary bleeds...")
        flat_words = result.get("word_segments", [])
        if flat_words:
            self._semantic_boundary_snap(flat_words)
            self._smooth_speakers(flat_words)
            segments = self._rebuild_segments_from_words(flat_words)
        else:
            logger.warning("No word_segments returned! Falling back to raw segments.")
            segments = self._to_raw_segments(result["segments"])
            
        logger.info("Transcript ready with %d segments.", len(segments))

        # 8. LLM Diarization Correction (Final Polish)
        segments = self._correct_diarization_with_llm(segments)

        return TranscriptionResult(
            segments=segments,
            language=detected_language,
            duration_seconds=duration,
        )


# Single shared instance exported for the application
pipeline = HybridPipeline()


