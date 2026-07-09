"""
WhisperX integration: transcription + word alignment + speaker diarization.

WHY THIS FILE OWNS ALL OF WHISPERX/PYANNOTE:
This is the only module that imports whisperx (and, transitively,
pyannote.audio). Every other module talks to this one through the plain
TranscriptionResult/RawSegment data shapes, never through WhisperX's own
types. That means:
- If WhisperX changes its API (it has, across versions), only this file
  needs to change.
- If we ever swap the engine entirely (e.g. a hosted transcription API),
  only this file changes — formatter.py, summarize.py, and app.py are
  unaffected because they only depend on RawSegment/TranscriptionResult.

WHY MODELS ARE LOADED LAZILY AND CACHED ON THE INSTANCE:
Whisper large-v3 and the diarization pipeline are multi-GB models that take
real time to load from disk. In a FastAPI server handling multiple
requests, we load them once (at first use) and reuse the same instance for
every subsequent request, rather than reloading per-request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.formatter import RawSegment
from backend.utils import get_logger

logger = get_logger(__name__)


class TranscriptionError(Exception):
    """Raised when audio loading, transcription, alignment, or diarization fails."""


@dataclass
class TranscriptionResult:
    """Everything downstream code (formatter.py) needs from a transcription run."""

    segments: list[RawSegment]
    language: str
    duration_seconds: float


class WhisperXPipeline:
    """
    Thin, stateful wrapper around WhisperX's three stages: transcribe,
    align, diarize. One instance is created and reused for the lifetime of
    the process (see app.py's startup hook).
    """

    def __init__(self) -> None:
        self._whisper_model: Any = None
        self._align_model: Any = None
        self._align_metadata: Any = None
        self._align_language: str | None = None
        self._diarize_pipeline: Any = None

    def _load_whisper_model(self) -> Any:
        if self._whisper_model is None:
            # Imported lazily, not at module load time, so importing
            # backend.transcribe (e.g. for type checking or from app.py's
            # module graph) doesn't force torch/whisperx to load immediately.
            import whisperx

            logger.info(
                "Loading Whisper model '%s' on device=%s compute_type=%s",
                settings.whisper_model,
                settings.device,
                settings.compute_type,
            )
            self._whisper_model = whisperx.load_model(
                settings.whisper_model,
                device=settings.device,
                compute_type=settings.compute_type,
            )
        return self._whisper_model

    def _load_align_model(self, language_code: str) -> tuple[Any, Any]:
        # Alignment models are language-specific, so we reload only if the
        # detected language changes between requests (rare, but possible
        # with multilingual meetings processed back to back).
        if self._align_model is None or self._align_language != language_code:
            import whisperx

            logger.info("Loading alignment model for language='%s'", language_code)
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
                    "token with access to pyannote/speaker-diarization-3.1 "
                    "(see README setup steps)."
                )

            import whisperx

            logger.info("Loading pyannote diarization pipeline")
            self._diarize_pipeline = whisperx.diarize.DiarizationPipeline(
                model_name=settings.diarization_model,
                token=settings.hf_token,
                device=settings.device,
            )
        return self._diarize_pipeline

    def _diarize_with_pyannote(self, audio_path: Path, segments: list[RawSegment]) -> list[RawSegment]:
        if len(segments) <= 1:
            return segments

        import whisperx
        
        logger.info("Performing Pyannote audio-based diarization to label Groq segments")
        try:
            # Load audio for pyannote
            audio = whisperx.load_audio(str(audio_path))
            
            # Run diarization pipeline
            diarize_pipeline = self._load_diarize_pipeline()
            diarization = diarize_pipeline(
                audio,
                min_speakers=settings.min_speakers,
                max_speakers=settings.max_speakers,
            )
            
            # Match pyannote output (pandas DataFrame) with Groq text segments based on time overlap
            for seg in segments:
                seg_start = seg.start
                seg_end = seg.end
                best_speaker = "Speaker 1"
                max_overlap = 0.0
                
                for _, row in diarization.iterrows():
                    d_start = row["start"]
                    d_end = row["end"]
                    
                    # Calculate intersection
                    overlap = max(0, min(seg_end, d_end) - max(seg_start, d_start))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        
                        # Convert SPEAKER_00 to Speaker 1, etc.
                        speaker_label = row["speaker"]
                        if speaker_label.startswith("SPEAKER_"):
                            try:
                                num = int(speaker_label.split("_")[1]) + 1
                                best_speaker = f"Speaker {num}"
                            except ValueError:
                                best_speaker = speaker_label
                        else:
                            best_speaker = speaker_label
                            
                if max_overlap > 0:
                    seg.speaker = best_speaker

            unique_speakers = set(seg.speaker for seg in segments)
            logger.info(f"Pyannote diarization complete: {len(unique_speakers)} unique speakers detected")
            
        except Exception as exc:
            logger.warning("Pyannote diarization failed: %s.", exc)
            raise

    def _diarize_with_resemblyzer(self, audio_path: Path, segments: list[RawSegment]) -> list[RawSegment]:
        if len(segments) <= 1:
            return segments

        import soundfile as sf
        import numpy as np
        from resemblyzer import VoiceEncoder, preprocess_wav
        from sklearn.cluster import AgglomerativeClustering

        logger.info("Performing Resemblyzer audio-based diarization to label Groq segments")
        try:
            # Load audio data
            wav_data, sr = sf.read(str(audio_path))
            duration = len(wav_data) / sr

            # Bin-based clustering to get stable embeddings
            bin_duration = 5.0
            bins = []
            t = 0.0
            while t < duration:
                end = min(t + bin_duration, duration)
                bins.append((t, end))
                t += bin_duration

            # Initialize VoiceEncoder
            encoder = VoiceEncoder()

            embeddings = []
            valid_bins = []
            for idx, (start_t, end_t) in enumerate(bins):
                start_sample = int(start_t * sr)
                end_sample = int(end_t * sr)
                audio_slice = wav_data[start_sample:end_sample]

                if len(audio_slice) < 16000:  # Needs at least 1s of audio to be stable
                    continue

                processed = preprocess_wav(audio_slice, source_sr=sr)
                embed = encoder.embed_utterance(processed)
                embeddings.append(embed)
                valid_bins.append(idx)

            if not embeddings:
                logger.warning("No valid speaker embeddings extracted. Defaulting all to Speaker 1.")
                return segments

            embeddings = np.array(embeddings)
            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_norm = embeddings / (norms + 1e-8)

            # Determine clustering constraints
            min_speakers = settings.min_speakers
            max_speakers = settings.max_speakers

            n_clusters = None
            distance_threshold = 0.25  # Cosine distance threshold (tuned for Resemblyzer speaker verification)

            # If min_speakers and max_speakers are equal and set, enforce that exact number of clusters
            if min_speakers and min_speakers == max_speakers:
                n_clusters = min_speakers
                distance_threshold = None

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average"
            )
            labels = clustering.fit_predict(embeddings_norm)
            n_speakers = len(set(labels))

            # Enforce speaker bounds if n_clusters was not hardcoded
            if n_clusters is None:
                if min_speakers and n_speakers < min_speakers:
                    logger.info(f"Detected {n_speakers} speakers, forcing min_speakers={min_speakers}")
                    clustering = AgglomerativeClustering(
                        n_clusters=min_speakers,
                        metric="cosine",
                        linkage="average"
                    )
                    labels = clustering.fit_predict(embeddings_norm)
                elif max_speakers and n_speakers > max_speakers:
                    logger.info(f"Detected {n_speakers} speakers, forcing max_speakers={max_speakers}")
                    clustering = AgglomerativeClustering(
                        n_clusters=max_speakers,
                        metric="cosine",
                        linkage="average"
                    )
                    labels = clustering.fit_predict(embeddings_norm)

            # Map valid bin index to cluster label
            bin_to_label = {bin_idx: label for bin_idx, label in zip(valid_bins, labels)}

            # Map cluster labels to Speaker 1, Speaker 2, etc. in order of appearance
            speaker_map = {}
            next_speaker_num = 1

            # Match segments to bins and assign speakers
            for seg in segments:
                # Find overlapping bins and sum up their overlap duration
                best_label = None
                max_overlap = 0.0

                for bin_idx, (b_start, b_end) in enumerate(bins):
                    if bin_idx not in bin_to_label:
                        continue
                    overlap = max(0, min(seg.end, b_end) - max(seg.start, b_start))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_label = bin_to_label[bin_idx]

                if best_label is None:
                    best_label = labels[0] if len(labels) > 0 else 0

                if best_label not in speaker_map:
                    speaker_map[best_label] = f"Speaker {next_speaker_num}"
                    next_speaker_num += 1

                seg.speaker = speaker_map[best_label]

            unique_speakers = set(seg.speaker for seg in segments)
            logger.info("Resemblyzer diarization complete: %d unique speakers detected", len(unique_speakers))

        except Exception as exc:
            logger.error("Resemblyzer diarization failed: %s. Defaulting all to Speaker 1.", exc, exc_info=True)

        return segments

    def _diarize_with_llm(self, segments: list[RawSegment]) -> list[RawSegment]:
        if len(segments) <= 1:
            return segments

        logger.info("Performing LLM-based semantic diarization fallback")
        from backend.prompts import LLM_DIARIZATION_PROMPT_TEMPLATE, SYSTEM_PROMPT        
        lines = []
        for i, seg in enumerate(segments, 1):
            lines.append(f"{i}: {seg.text}")
            
        prompt = LLM_DIARIZATION_PROMPT_TEMPLATE.format(content="\n".join(lines))

        try:
            if settings.groq_api_key:
                from openai import OpenAI
                client = OpenAI(
                    api_key=settings.groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                    timeout=settings.groq_request_timeout,
                )
                response = client.chat.completions.create(
                    model=settings.groq_model,
                    messages=[
                        {"role": "system", "content": "You are a data processing API. Output ONLY plain text mappings (e.g. '1: Speaker 1'). Do not include explanations."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                )
                content = response.choices[0].message.content
            else:
                import ollama
                client = ollama.Client(host=settings.ollama_host, timeout=settings.ollama_request_timeout)
                response = client.chat(
                    model=settings.ollama_model,
                    messages=[
                        {"role": "system", "content": "You are a data processing API. Output ONLY plain text mappings (e.g. '1: Speaker 1'). Do not include explanations."},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = response["message"]["content"]
            
            # Parse the numbered lines
            parsed_count = 0
            for line in content.splitlines():
                line = line.strip()
                if ":" in line:
                    parts = line.split(":", 1)
                    num_str = parts[0].strip()
                    speaker = parts[1].strip().replace("*", "").replace("`", "")
                    if num_str.isdigit():
                        idx = int(num_str) - 1
                        if 0 <= idx < len(segments):
                            segments[idx].speaker = speaker
                            parsed_count += 1
            
            unique_speakers = set(seg.speaker for seg in segments)
            logger.info("LLM diarization complete: %d unique speakers detected across %d mapped segments", len(unique_speakers), parsed_count)

        except Exception as exc:
            logger.error("LLM diarization failed: %s", exc)

        return segments

    def _diarize(self, audio_path: Path, segments: list[RawSegment]) -> list[RawSegment]:
        result = segments
        if settings.hf_token:
            logger.info("HF_TOKEN is set. Attempting Pyannote diarization first.")
            try:
                result = self._diarize_with_pyannote(audio_path, segments)
            except Exception as exc:
                logger.warning("Pyannote diarization failed: %s. Falling back to Resemblyzer.", exc)
                result = self._diarize_with_resemblyzer(audio_path, segments)
        else:
            result = self._diarize_with_resemblyzer(audio_path, segments)
            
        # Check if audio-based diarization failed to find multiple speakers
        unique_speakers = set(seg.speaker for seg in result)
        if len(unique_speakers) <= 1 and len(result) > 1:
            logger.warning("Audio-based diarization detected only 1 speaker. Cascading to LLM fallback.")
            result = self._diarize_with_llm(result)
            
        return result

    def _transcribe_groq(self, audio_path: Path) -> TranscriptionResult:
        from openai import OpenAI
        from backend.formatter import RawSegment
        import json

        logger.info("Using Groq API for batch transcription (Model: %s)", settings.groq_whisper_model)
        client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=settings.groq_request_timeout,
        )

        try:
            with open(audio_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    file=(audio_path.name, audio_file, "audio/wav"),
                    model=settings.groq_whisper_model,
                    response_format="verbose_json",
                    language=settings.language if settings.language else None,
                    timestamp_granularities=["word"],
                )
        except Exception as exc:
            raise TranscriptionError(f"Groq transcription API failed: {exc}") from exc

        response_dict = getattr(response, "model_dump", lambda: None)() or response
        if isinstance(response_dict, str):
            response_dict = json.loads(response_dict)

        raw_segments_list = response_dict.get("segments", [])
        words_list = response_dict.get("words", [])
        detected_language = response_dict.get("language", "en")
        if len(detected_language) > 2:
            detected_language = detected_language[:2].lower()

        segments: list[RawSegment] = []

        if words_list:
            logger.info("Reconstructing sentence-level segments from word-level timestamps for precise diarization")
            current_words = []
            for i, w in enumerate(words_list):
                current_words.append(w)
                word_text = w.get("word", "")

                # Check for segment boundary conditions:
                # 1. Word ends with sentence punctuation (. ? !)
                # 2. Significant silence/gap before next word
                # 3. Maximum word count reached to keep segments short
                is_sentence_end = any(word_text.endswith(p) for p in [".", "?", "!"])
                
                has_gap = False
                if i < len(words_list) - 1:
                    next_w = words_list[i+1]
                    gap = next_w.get("start", 0) - w.get("end", 0)
                    if gap > 1.0:
                        has_gap = True

                if is_sentence_end or has_gap or len(current_words) >= 20:
                    text = " ".join([cw.get("word", "") for cw in current_words]).strip()
                    if text:
                        segments.append(
                            RawSegment(
                                start=float(current_words[0]["start"]),
                                end=float(current_words[-1]["end"]),
                                text=text,
                                speaker="Speaker 1",
                            )
                        )
                    current_words = []

            # Add any trailing words
            if current_words:
                text = " ".join([cw.get("word", "") for cw in current_words]).strip()
                if text:
                    segments.append(
                        RawSegment(
                            start=float(current_words[0]["start"]),
                            end=float(current_words[-1]["end"]),
                            text=text,
                            speaker="Speaker 1",
                        )
                    )
        else:
            # Fallback to standard segments if words list is not present
            logger.warning("No word-level timestamps returned by Groq. Falling back to default segments.")
            for seg in raw_segments_list:
                text = (seg.get("text") or "").strip()
                if not text:
                    continue
                segments.append(
                    RawSegment(
                        start=float(seg["start"]),
                        end=float(seg["end"]),
                        text=text,
                        speaker="Speaker 1",
                    )
                )

        # Apply diarization (Pyannote or Resemblyzer fallback)
        segments = self._diarize(audio_path, segments)

        duration = response_dict.get("duration", max((s.end for s in segments), default=0.0))

        return TranscriptionResult(
            segments=segments,
            language=detected_language,
            duration_seconds=duration,
        )

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """
        Run the transcription pipeline on a preprocessed (mono, 16kHz WAV)
        audio file. Uses Groq API if settings.groq_api_key is set, otherwise WhisperX.
        """
        if settings.groq_api_key:
            return self._transcribe_groq(audio_path)

        import whisperx

        try:
            audio = whisperx.load_audio(str(audio_path))
        except Exception as exc:  # whisperx/ffmpeg-backed loader raises broadly
            raise TranscriptionError(f"Failed to load audio {audio_path}: {exc}") from exc

        try:
            whisper_model = self._load_whisper_model()
            transcription = whisper_model.transcribe(
                audio,
                batch_size=settings.batch_size,
                language=settings.language,
            )
            detected_language = transcription["language"]
            # Whisper large-v3 sometimes returns variants like 'en_US'
            # but the alignment model expects strict 2-letter ISO codes.
            if len(detected_language) > 2:
                detected_language = detected_language[:2].lower()
            logger.info("Detected language (normalized): %s", detected_language)

            align_model, align_metadata = self._load_align_model(detected_language)
            aligned = whisperx.align(
                transcription["segments"],
                align_model,
                align_metadata,
                audio,
                settings.device,
                return_char_alignments=False,
            )

            if settings.hf_token:
                try:
                    diarize_pipeline = self._load_diarize_pipeline()
                    diarization = diarize_pipeline(
                        audio,
                        min_speakers=settings.min_speakers,
                        max_speakers=settings.max_speakers,
                    )
                    result = whisperx.assign_word_speakers(diarization, aligned)
                    segments = self._to_raw_segments(result["segments"])
                except Exception as exc:
                    logger.warning("Local Pyannote diarization failed: %s. Falling back to Resemblyzer.", exc)
                    segments = self._to_raw_segments(aligned["segments"])
                    segments = self._diarize_with_resemblyzer(audio_path, segments)
            else:
                logger.info("HF_TOKEN not set. Using Resemblyzer for local diarization fallback.")
                segments = self._to_raw_segments(aligned["segments"])
                segments = self._diarize_with_resemblyzer(audio_path, segments)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"WhisperX pipeline failed on {audio_path}: {exc}") from exc
        duration = max((s.end for s in segments), default=0.0)

        return TranscriptionResult(
            segments=segments,
            language=detected_language,
            duration_seconds=duration,
        )

    @staticmethod
    def _to_raw_segments(whisperx_segments: list[dict]) -> list[RawSegment]:
        """
        Convert WhisperX's raw segment dicts into our typed RawSegment
        contract. A segment can be missing a "speaker" key if diarization
        couldn't confidently assign one (e.g. cross-talk) — we label those
        "SPEAKER_UNKNOWN" rather than dropping the text, so no speech is
        silently lost from the transcript.
        """
        segments: list[RawSegment] = []
        for seg in whisperx_segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                RawSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=text,
                    speaker=seg.get("speaker", "SPEAKER_UNKNOWN"),
                )
            )
        return segments


# Single shared instance, analogous to `settings` in config.py — app.py
# imports this rather than constructing a new WhisperXPipeline per request,
# so multi-GB models are loaded once per process, not once per upload.
pipeline = WhisperXPipeline()
