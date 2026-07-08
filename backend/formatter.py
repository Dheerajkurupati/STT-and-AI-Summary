"""
Pure presentation logic: turns raw WhisperX/pyannote output into a
Google Meet-style transcript.

WHY THIS FILE EXISTS AND IS SEPARATE FROM transcribe.py:
transcribe.py's job ends the moment we have timestamped, speaker-labeled
segments. Everything below is formatting/business logic with zero
dependency on ML models — it takes plain data in and plain data out. Keeping
it separate means:
- We can unit test "does Speaker_2 correctly become 'Speaker 3'" and "are
  adjacent same-speaker segments merged correctly" without ever loading
  Whisper or pyannote (fast tests, no GPU/CPU cost).
- If we ever change the transcription engine, this file doesn't change at
  all — it only cares about the segment data shape, not where it came from.

WHY WE MERGE ADJACENT SEGMENTS:
Pyannote diarization + WhisperX often splits a single continuous sentence
from one speaker into several short segments (e.g. due to brief pauses).
Google Meet's transcript UI shows one block per speaker "turn", not one
block per micro-segment, so we merge consecutive segments from the same
speaker when the gap between them is small.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Segments from a same speaker separated by less than this many seconds are
# merged into a single transcript block. Chosen to bridge natural speech
# pauses (breathing, short pauses) without merging across an actual speaker
# hand-off that happens to be quick.
MERGE_GAP_SECONDS = 1.5


@dataclass
class RawSegment:
    """
    Mirrors the shape WhisperX returns per segment after diarization:
    {"start": float, "end": float, "text": str, "speaker": str}.
    Declared explicitly here (rather than passing raw dicts around) so
    formatter.py has a typed, documented contract with transcribe.py.
    """

    start: float
    end: float
    text: str
    speaker: str  # e.g. "SPEAKER_00" — raw pyannote label


@dataclass
class TranscriptBlock:
    """One Google Meet-style block: a timestamp, a speaker, and their text."""

    speaker_label: str  # e.g. "Speaker 1"
    timestamp: str  # e.g. "00:06" or "01:02:06" for long meetings
    start_seconds: float
    end_seconds: float
    text: str


@dataclass
class MeetingTranscript:
    """The full formatted transcript plus metadata worth keeping around."""

    blocks: list[TranscriptBlock] = field(default_factory=list)
    speaker_count: int = 0
    duration_seconds: float = 0.0


def _format_timestamp(seconds: float) -> str:
    """Render seconds as MM:SS, or HH:MM:SS once a meeting passes an hour."""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _build_speaker_label_map(segments: list[RawSegment]) -> dict[str, str]:
    """
    Map raw diarization labels (SPEAKER_00, SPEAKER_01, ...) to display
    labels (Speaker 1, Speaker 2, ...) in order of first appearance.

    Order-of-first-appearance (not alphabetical/numeric sort of the raw
    label) matches how Google Meet assigns speaker identity: whoever talks
    first is "Speaker 1", regardless of pyannote's internal numbering.
    """
    label_map: dict[str, str] = {}
    next_index = 1
    for segment in segments:
        if segment.speaker not in label_map:
            label_map[segment.speaker] = f"Speaker {next_index}"
            next_index += 1
    return label_map


def format_transcript(segments: list[RawSegment]) -> MeetingTranscript:
    """
    Convert raw diarized segments into a merged, display-ready transcript.

    This is the single entry point formatter.py exposes — transcribe.py (or
    app.py) calls this once with the full segment list from WhisperX.
    """
    if not segments:
        return MeetingTranscript()

    label_map = _build_speaker_label_map(segments)

    blocks: list[TranscriptBlock] = []
    for segment in segments:
        speaker_label = label_map[segment.speaker]
        text = segment.text.strip()
        if not text:
            continue

        can_merge = (
            blocks
            and blocks[-1].speaker_label == speaker_label
            and (segment.start - blocks[-1].end_seconds) <= MERGE_GAP_SECONDS
        )

        if can_merge:
            last = blocks[-1]
            last.text = f"{last.text} {text}"
            last.end_seconds = segment.end
        else:
            blocks.append(
                TranscriptBlock(
                    speaker_label=speaker_label,
                    timestamp=_format_timestamp(segment.start),
                    start_seconds=segment.start,
                    end_seconds=segment.end,
                    text=text,
                )
            )

    duration = max((s.end for s in segments), default=0.0)

    return MeetingTranscript(
        blocks=blocks,
        speaker_count=len(label_map),
        duration_seconds=duration,
    )


def transcript_to_plain_text(transcript: MeetingTranscript) -> str:
    """
    Render as the Google Meet-style TXT format:

        00:00

        Speaker 1

        Good morning everyone.

    Blank lines between fields are intentional — this matches the expected
    output format specified for this project.
    """
    lines: list[str] = []
    for block in transcript.blocks:
        lines.extend([block.timestamp, "", block.speaker_label, "", block.text, ""])
    return "\n".join(lines).rstrip() + "\n"


def transcript_to_dict(transcript: MeetingTranscript) -> dict:
    """JSON-serializable representation, used for outputs/transcript.json."""
    return {
        "speaker_count": transcript.speaker_count,
        "duration_seconds": transcript.duration_seconds,
        "blocks": [asdict(block) for block in transcript.blocks],
    }


def write_transcript_outputs(transcript: MeetingTranscript, output_dir: Path) -> tuple[Path, Path]:
    """
    Persist both required output formats (JSON + TXT) as specified in the
    project structure. Returns (json_path, txt_path) for the caller (app.py)
    to reference in an API response.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "transcript.json"
    txt_path = output_dir / "transcript.txt"

    json_path.write_text(json.dumps(transcript_to_dict(transcript), indent=2), encoding="utf-8")
    txt_path.write_text(transcript_to_plain_text(transcript), encoding="utf-8")

    return json_path, txt_path
