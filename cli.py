"""
Command-line interface for transcribing large audio files.
Bypasses the browser's 5-minute timeout limit, making it ideal for slow CPUs.

BENCHMARK FLAGS (version2):
--vad/--stt/--diarization override the corresponding ENGINE setting for this
run only (falling back to .env/defaults when omitted) — see backend/config.py's
vad_engine/stt_engine/diarization_engine. Overrides are applied to `settings`
before the first call to pipeline.transcribe(), which is required: each engine
is chosen once and cached on first use (see backend/transcribe.py), so this
only works if set before that first call. Running the same file twice with
different flags writes to distinctly-named outputs instead of overwriting each
other, so two runs can be compared side by side — e.g.:
    python cli.py uploads/meeting_3_speakers.wav --diarization pyannote
    python cli.py uploads/meeting_3_speakers.wav --diarization campplusplus
"""

import argparse
import sys
from pathlib import Path

from backend.config import settings
from backend.formatter import format_transcript, write_transcript_outputs
from backend.transcribe import pipeline
from backend.utils import convert_to_wav, cleanup_temp_file

def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file locally.")
    parser.add_argument("audio_file", type=str, help="Path to the audio or video file")
    parser.add_argument("--vad", choices=["none", "fsmn"], help="Override VAD_ENGINE for this run")
    parser.add_argument("--stt", choices=["whisperx", "sensevoice"], help="Override STT_ENGINE for this run")
    parser.add_argument(
        "--diarization",
        choices=["pyannote", "campplusplus"],
        help="Override DIARIZATION_ENGINE for this run",
    )
    args = parser.parse_args()

    input_path = Path(args.audio_file)
    if not input_path.exists():
        print(f"Error: File not found -> {input_path}")
        sys.exit(1)

    # Must happen before pipeline.transcribe() is called for the first time —
    # each engine is constructed once from `settings` and cached (see
    # WhisperXPipeline._get_*_engine in transcribe.py).
    if args.vad:
        settings.vad_engine = args.vad
    if args.stt:
        settings.stt_engine = args.stt
    if args.diarization:
        settings.diarization_engine = args.diarization

    print(f"\n🎙️  Starting processing for: {input_path.name}")
    print(f"⚙️  Engines: vad={settings.vad_engine} stt={settings.stt_engine} diarization={settings.diarization_engine}")
    print("⏳ This may take several minutes (roughly 1-2x the audio duration on CPU)...\n")

    wav_path = None
    try:
        # 1. Convert
        wav_path = convert_to_wav(input_path)

        # 2. Transcribe & Diarize
        transcription = pipeline.transcribe(wav_path)

        # 3. Format
        transcript = format_transcript(transcription.segments)

        # 4. Save — suffix with the engine combo when any override is set, so
        # benchmark runs don't overwrite each other or the default output.
        output_dir = settings.output_dir
        if args.vad or args.stt or args.diarization:
            stem = input_path.stem
            suffix = f"{stem}.{settings.stt_engine}.{settings.diarization_engine}.{settings.vad_engine}"
            json_path = output_dir / f"{suffix}.transcript.json"
            txt_path = output_dir / f"{suffix}.transcript.txt"
            output_dir.mkdir(parents=True, exist_ok=True)
            import json as json_module
            from backend.formatter import transcript_to_dict, transcript_to_plain_text
            json_path.write_text(json_module.dumps(transcript_to_dict(transcript), indent=2), encoding="utf-8")
            txt_path.write_text(transcript_to_plain_text(transcript), encoding="utf-8")
        else:
            json_path, txt_path = write_transcript_outputs(transcript, output_dir)

        print("\n✅ Transcription Complete!")
        print(f"📝 Text file saved to: {txt_path}")
        print(f"📊 JSON file saved to: {json_path}")

    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
    finally:
        if wav_path:
            cleanup_temp_file(wav_path)

if __name__ == "__main__":
    main()
