"""
Command-line interface for transcribing large audio files.
Bypasses the browser's 5-minute timeout limit, making it ideal for slow CPUs.
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
    args = parser.parse_args()

    input_path = Path(args.audio_file)
    if not input_path.exists():
        print(f"Error: File not found -> {input_path}")
        sys.exit(1)

    print(f"\n🎙️  Starting processing for: {input_path.name}")
    print(f"⚙️  Using model: {settings.whisper_model} (Device: {settings.device})")
    print("⏳ This may take several minutes (roughly 1-2x the audio duration on CPU)...\n")

    wav_path = None
    try:
        # 1. Convert
        wav_path = convert_to_wav(input_path)
        
        # 2. Transcribe & Diarize
        transcription = pipeline.transcribe(wav_path)
        
        # 3. Format
        transcript = format_transcript(transcription.segments)
        
        # 4. Save
        json_path, txt_path = write_transcript_outputs(transcript, settings.output_dir)
        
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
