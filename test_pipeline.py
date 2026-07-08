from pathlib import Path
from backend.transcribe import pipeline
from backend.utils import convert_to_wav
from backend.formatter import format_transcript, write_transcript_outputs
from backend.config import settings
from backend.summarize import summarizer, write_summary_outputs

def main():
    print("1. Converting/Verifying audio...")
    wav = convert_to_wav(Path('uploads/test_meeting.wav'))
    
    print("2. Transcribing and Diarizing with WhisperX and Pyannote (this might take a moment)...")
    result = pipeline.transcribe(wav)
    print(f"   Detected language: {result.language}, {len(result.segments)} segments found.")
    
    print("3. Formatting transcript...")
    transcript = format_transcript(result.segments)
    write_transcript_outputs(transcript, settings.output_dir)
    print("   Transcript saved to outputs/transcript.txt")
    
    print("4. Summarizing with Ollama...")
    summary = summarizer.summarize(transcript)
    write_summary_outputs(summary, settings.output_dir)
    print("   Summary saved to outputs/summary.txt")
    
    print("\n✅ All done! The pipeline is working perfectly.")

if __name__ == "__main__":
    main()
