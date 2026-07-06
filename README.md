# meeting-ai

Local, production-oriented meeting transcription pipeline: audio in,
speaker-diarized Google Meet-style transcript + AI summary out.

Pipeline: FFmpeg preprocessing -> WhisperX (Whisper large-v3 + alignment +
pyannote diarization) -> transcript formatting -> Ollama (llama3.1:8b /
qwen3:8b) summarization.

## Setup (macOS)

1. **Homebrew** — if not already installed:
   `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

2. **FFmpeg**
   `brew install ffmpeg`

3. **Python 3.11 virtual environment**
   ```
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Hugging Face token**
   - Create a "Read" token: https://huggingface.co/settings/tokens
   - Copy `.env.example` to `.env` and set `HF_TOKEN=<your token>`

5. **Accept pyannote model licenses** (required — diarization will fail
   without this, even with a valid token). While logged into Hugging Face,
   visit and accept the license on both:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

6. **Install Ollama**
   `brew install ollama`
   then start the server: `ollama serve` (leave running in its own terminal,
   or let the Ollama menu-bar app manage it)

7. **Pull a model**
   `ollama pull llama3.1:8b` (or `ollama pull qwen3:8b`)
   Set whichever you choose as `OLLAMA_MODEL` in `.env`.

8. **First run — sanity check the pipeline directly (no API yet)**
   ```
   python -c "
   from pathlib import Path
   from backend.transcribe import pipeline
   from backend.utils import convert_to_wav

   wav = convert_to_wav(Path('uploads/your_test_file.mp3'))
   result = pipeline.transcribe(wav)
   print(result.language, len(result.segments), 'segments')
   "
   ```

9. **Generate a transcript**
   ```
   python -c "
   from backend.formatter import format_transcript, write_transcript_outputs
   from backend.config import settings
   transcript = format_transcript(result.segments)
   write_transcript_outputs(transcript, settings.output_dir)
   "
   ```

10. **Generate the AI summary**
    ```
    python -c "
    from backend.summarize import summarizer, write_summary_outputs
    from backend.config import settings
    summary = summarizer.summarize(transcript)
    write_summary_outputs(summary, settings.output_dir)
    "
    ```

11. **Run via FastAPI**
    ```
    uvicorn backend.app:app --reload
    ```
    Then:
    ```
    curl -X POST http://localhost:8000/transcribe \
      -F "file=@uploads/your_test_file.mp3"
    ```

## Project structure

```
backend/
  config.py      settings (paths, model names, device, HF/Ollama config)
  utils.py       logging setup, ffmpeg audio conversion, validation
  transcribe.py  WhisperX: transcription + alignment + diarization
  formatter.py   raw segments -> Google Meet-style transcript
  prompts.py     Ollama prompt templates
  summarize.py   Ollama: chunked summarization -> structured summary
  app.py         FastAPI endpoints (thin orchestration only)
uploads/         raw user-submitted audio (persisted)
outputs/         transcript.json, transcript.txt, summary.json, summary.txt
temp/            intermediate WAV files (safe to purge)
logs/            pipeline.log
```

## Notes

- `DEVICE`/`COMPUTE_TYPE` in `.env`: use `mps`/`float32` on Apple Silicon,
  `cpu`/`float32` on Intel Macs. Do not use `float16` unless you've verified
  your WhisperX version supports it on your hardware without crashing.
- Long transcripts are automatically chunked before summarization
  (`MAX_WORDS_PER_CHUNK` in `.env`) with a final pass that merges chunk
  summaries into one coherent result — see `backend/summarize.py`.
