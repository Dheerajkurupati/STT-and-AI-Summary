# meeting-ai

Local, production-oriented meeting transcription pipeline: audio in,
speaker-diarized Google Meet-style transcript + AI summary out.

Pipeline: FFmpeg preprocessing -> WhisperX (Whisper large-v3 + alignment +
pyannote diarization) -> transcript formatting -> Ollama (llama3.1:8b /
qwen3:8b) summarization.

## 🚀 Quick Start Guide (From Zip File)

Follow these steps in order to set up and run the pipeline locally on your machine (e.g., using **VS Code** or **Antigravity IDE**). 
*(Note: This project is meant to be run in a local IDE environment, not in cloud notebook environments like Google Colab).*

### 1. Unzip and Open in your IDE
1. Extract the zip file you received.
2. Open the unzipped folder directly in your IDE (**VS Code** or **Antigravity**).
3. Open a new Integrated Terminal within your IDE. You should automatically be in the correct project folder.

### 2. Install System Dependencies
**For Windows:**
Install FFmpeg using Winget (in Command Prompt or PowerShell):
`winget install "FFmpeg (CLI)"`
*(Alternatively, download from [ffmpeg.org](https://ffmpeg.org/) and add it to your system PATH).*

**For macOS:**
1. **Homebrew** — if not already installed:
   `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. **FFmpeg**
   `brew install ffmpeg`

**For Linux / Cloud GPU:**
`sudo apt update && sudo apt install ffmpeg`

### 3. Set Up Python Environment
Ensure you have Python 3.11 installed. Create and activate a virtual environment:

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

Then, install dependencies:
```bash
pip install --upgrade pip

# For Windows / macOS / CPU:
pip install -r requirements.txt

# For Linux/Windows with NVIDIA GPU (CUDA 12.1):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install -r requirements.txt
```

### 4. Configure Environment & Hugging Face Token
1. **Copy the environment file**:
   - **macOS / Linux**: `cp .env.example .env`
   - **Windows**: `copy .env.example .env`
2. **Create a "Read" token**: Visit [Hugging Face Tokens](https://huggingface.co/settings/tokens)
3. **Edit `.env`**: Open the `.env` file and set `HF_TOKEN=<your token>`
   *(For Cloud GPU users, also set `DEVICE=cuda` and `COMPUTE_TYPE=float16` if supported).*

### 5. Accept Pyannote Model Licenses
*(Required — diarization will fail without this, even with a valid token).* 
While logged into Hugging Face, visit and accept the license on **both** of these pages:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

### 6. Install & Configure Ollama
1. **Install Ollama**:
   - **Windows**: Download and install from [ollama.com/download/windows](https://ollama.com/download/windows)
   - **macOS**: `brew install ollama`
   - **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
2. **Start the Ollama server**:
   ```bash
   ollama serve
   ```
   *(Leave this running in its own terminal window, or let the Ollama menu-bar app manage it on macOS)*
3. **Pull the AI Model** (in a new terminal window, make sure to navigate to the project directory):
   ```bash
   ollama pull llama3.1:8b
   ```
   *(You can also use `qwen3:8b`. Set whichever you choose as `OLLAMA_MODEL` in `.env`.)*

### 7. Run the Application
Ensure your virtual environment is activated (e.g. `source .venv/bin/activate` or `.venv\Scripts\activate`), then start the FastAPI server:
```bash
uvicorn backend.app:app --reload
# Or for Linux/Cloud: uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### 8. Process an Audio File (Getting Output)
With the server running, you can test the pipeline by submitting an audio file. Open a new terminal window and run:

**macOS / Linux / Windows (Command Prompt / Git Bash):**
```bash
curl -X POST http://localhost:8000/transcribe -F "file=@uploads/your_test_file.mp3"
```
*(Replace `uploads/your_test_file.mp3` with the actual path to an audio or video file on your system. Note: On Windows PowerShell, `curl` may act differently; if so, please use standard Command Prompt).*

**Viewing the Output:**
Once the process finishes, the pipeline will generate several files in the `outputs/` directory in your project folder. You will find:
- `transcript.txt` and `transcript.json` (Speaker diarized transcript)
- `summary.txt` and `summary.json` (AI-generated meeting summary)

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
