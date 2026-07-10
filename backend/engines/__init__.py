"""
Pluggable pipeline-stage engines (version2).

WHY THIS PACKAGE EXISTS:
transcribe.py orchestrates the upload pipeline but should not need to know the
internals of every candidate VAD/STT/diarization model — only that each stage
exposes one small, typed entry point. Each module here owns exactly one model
family (mirroring the isolation principle transcribe.py already applies to
WhisperX and summarize.py applies to Ollama): if a specific engine's library
changes its API, only that one file changes.

Every engine is selected by name via backend.config.settings (vad_engine,
stt_engine, diarization_engine, live_vad_engine) and constructed lazily by the
get_*_engine() factory in each module — nothing here loads a model at import
time.
"""
