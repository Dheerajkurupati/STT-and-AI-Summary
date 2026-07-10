"""
Ollama integration: turns a formatted transcript into a structured meeting
summary (executive summary, key topics, action items, decisions, risks,
next steps).

WHY THIS FILE OWNS ALL OLLAMA INTERACTION:
Same isolation principle as transcribe.py owning WhisperX: nothing else in
the codebase needs to know we're using Ollama specifically, or how chunking
works. If we ever swap to a different local/hosted LLM, only this file
changes.

WHY CHUNKING EXISTS:
An 8B model (llama3.1:8b / qwen3:8b) has a limited context window (~8k
tokens by default in Ollama). A one-hour meeting transcript can easily
exceed that. Rather than truncating (losing the back half of the meeting)
we split the transcript into word-budgeted chunks, summarize each chunk
independently, then run one final pass that merges the chunk summaries into
a single deduplicated, coherent summary. Short transcripts skip chunking
entirely and go straight to the final pass — no reason to pay for two LLM
calls when one fits.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.formatter import MeetingTranscript
from backend.prompts import (
    SYSTEM_PROMPT,
    build_chunk_summary_prompt,
    build_final_summary_prompt,
)
from backend.utils import get_logger

logger = get_logger(__name__)


class SummarizationError(Exception):
    """Raised when Ollama is unreachable or returns unparseable output."""


@dataclass
class SummaryResult:
    executive_summary: str = ""
    key_topics: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)


def _transcript_to_llm_text(transcript: MeetingTranscript) -> list[str]:
    """
    Render transcript blocks as plain "Speaker N: text" lines for LLM input.

    Deliberately drops timestamps present in the human-facing TXT output
    (see formatter.transcript_to_plain_text) — timestamps add tokens the
    model doesn't need to extract meaning, and every token here has a real
    cost against the context window budget.
    """
    return [f"{block.speaker_label}: {block.text}" for block in transcript.blocks]


def _split_into_chunks(lines: list[str], max_words: int) -> list[str]:
    """
    Group transcript lines into chunks under a word budget, splitting only
    at line boundaries so a sentence is never cut mid-way. If a single line
    alone exceeds max_words (an unusually long uninterrupted turn), it
    becomes its own oversized chunk rather than being split further —
    accepted as a rare edge case.
    """
    chunks: list[str] = []
    current_lines: list[str] = []
    current_words = 0

    for line in lines:
        word_count = len(line.split())
        if current_lines and current_words + word_count > max_words:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_words = 0
        current_lines.append(line)
        current_words += word_count

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


class SummarizerService:
    """Stateful wrapper around the Ollama client, reused across requests."""

    def __init__(self) -> None:
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import ollama

            self._client = ollama.Client(
                host=settings.ollama_host,
                timeout=settings.ollama_request_timeout,
            )
        return self._client

    def _call_ollama_json(self, prompt: str) -> dict:
        """
        Send one prompt to Ollama and parse the response as JSON.

        format="json" constrains the model's output to valid JSON syntax
        (an Ollama server-side feature), but it does NOT guarantee the keys
        match our schema — the model could still return {} or unexpected
        fields. Callers are responsible for defaulting missing keys.

        WHY think=False: qwen3:8b (the version2 default) is a hybrid-reasoning
        model that by default prepends a chain-of-thought "thinking" trace
        before its actual answer. That's wasted latency for a fixed-schema
        JSON extraction task — confirmed during version2 testing that leaving
        thinking on caused real summarization calls to exceed the 300s
        request timeout on this CPU-only machine. think=False is a no-op for
        models without reasoning support (e.g. llama3.1:8b), so this is safe
        regardless of which OLLAMA_MODEL is configured.
        """
        client = self._get_client()
        try:
            response = client.chat(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                format="json",
                think=False,
            )
        except Exception as exc:
            raise SummarizationError(
                f"Failed to reach Ollama at {settings.ollama_host} "
                f"(is `ollama serve` running and is '{settings.ollama_model}' pulled?): {exc}"
            ) from exc

        content = response["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise SummarizationError(
                f"Ollama returned non-JSON content despite format='json': {content[:200]}"
            ) from exc

    def _summarize_chunk(self, chunk_text: str, index: int, total: int) -> dict:
        logger.info("Summarizing chunk %d/%d", index, total)
        prompt = build_chunk_summary_prompt(chunk_text, index, total)
        return self._call_ollama_json(prompt)

    def summarize(self, transcript: MeetingTranscript) -> SummaryResult:
        """
        Full summarization entry point. Chunks only if the transcript
        exceeds max_words_per_chunk; otherwise sends it in one call.
        """
        lines = _transcript_to_llm_text(transcript)
        if not lines:
            logger.warning("Empty transcript passed to summarize(); returning empty summary")
            return SummaryResult()

        chunks = _split_into_chunks(lines, settings.max_words_per_chunk)

        if len(chunks) == 1:
            final_prompt = build_final_summary_prompt(chunks[0], from_chunk_summaries=False)
        else:
            logger.info("Transcript split into %d chunks for summarization", len(chunks))
            chunk_summaries = [
                self._summarize_chunk(chunk, i, len(chunks))
                for i, chunk in enumerate(chunks, start=1)
            ]
            combined = json.dumps(chunk_summaries, indent=2)
            final_prompt = build_final_summary_prompt(combined, from_chunk_summaries=True)

        raw = self._call_ollama_json(final_prompt)

        return SummaryResult(
            executive_summary=raw.get("executive_summary", ""),
            key_topics=raw.get("key_topics", []) or [],
            action_items=raw.get("action_items", []) or [],
            decisions=raw.get("decisions", []) or [],
            risks=raw.get("risks", []) or [],
            next_steps=raw.get("next_steps", []) or [],
        )


def summary_to_plain_text(summary: SummaryResult) -> str:
    """Render as the expected human-readable summary.txt sections."""

    def section(title: str, items: list[str]) -> str:
        body = "\n".join(f"- {item}" for item in items) if items else "None identified."
        return f"{title}\n\n{body}\n"

    parts = [
        f"Executive Summary\n\n{summary.executive_summary or 'None identified.'}\n",
        section("Key Topics", summary.key_topics),
        section("Action Items", summary.action_items),
        section("Decisions", summary.decisions),
        section("Risks", summary.risks),
        section("Next Steps", summary.next_steps),
    ]
    return "\n".join(parts)


def write_summary_outputs(summary: SummaryResult, output_dir: Path) -> tuple[Path, Path]:
    """Persist outputs/summary.json and outputs/summary.txt."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "summary.json"
    txt_path = output_dir / "summary.txt"

    json_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    txt_path.write_text(summary_to_plain_text(summary), encoding="utf-8")

    return json_path, txt_path


# Shared instance, consistent with `settings` and `pipeline` singletons
# elsewhere — the Ollama client can be reused across requests.
summarizer = SummarizerService()
