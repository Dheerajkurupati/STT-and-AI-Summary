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
    """Stateful wrapper around the Groq client, reused across requests."""

    def __init__(self) -> None:
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not settings.groq_api_key:
                raise SummarizationError("GROQ_API_KEY is not set.")
            from groq import Groq

            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    def _call_groq_json(self, prompt: str) -> dict:
        """
        Send one prompt to Groq (Llama-3.3-70B) and parse the response as JSON.
        """
        logger.info("Generating summary using Groq API (%s)...", settings.groq_llm_model)

        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=settings.groq_llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise SummarizationError(
                f"Failed to reach Groq API: {exc}"
            ) from exc

        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise SummarizationError(
                f"Groq returned non-JSON content despite JSON mode: {content[:200]}"
            ) from exc

    def _summarize_chunk(self, chunk_text: str, index: int, total: int) -> dict:
        logger.info("Summarizing chunk %d/%d", index, total)
        prompt = build_chunk_summary_prompt(chunk_text, index, total)
        return self._call_groq_json(prompt)

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

        raw = self._call_groq_json(final_prompt)

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
