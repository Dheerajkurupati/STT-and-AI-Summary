"""
Prompt templates sent to Ollama.

WHY THIS FILE EXISTS AND IS SEPARATE FROM summarize.py:
Prompts get iterated on constantly during development (wording tweaks,
schema changes, few-shot examples) and that churn has nothing to do with
the plumbing of *how* we call Ollama, chunk text, or parse responses. By
keeping prompts as plain string templates here, we can tune output quality
without touching summarize.py's logic, and vice versa.

WHY WE ASK FOR JSON OUTPUT:
Ollama's Python client supports `format="json"` (see summarize.py), which
constrains the model to emit syntactically valid JSON. Combined with an
explicit schema in the prompt, this lets summarize.py parse the response
directly into a SummaryResult instead of doing fragile regex/markdown
parsing on free-form prose.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert meeting analyst. You read meeting transcripts and extract \
accurate, concise, non-redundant information. You never invent information \
that is not present in the transcript. If a category has no relevant \
content, return an empty list for it rather than fabricating an entry."""


# Used once per chunk when a transcript is too long to fit in a single
# Ollama context window. Each chunk is summarized independently first, and
# these partial summaries are later combined by FINAL_SUMMARY_PROMPT.
CHUNK_SUMMARY_PROMPT_TEMPLATE = """\
Below is one part ({chunk_index} of {total_chunks}) of a longer meeting \
transcript. Extract only what is explicitly discussed in THIS part.

Respond with JSON only, matching this exact schema:
{{
  "key_points": ["..."],
  "action_items": ["..."],
  "decisions": ["..."],
  "risks": ["..."]
}}

Transcript part:
---
{chunk_text}
---
"""


# Used on the final pass: either the full transcript (if it fit in one
# chunk) or the concatenation of all chunk summaries. Produces the final
# structured output returned to the user.
FINAL_SUMMARY_PROMPT_TEMPLATE = """\
Below is {source_description} of a meeting. Produce a final, deduplicated \
summary of the entire meeting.

Respond with JSON only, matching this exact schema:
{{
  "executive_summary": "2-4 sentence overview of the whole meeting",
  "key_topics": ["..."],
  "action_items": ["..."],
  "decisions": ["..."],
  "risks": ["..."],
  "next_steps": ["..."]
}}

Content:
---
{content}
---
"""


def build_chunk_summary_prompt(chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    return CHUNK_SUMMARY_PROMPT_TEMPLATE.format(
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )


def build_final_summary_prompt(content: str, from_chunk_summaries: bool) -> str:
    source_description = (
        "a set of partial summaries (one per section)"
        if from_chunk_summaries
        else "the full transcript"
    )
    return FINAL_SUMMARY_PROMPT_TEMPLATE.format(
        source_description=source_description,
        content=content,
    )
