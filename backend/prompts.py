"""
Prompt templates for LLM-based meeting summarization.

DESIGN — RAG-STYLE MAP-REDUCE:
  1. MAP  — Each transcript chunk gets CHUNK_EXTRACT_PROMPT:
             Extract raw facts (topics, decisions, actions, risks, quotes)
             from that window only. No synthesis yet.
  2. REDUCE — All chunk extractions are fed into FINAL_SUMMARY_PROMPT:
             Deduplicate, resolve conflicts, and produce the final structured
             output. This is where narrative writing happens.

WHY SEPARATE MAP AND REDUCE:
  A small 8B model can't reliably hold a long meeting in its context window
  *and* produce a good summary simultaneously. Splitting the work into focused
  extraction passes (MAP) followed by a single merge pass (REDUCE) dramatically
  improves output quality — each call is a simpler, well-scoped task.

WHY JSON OUTPUT:
  Groq's response_format={"type": "json_object"} constrains the model to emit
  syntactically valid JSON. Combined with an explicit schema in the prompt, this
  lets summarize.py parse the response directly into SummaryResult.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt — sets the model's persona for ALL calls
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior meeting analyst with deep expertise in extracting \
actionable intelligence from business and technical meeting transcripts. \
Your job is to produce structured, specific, and accurate summaries — \
never vague or generic. \
Rules you MUST follow:
1. Only report information that is EXPLICITLY stated in the transcript.
2. Be SPECIFIC: include names, numbers, dates, product names, and verbatim \
decisions whenever they appear in the transcript.
3. Never fabricate or infer beyond what is stated.
4. If a category has no content in the transcript, return an empty list [].
5. Write ALL output in English. Translate non-English content as needed.\
"""

# ---------------------------------------------------------------------------
# MAP prompt — per-chunk extraction (does NOT write narrative prose)
# ---------------------------------------------------------------------------

CHUNK_EXTRACT_PROMPT_TEMPLATE = """\
Below is PART {chunk_index} of {total_chunks} from a meeting transcript.
Your task: extract raw facts from THIS part only. Do not summarize — extract.

Return ONLY a JSON object matching this exact schema. \
No markdown, no explanation outside the JSON:
{{
  "topics_discussed": [
    "Specific topic or agenda item discussed, with context"
  ],
  "decisions_made": [
    "Exact decision stated, by whom if known, and any conditions"
  ],
  "action_items": [
    "Task owner: what they agreed to do, and deadline if mentioned"
  ],
  "risks_or_issues": [
    "Problem, blocker, or risk raised, with context"
  ],
  "key_facts": [
    "Any specific number, date, name, metric, or technical detail mentioned"
  ]
}}

Rules:
- Each string must be a COMPLETE, STANDALONE sentence or phrase.
- Include speaker names or roles if mentioned (e.g. "John agreed to...", \
"The PM said...").
- Do NOT write generic statements like "discussed project status". \
Write "Discussed Q3 launch timeline — currently 2 weeks behind schedule."
- If nothing applies for a field, use [].

Transcript Part {chunk_index}/{total_chunks}:
---
{chunk_text}
---
"""

# ---------------------------------------------------------------------------
# REDUCE prompt — final merge pass over all chunk extractions
# ---------------------------------------------------------------------------

FINAL_SUMMARY_PROMPT_TEMPLATE = """\
Below is {source_description}.
Your task: produce a complete, deduplicated, well-written final meeting summary.

IMPORTANT RULES:
1. Write in English only.
2. Merge duplicates — if the same decision or action item appears multiple \
times, list it ONCE with the most complete version.
3. Be SPECIFIC — include names, numbers, dates, and exact wording of \
decisions/actions from the input.
4. The executive_summary must be 3-5 sentences that tell a first-time reader \
exactly what this meeting was about, what was decided, and what happens next.
5. key_topics: list every distinct subject area discussed (e.g. \
"Q3 product roadmap", "budget approval for cloud infrastructure", \
"onboarding process for new hires").
6. action_items: format as "Person: task (deadline if known)". \
If no owner is named, use "Team:".
7. decisions: exact decisions reached — not topics discussed. \
Something is a decision only if the group agreed on an outcome.
8. risks: concrete problems, blockers, or concerns raised — not hypotheticals.
9. next_steps: what happens after this meeting (follow-ups, meetings scheduled, \
deliverables due).

Return ONLY a JSON object matching this exact schema. \
No markdown, no explanation outside the JSON:
{{
  "executive_summary": "3-5 sentence narrative overview",
  "key_topics": ["topic 1", "topic 2"],
  "action_items": ["Owner: task (deadline)", "Owner: task"],
  "decisions": ["Specific decision reached"],
  "risks": ["Concrete risk or blocker raised"],
  "next_steps": ["Follow-up or deliverable"]
}}

Content to synthesize:
---
{content}
---
"""

# ---------------------------------------------------------------------------
# Builder functions used by summarize.py
# ---------------------------------------------------------------------------


def build_chunk_extract_prompt(chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    """Build the MAP-phase extraction prompt for one chunk."""
    return CHUNK_EXTRACT_PROMPT_TEMPLATE.format(
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )


def build_final_summary_prompt(content: str, from_chunk_summaries: bool) -> str:
    """Build the REDUCE-phase prompt from either raw transcript or chunk extractions."""
    source_description = (
        "a set of structured extractions — one per section of the transcript"
        if from_chunk_summaries
        else "the full meeting transcript"
    )
    return FINAL_SUMMARY_PROMPT_TEMPLATE.format(
        source_description=source_description,
        content=content,
    )


# ---------------------------------------------------------------------------
# Legacy aliases — keep old names so any external code still works
# ---------------------------------------------------------------------------

def build_chunk_summary_prompt(chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    """Alias for build_chunk_extract_prompt (backward compatibility)."""
    return build_chunk_extract_prompt(chunk_text, chunk_index, total_chunks)

# ---------------------------------------------------------------------------
# DIARIZATION prompt — LLM semantic fallback
# ---------------------------------------------------------------------------

LLM_DIARIZATION_PROMPT_TEMPLATE = """\
Below is a numbered list of sequential text segments from a meeting. Acoustic diarization failed, so all segments lack distinct speaker labels.
Your task: Read the conversational flow, natural turn-taking, questions, and answers to assign a speaker label (e.g., "Speaker 1", "Speaker 2") to EACH segment.

Return a plain text list where each line corresponds to a segment number and its assigned speaker.
DO NOT output any explanations, markdown, or JSON. Just the mapping.

Example output:
1: Speaker 1
2: Speaker 2
3: Speaker 1

Input segments:
{content}
"""
