from __future__ import annotations

SYSTEM_JSON_ONLY = (
    "You are an information extraction engine. "
    "Return ONLY valid JSON. No markdown, no commentary, no extra keys."
)

def section_extraction_prompt(
    segment_title: str,
    seg_start: int,
    seg_end: int,
    window_text: str,
    top_facts: str
) -> str:
    return f"""
Extract a structured meeting-minutes section from the transcript window.

Constraints:
- Use ONLY information supported by the window text.
- Every decision/action_item MUST include evidence.start_turn and evidence.end_turn within [{seg_start}, {seg_end}].
- evidence.snippet must be an exact quote or near-exact phrase from the window.
- If owner or due date is unknown, set it to null.
- Keep text concise and operational.

Return JSON with EXACT schema:
{{
  "title": string,
  "summary_bullets": [string],
  "decisions": [{{"decision": string, "evidence": {{"start_turn": int, "end_turn": int, "snippet": string}}}}],
  "action_items": [{{"owner": string|null, "task": string, "due": string|null, "priority": "Low"|"Med"|"High",
                    "evidence": {{"start_turn": int, "end_turn": int, "snippet": string}}}}],
  "risks": [string],
  "open_questions": [string]
}}

Segment title hint: {segment_title}
Turn index bounds: {seg_start}-{seg_end}

Top facts (salient, may be used as anchors):
{top_facts}

Window text:
{window_text}
""".strip()