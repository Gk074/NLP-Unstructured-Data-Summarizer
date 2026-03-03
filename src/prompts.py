from __future__ import annotations

SYSTEM_JSON_ONLY = (
    "You are an information extraction engine. "
    "Return ONLY valid JSON. No markdown, no commentary, no extra keys."
)

SUPPLY_CHAIN_RULES = """
You are generating Minutes of Meeting (MoM) for SUPPLY CHAIN / S&OP / DEMAND PLANNING meetings.

Prioritize capturing:
1) Demand / forecast changes (what changed, where, magnitude, timeframe)
2) Supply constraints (supplier delays, lead time, shortages, allocations)
3) Capacity constraints (plant/line/DC capacity, shifts, throughput, bottlenecks)
4) Logistics/DC execution issues (expedites, carrier issues, inbound/outbound constraints)
5) Decisions (what was agreed / finalized)
6) Action items (owner, task, due date if explicitly stated)
7) Risks/blockers and open questions

Writing style:
- Concise, operational bullets (no storytelling).
- Use numbers, weeks, lanes, suppliers, sites when present in the transcript.
- Do NOT invent owners, dates, or quantities.
- If due date is not explicitly stated, set "due": null.
- If owner is not explicitly stated, set "owner": null.
""".strip()


def section_extraction_prompt(
    segment_title: str,
    seg_start: int,
    seg_end: int,
    window_text: str,
    top_facts: str
) -> str:
    return f"""
Extract a structured meeting-minutes section from the transcript window.

Supply chain MoM rules:
{SUPPLY_CHAIN_RULES}

Hard constraints (must follow):
- Use ONLY information supported by the window text.
- Every decision/action_item MUST include evidence.start_turn and evidence.end_turn within [{seg_start}, {seg_end}].
- evidence.snippet must be an exact quote or near-exact phrase from the window.
- Do NOT guess owner or due date. Use null if unknown.
- Keep text concise and operational.

Return JSON with EXACT schema (no extra keys):
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

Top facts (salient anchors):
{top_facts}

Window text:
{window_text}
""".strip()