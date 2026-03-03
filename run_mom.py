from __future__ import annotations
from typing import List

from src.utils import read_text_file
from src.ingest import parse_transcript, merge_consecutive_same_speaker
from src.segment import build_segments
from src.facts import build_facts, salience_score_facts, top_facts
from src.theme import cluster_facts_into_themes

from src.llm_provider import groq_chat_json
from src.prompts import SYSTEM_JSON_ONLY, section_extraction_prompt
from src.verify import verify_decisions_actions
from src.domain_sc import classify_topic
from src.schema import MoM, TopicSection, Evidence, Meta
from src.render import mom_to_markdown


def build_mom(transcript_path: str, meeting_title: str = "MoM") -> MoM:
    lines = read_text_file(transcript_path)
    turns = merge_consecutive_same_speaker(parse_transcript(lines))
    if not turns:
        raise ValueError("Empty transcript after parsing.")

    segments = build_segments(turns)

    topic_sections: List[TopicSection] = []
    all_decisions = []
    all_actions = []
    all_risks = []
    all_questions = []

    for seg in segments:
        s, e = seg.start, seg.end

        # Phase-2 anchors: facts + salience + themes
        facts = build_facts(turns, s, e)
        facts = salience_score_facts(facts)
        picked = top_facts(facts, k=12)
        themes = cluster_facts_into_themes(picked)

        # Segment-level evidence object (always present)
        ev = Evidence(
            start_turn=s,
            end_turn=e,
            snippet=" ".join([turns[i].text for i in range(s, min(e + 1, s + 2))])[:220],
        )

        # Build window and anchors for the LLM prompt
        window_text = "\n".join([f"{turns[i].speaker}: {turns[i].text}" for i in range(s, e + 1)])
        top_facts_text = "\n".join(
            [f"- ({f.kind}, {f.score:.2f}) {f.speaker}: {f.text} [turn {f.turn_idx}]"
             for f in picked[:12]]
        )

        # Title hint + theme hint
        segment_title = seg.title_hint
        if themes:
            segment_title = f"{seg.title_hint} | {themes[0].title}"

        prompt = section_extraction_prompt(
            segment_title=segment_title,
            seg_start=s,
            seg_end=e,
            window_text=window_text,
            top_facts=top_facts_text,
        )

        # ---- Groq JSON extraction ----
        raw = groq_chat_json(
            system=SYSTEM_JSON_ONLY,
            user=prompt,
            temperature=0.2,
            max_tokens=1200,
        )

        # Pydantic validation into TopicSection
        # (We inject "evidence" for the whole section separately)
        section = TopicSection.model_validate({
            "title": raw.get("title", segment_title),
            "summary_bullets": raw.get("summary_bullets", []),
            "decisions": raw.get("decisions", []),
            "action_items": raw.get("action_items", []),
            "risks": raw.get("risks", []),
            "open_questions": raw.get("open_questions", []),
            "evidence": ev.model_dump(),
        })

        # Evidence verification gate (drops hallucinated/unsupported items)
        section.decisions, section.action_items = verify_decisions_actions(
            turns=turns,
            seg_start=s,
            seg_end=e,
            decisions=section.decisions,
            actions=section.action_items,
            sim_threshold=0.55,
        )
        domain_label = classify_topic(" ".join(section.summary_bullets + section.risks + section.open_questions))
        section.title = f"[{domain_label}] {section.title}"
        topic_sections.append(section)

        all_decisions.extend(section.decisions)
        all_actions.extend(section.action_items)
        all_risks.extend(section.risks)
        all_questions.extend(section.open_questions)

    # TL;DR: concise from first topics
    tldr_parts = []
    for sec in topic_sections[:2]:
        tldr_parts.extend(sec.summary_bullets[:2])
    tldr = " ".join(tldr_parts) if tldr_parts else "Meeting minutes generated."

    mom = MoM(
        meeting_title=meeting_title,
        tldr=tldr,
        topics=topic_sections,
        decisions=all_decisions,
        action_items=all_actions,
        risks=all_risks[:10],
        open_questions=all_questions[:10],
        meta=Meta(total_turns=len(turns))
    )
    return mom


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--transcript", type=str, required=True)
    p.add_argument("--title", type=str, default="MoM")
    p.add_argument("--out_json", type=str, default="mom.json")
    p.add_argument("--out_md", type=str, default="mom.md")
    args = p.parse_args()

    mom = build_mom(args.transcript, meeting_title=args.title)

    with open(args.out_json, "w", encoding="utf-8") as f:
        f.write(mom.model_dump_json(indent=2))

    md = mom_to_markdown(mom)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"✅ Wrote {args.out_json} and {args.out_md}")