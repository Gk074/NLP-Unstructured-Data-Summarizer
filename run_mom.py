from __future__ import annotations
import json
from typing import List

from src.utils import read_text_file
from src.ingest import parse_transcript, merge_consecutive_same_speaker, Turn
from src.segment import build_segments
from src.extract import (
    extract_decisions, extract_actions, extract_risks, extract_questions, pick_key_turns
)
from src.summarize import extractive_summary_bullets
from src.schema import MoM, TopicSection, Evidence
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
        dec = extract_decisions(turns, s, e)
        act = extract_actions(turns, s, e)
        risks = extract_risks(turns, s, e)
        qs = extract_questions(turns, s, e)

        key_turns = pick_key_turns(turns, s, e, k=6)
        bullets = extractive_summary_bullets(turns, key_turns, max_bullets=5)

        ev = Evidence(start_turn=s, end_turn=e, snippet=" ".join([turns[i].text for i in range(s, min(e+1, s+2))])[:220])

        section = TopicSection(
            title=seg.title_hint,
            summary_bullets=bullets,
            decisions=dec,
            action_items=act,
            risks=risks,
            open_questions=qs,
            evidence=ev
        )
        topic_sections.append(section)

        all_decisions.extend(dec)
        all_actions.extend(act)
        all_risks.extend(risks)
        all_questions.extend(qs)

    # TL;DR: take top bullets from first 2 segments
    tldr_parts = []
    for sec in topic_sections[:2]:
        tldr_parts.extend(sec.summary_bullets[:2])
    tldr = " ".join(tldr_parts) if tldr_parts else "Meeting summary generated."

    mom = MoM(
        meeting_title=meeting_title,
        tldr=tldr,
        topics=topic_sections,
        decisions=all_decisions,
        action_items=all_actions,
        risks=all_risks[:10],
        open_questions=all_questions[:10],
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