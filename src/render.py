from __future__ import annotations
from typing import List
from .schema import MoM

def mom_to_markdown(mom: MoM) -> str:
    md = []
    md.append(f"# {mom.meeting_title}\n")
    md.append("## TL;DR\n")
    md.append(f"{mom.tldr}\n")

    md.append("## Topics\n")
    for t in mom.topics:
        md.append(f"### {t.title}\n")
        if t.summary_bullets:
            md.append("**Summary**\n")
            for b in t.summary_bullets:
                md.append(f"- {b}")
            md.append("")
        if t.decisions:
            md.append("**Decisions**\n")
            for d in t.decisions:
                md.append(f"- {d.decision} *(evidence turns {d.evidence.start_turn}-{d.evidence.end_turn})*")
            md.append("")
        if t.action_items:
            md.append("**Action Items**\n")
            for a in t.action_items:
                owner = a.owner or "Unassigned"
                due = f", due: {a.due}" if a.due else ""
                md.append(f"- [{a.priority}] {owner}: {a.task}{due} *(evidence turns {a.evidence.start_turn}-{a.evidence.end_turn})*")
            md.append("")
        if t.risks:
            md.append("**Risks/Blockers**\n")
            for r in t.risks:
                md.append(f"- {r}")
            md.append("")
        if t.open_questions:
            md.append("**Open Questions**\n")
            for q in t.open_questions:
                md.append(f"- {q}")
            md.append("")

    if mom.decisions:
        md.append("## All Decisions\n")
        for d in mom.decisions:
            md.append(f"- {d.decision} *(turns {d.evidence.start_turn}-{d.evidence.end_turn})*")
        md.append("")

    if mom.action_items:
        md.append("## All Action Items\n")
        for a in mom.action_items:
            owner = a.owner or "Unassigned"
            due = f", due: {a.due}" if a.due else ""
            md.append(f"- [{a.priority}] {owner}: {a.task}{due} *(turns {a.evidence.start_turn}-{a.evidence.end_turn})*")
        md.append("")

    if mom.risks:
        md.append("## Risks\n")
        for r in mom.risks:
            md.append(f"- {r}")
        md.append("")

    if mom.open_questions:
        md.append("## Open Questions\n")
        for q in mom.open_questions:
            md.append(f"- {q}")
        md.append("")

    return "\n".join(md).strip() + "\n"