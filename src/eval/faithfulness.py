from __future__ import annotations
from typing import Dict
from src.schema import MoM


def evidence_coverage(mom: MoM) -> Dict[str, float]:
    d = len(mom.decisions)
    a = len(mom.action_items)
    total = d + a
    if total == 0:
        return {"coverage": 1.0, "num_items": 0}

    ok = 0
    for x in mom.decisions:
        if x.evidence and x.evidence.snippet:
            ok += 1
    for x in mom.action_items:
        if x.evidence and x.evidence.snippet:
            ok += 1

    return {"coverage": ok / total, "num_items": total}