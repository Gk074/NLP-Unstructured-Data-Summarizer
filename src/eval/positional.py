from __future__ import annotations
from typing import List, Dict
from src.schema import Decision, ActionItem

def positional_bins(total_turns: int, idx: int) -> str:
    if total_turns <= 0:
        return "unknown"
    r = idx / max(1, total_turns - 1)
    if r < 1/3:
        return "early"
    if r < 2/3:
        return "middle"
    return "late"

def evidence_position_stats(total_turns: int, decisions: List[Decision], actions: List[ActionItem]) -> Dict[str, float]:
    bins = {"early": 0, "middle": 0, "late": 0}
    n = 0

    for d in decisions:
        mid = (d.evidence.start_turn + d.evidence.end_turn) // 2
        b = positional_bins(total_turns, mid)
        if b in bins:
            bins[b] += 1
            n += 1

    for a in actions:
        mid = (a.evidence.start_turn + a.evidence.end_turn) // 2
        b = positional_bins(total_turns, mid)
        if b in bins:
            bins[b] += 1
            n += 1

    if n == 0:
        return {k: 0.0 for k in bins}
    return {k: v / n for k, v in bins.items()}