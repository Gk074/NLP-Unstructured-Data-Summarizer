from __future__ import annotations
from typing import List
import re
from sentence_transformers import SentenceTransformer
import numpy as np

from .schema import Decision, ActionItem
from .ingest import Turn

def _in_bounds(a: int, b: int, lo: int, hi: int) -> bool:
    return lo <= a <= b <= hi

def _contains_snippet(window: str, snippet: str) -> bool:
    if not snippet:
        return False
    # relaxed containment
    w = re.sub(r"\s+", " ", window.lower())
    s = re.sub(r"\s+", " ", snippet.lower()).strip()
    return s in w

def _similarity(model: SentenceTransformer, a: str, b: str) -> float:
    ea = model.encode([a], normalize_embeddings=True, show_progress_bar=False)[0]
    eb = model.encode([b], normalize_embeddings=True, show_progress_bar=False)[0]
    return float(np.dot(ea, eb))

def verify_decisions_actions(
    turns: List[Turn],
    seg_start: int,
    seg_end: int,
    decisions: List[Decision],
    actions: List[ActionItem],
    sim_threshold: float = 0.55,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    window_text = " ".join([turns[i].text for i in range(seg_start, seg_end + 1)])
    model = SentenceTransformer(model_name)

    kept_decisions: List[Decision] = []
    for d in decisions:
        ev = d.evidence
        if not _in_bounds(ev.start_turn, ev.end_turn, seg_start, seg_end):
            continue
        if not _contains_snippet(window_text, ev.snippet):
            continue
        sim = _similarity(model, d.decision, ev.snippet)
        if sim < sim_threshold:
            continue
        kept_decisions.append(d)

    kept_actions: List[ActionItem] = []
    for a in actions:
        ev = a.evidence
        if not _in_bounds(ev.start_turn, ev.end_turn, seg_start, seg_end):
            continue
        if not _contains_snippet(window_text, ev.snippet):
            continue
        sim = _similarity(model, a.task, ev.snippet)
        if sim < sim_threshold:
            continue
        kept_actions.append(a)

    return kept_decisions, kept_actions