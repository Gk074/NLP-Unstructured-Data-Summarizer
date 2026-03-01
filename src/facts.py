from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import re
from collections import Counter
import numpy as np

from .ingest import Turn

@dataclass
class Fact:
    fact_id: str
    turn_idx: int
    speaker: str
    text: str
    kind: str  # "claim" | "decision" | "action" | "risk" | "question"
    score: float = 0.0

def _sent_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def build_facts(turns: List[Turn], start: int, end: int) -> List[Fact]:
    facts: List[Fact] = []
    fid = 0
    for i in range(start, end + 1):
        t = turns[i]
        for s in _sent_split(t.text):
            kind = classify_fact(s)
            facts.append(Fact(
                fact_id=f"F{start}_{end}_{fid}",
                turn_idx=i,
                speaker=t.speaker,
                text=s,
                kind=kind
            ))
            fid += 1
    return facts

def classify_fact(sent: str) -> str:
    low = sent.lower().strip()
    if low.endswith("?") or re.match(r"^(any|what|why|how|when|where|who)\b", low):
        return "question"
    if "decision" in low or "we decided" in low or "go with" in low:
        return "decision"
    if "action item" in low or re.search(r"\b[A-Z][a-zA-Z0-9_\-]{1,20}\s+(will|to|can)\b", sent):
        return "action"
    if re.search(r"\brisk\b|\bblocker\b|\bconcern\b|\bdelay\b|\btight\b", low):
        return "risk"
    return "claim"

def salience_score_facts(facts: List[Fact]) -> List[Fact]:
    """
    Lightweight salience (Part-2 baseline):
    - boost decision/action/risk/question
    - boost numbers/cost/weeks/dates
    - boost supply-chain keywords (niche hook)
    """
    sc_keywords = ["demand", "forecast", "supplier", "capacity", "expedite", "backorder",
                   "lead time", "inventory", "plant", "dc", "eta", "allocation", "fill rate"]

    for f in facts:
        s = 0.2
        low = f.text.lower()

        if f.kind in ("decision", "action"):
            s += 1.2
        elif f.kind in ("risk", "question"):
            s += 0.7

        if re.search(r"(\$|%|\bweek\b|\beta\b|\bq[1-4]\b|\d{2,})", low):
            s += 0.4

        if any(k in low for k in sc_keywords):
            s += 0.4

        if len(f.text) < 25:
            s -= 0.1

        f.score = max(0.0, s)
    return facts

def top_facts(facts: List[Fact], k: int = 12) -> List[Fact]:
    facts_sorted = sorted(facts, key=lambda x: x.score, reverse=True)
    picked = []
    seen = set()
    for f in facts_sorted:
        key = re.sub(r"\W+", "", f.text.lower())[:90]
        if key in seen:
            continue
        seen.add(key)
        picked.append(f)
        if len(picked) >= k:
            break
    return picked