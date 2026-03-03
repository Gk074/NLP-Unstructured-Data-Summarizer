from __future__ import annotations
from typing import List, Dict, Tuple
import re

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 %$\-]", "", s)
    return s

def _match(pred: str, gold: str) -> bool:
    # simple robust match: substring either way after normalization
    p, g = _norm(pred), _norm(gold)
    return (p in g) or (g in p) or (jaccard(p, g) >= 0.6)

def jaccard(a: str, b: str) -> float:
    A, B = set(a.split()), set(b.split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def prf1(pred_items: List[str], gold_items: List[str]) -> Dict[str, float]:
    if not pred_items and not gold_items:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_items:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gold_items:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_gold = set()
    tp = 0
    for p in pred_items:
        hit = False
        for j, g in enumerate(gold_items):
            if j in matched_gold:
                continue
            if _match(p, g):
                matched_gold.add(j)
                tp += 1
                hit = True
                break

    fp = len(pred_items) - tp
    fn = len(gold_items) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}