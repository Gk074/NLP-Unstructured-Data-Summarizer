from __future__ import annotations
from typing import List
import re
from collections import Counter

from .ingest import Turn

def _sentence_split(text: str) -> List[str]:
    # simple splitter
    parts = re.split(r"(?<=[\.\?!])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def extractive_summary_bullets(turns: List[Turn], turn_idxs: List[int], max_bullets: int = 5) -> List[str]:
    """
    Very lightweight extractive summarizer:
    - gather key turns
    - split into sentences
    - score sentences by word frequency
    - return top N as bullets
    """
    text = " ".join([turns[i].text for i in turn_idxs]).strip()
    sents = _sentence_split(text)
    if not sents:
        return []

    tokens = re.findall(r"[A-Za-z0-9\-]+", text.lower())
    stop = set(["the","a","an","and","or","to","of","in","on","for","we","i","you","is","are","was","were","be","it","this","that","with","as","at"])
    tokens = [t for t in tokens if t not in stop and len(t) > 2]
    freq = Counter(tokens)

    def score(sent: str) -> float:
        w = re.findall(r"[A-Za-z0-9\-]+", sent.lower())
        w = [t for t in w if t not in stop]
        if not w:
            return 0.0
        return sum(freq.get(t, 0) for t in w) / (len(w) ** 0.7)

    ranked = sorted(sents, key=score, reverse=True)
    bullets = []
    for s in ranked:
        s = s.strip()
        if s and s not in bullets:
            bullets.append(s)
        if len(bullets) >= max_bullets:
            break
    return bullets