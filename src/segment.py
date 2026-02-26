from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from .ingest import Turn

@dataclass
class Segment:
    seg_id: int
    start: int
    end: int  # inclusive
    title_hint: str

def embed_turns(turns: List[Turn], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    texts = [f"{t.speaker}: {t.text}" for t in turns]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb)

def segment_by_similarity_drop(
    turns: List[Turn],
    emb: np.ndarray,
    window: int = 1,
    threshold: float = 0.72,
    min_seg_len: int = 4
) -> List[Tuple[int, int]]:
    """
    Simple, robust segmentation:
    - compute cosine similarity between adjacent turns
    - cut when similarity drops below threshold
    - enforce minimum segment length
    """
    n = len(turns)
    if n == 0:
        return []

    # cosine since normalized
    sims = []
    for i in range(n - 1):
        sims.append(float(np.dot(emb[i], emb[i + 1])))
    sims = np.array(sims)

    cut_points = [0]
    for i, s in enumerate(sims):
        if s < threshold:
            cut_points.append(i + 1)
    cut_points.append(n)

    # merge tiny segments
    spans = []
    for a, b in zip(cut_points[:-1], cut_points[1:]):
        spans.append((a, b - 1))
    merged = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if (cur_e - cur_s + 1) < min_seg_len:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged

def title_from_segment(turns: List[Turn], start: int, end: int) -> str:
    # crude but useful: pick top keywordy words from first 3 turns
    import re
    stop = set(["the","a","an","and","or","to","of","in","on","for","we","i","you","is","are","was","were","be","it","this","that"])
    text = " ".join([turns[i].text for i in range(start, min(end+1, start+3))]).lower()
    words = [w for w in re.findall(r"[a-z0-9\-]+", text) if w not in stop and len(w) > 3]
    if not words:
        return "General"
    # take up to 3 most frequent
    from collections import Counter
    top = [w for w, _ in Counter(words).most_common(3)]
    return " / ".join(top).title()

def build_segments(turns: List[Turn], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[Segment]:
    emb = embed_turns(turns, model_name=model_name)
    spans = segment_by_similarity_drop(turns, emb)
    segs: List[Segment] = []
    for k, (s, e) in enumerate(spans):
        segs.append(Segment(seg_id=k, start=s, end=e, title_hint=title_from_segment(turns, s, e)))
    return segs