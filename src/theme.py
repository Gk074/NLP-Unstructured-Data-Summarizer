from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from .facts import Fact

@dataclass
class Theme:
    theme_id: int
    title: str
    facts: List[Fact]

def cluster_facts_into_themes(
    facts: List[Fact],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    distance_threshold: float = 1.05
) -> List[Theme]:
    if not facts:
        return []

    model = SentenceTransformer(model_name)
    texts = [f.text for f in facts]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # cosine distance = 1 - cosine similarity
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average"
    )
    labels = clustering.fit_predict(emb)

    groups: Dict[int, List[Fact]] = {}
    for f, lab in zip(facts, labels):
        groups.setdefault(int(lab), []).append(f)

    themes: List[Theme] = []
    for tid, fs in sorted(groups.items(), key=lambda x: -sum(f.score for f in x[1])):
        title = suggest_theme_title(fs)
        themes.append(Theme(theme_id=tid, title=title, facts=sorted(fs, key=lambda x: (-x.score, x.turn_idx))))
    return themes

def suggest_theme_title(facts: List[Fact]) -> str:
    # simple keyword title from top facts
    import re
    from collections import Counter
    stop = set(["the","a","an","and","or","to","of","in","on","for","we","i","you","is","are","was","were","be","it","this","that"])
    text = " ".join(f.text.lower() for f in facts[:5])
    words = [w for w in re.findall(r"[a-z0-9\-]+", text) if w not in stop and len(w) > 3]
    if not words:
        return "General"
    top = [w for w, _ in Counter(words).most_common(3)]
    return " / ".join(top).title()