from __future__ import annotations
from typing import List

def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

def read_text_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().splitlines()