from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional

TS_RE = re.compile(r"^(?P<ts>\d{1,2}:\d{2}:\d{2})\s+")
SPK_RE = re.compile(r"^(?P<speaker>[^:]{1,40}):\s*(?P<text>.*)$")

@dataclass
class Turn:
    idx: int
    speaker: str
    text: str
    timestamp: Optional[str] = None

def _clean_text(t: str) -> str:
    # light cleanup: collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    # remove some filler patterns (keep conservative)
    t = re.sub(r"\b(um+|uh+|you know|like)\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def parse_transcript(lines: List[str]) -> List[Turn]:
    turns: List[Turn] = []
    idx = 0
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue

        ts = None
        m_ts = TS_RE.match(raw)
        if m_ts:
            ts = m_ts.group("ts")
            raw = raw[m_ts.end():].strip()

        m_spk = SPK_RE.match(raw)
        if m_spk:
            speaker = m_spk.group("speaker").strip()
            text = m_spk.group("text").strip()
        else:
            # fallback: unknown speaker
            speaker = "Unknown"
            text = raw

        text = _clean_text(text)
        if text:
            turns.append(Turn(idx=idx, speaker=speaker, text=text, timestamp=ts))
            idx += 1

    return turns

def merge_consecutive_same_speaker(turns: List[Turn]) -> List[Turn]:
    if not turns:
        return []

    merged: List[Turn] = [turns[0]]
    for t in turns[1:]:
        last = merged[-1]
        if t.speaker == last.speaker:
            last.text = (last.text + " " + t.text).strip()
        else:
            merged.append(t)

    # reindex
    for i, t in enumerate(merged):
        t.idx = i
    return merged