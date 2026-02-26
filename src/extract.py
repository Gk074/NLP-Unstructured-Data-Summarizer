from __future__ import annotations
import re
from typing import List, Tuple, Optional
from collections import Counter

from .ingest import Turn
from .schema import Decision, ActionItem, Evidence

DECISION_PATTERNS = [
    r"\bdecision\b\s*[:\-]?\s*(.*)",
    r"\bwe decided\b\s*(.*)",
    r"\blet[' ]?s decide\b\s*(.*)",
    r"\bfinalize\b\s*(.*)",
    r"\bgo with\b\s*(.*)"
]

ACTION_PATTERNS = [
    r"\baction item\b\s*[:\-]?\s*(.*)",
    r"\b(?P<owner>[A-Z][a-zA-Z0-9_\-]{1,20})\s+(will|to)\s+(?P<task>.+)",
    r"\b(?P<owner>[A-Z][a-zA-Z0-9_\-]{1,20})\s+can\s+(?P<task>.+)",
]

DUE_PATTERNS = [
    r"\bby\s+(?P<due>tomorrow|today|friday|monday|tuesday|wednesday|thursday|saturday|sunday|\d{1,2}/\d{1,2}|\d{4}-\d{2}-\d{2})\b",
    r"\bon\s+(?P<due>\d{4}-\d{2}-\d{2})\b"
]

RISK_PATTERNS = [r"\brisk\b", r"\bblocker\b", r"\bconcern\b", r"\btight\b", r"\bdelay\b"]

QUESTION_PAT = re.compile(r"\?$|^(any|what|why|how|when|where|who)\b", re.IGNORECASE)

def _snippet(turns: List[Turn], a: int, b: int, max_len: int = 220) -> str:
    txt = " ".join([turns[i].text for i in range(a, b + 1)])
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt[:max_len] + ("..." if len(txt) > max_len else "")

def _salience_scores(turns: List[Turn]) -> List[float]:
    """
    Lightweight salience:
    - boost if contains decision/action/risk keywords
    - boost if has numbers/dates ($, %, week)
    - downweight very short turns
    """
    scores = []
    for t in turns:
        s = 0.2
        low = t.text.lower()
        if any(k in low for k in ["decision", "action item", "we decided", "approve", "finalize"]):
            s += 1.2
        if any(k in low for k in ["by ", "tomorrow", "today", "next week", "eod"]):
            s += 0.6
        if re.search(r"(\$|\bweek\b|\bq[1-4]\b|%|\d{2,})", low):
            s += 0.4
        if len(t.text) < 25:
            s -= 0.1
        scores.append(max(0.0, s))
    return scores

def extract_decisions(turns: List[Turn], start: int, end: int) -> List[Decision]:
    out: List[Decision] = []
    for i in range(start, end + 1):
        txt = turns[i].text.strip()
        low = txt.lower()
        for pat in DECISION_PATTERNS:
            m = re.search(pat, low, flags=re.IGNORECASE)
            if m:
                # keep original casing from txt by slicing approximate; simplest: use captured group on original via re on original too
                m2 = re.search(pat, txt, flags=re.IGNORECASE)
                decision_text = (m2.group(1).strip() if m2 and m2.lastindex else txt).strip()
                if not decision_text:
                    decision_text = txt
                out.append(
                    Decision(
                        decision=decision_text,
                        evidence=Evidence(start_turn=i, end_turn=i, snippet=_snippet(turns, i, i))
                    )
                )
                break
    return out

def extract_actions(turns: List[Turn], start: int, end: int) -> List[ActionItem]:
    out: List[ActionItem] = []
    for i in range(start, end + 1):
        txt = turns[i].text.strip()
        # Try explicit "Action item:"
        m0 = re.search(ACTION_PATTERNS[0], txt, flags=re.IGNORECASE)
        if m0:
            task = m0.group(1).strip()
            owner = None
            due = _extract_due(task) or _extract_due(txt)
            out.append(ActionItem(owner=owner, task=task, due=due, priority="Med",
                                 evidence=Evidence(start_turn=i, end_turn=i, snippet=_snippet(turns, i, i))))
            continue

        # Try "<Owner> will/to ..." patterns
        for pat in ACTION_PATTERNS[1:]:
            m = re.search(pat, txt)
            if m:
                owner = m.groupdict().get("owner")
                task = m.groupdict().get("task") or m.group(0)
                task = task.strip()
                due = _extract_due(txt) or _extract_due(task)
                out.append(ActionItem(owner=owner, task=task, due=due, priority="Med",
                                     evidence=Evidence(start_turn=i, end_turn=i, snippet=_snippet(turns, i, i))))
                break
    return out

def _extract_due(text: str) -> Optional[str]:
    for pat in DUE_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group("due")
    return None

def extract_risks(turns: List[Turn], start: int, end: int) -> List[str]:
    risks = []
    for i in range(start, end + 1):
        low = turns[i].text.lower()
        if any(re.search(p, low) for p in RISK_PATTERNS):
            risks.append(turns[i].text)
    return _dedupe_text(risks)

def extract_questions(turns: List[Turn], start: int, end: int) -> List[str]:
    qs = []
    for i in range(start, end + 1):
        t = turns[i].text.strip()
        if QUESTION_PAT.search(t):
            qs.append(t)
    return _dedupe_text(qs)

def pick_key_turns(turns: List[Turn], start: int, end: int, k: int = 5) -> List[int]:
    scores = _salience_scores(turns[start:end+1])
    idxs = list(range(start, end + 1))
    ranked = sorted(zip(idxs, scores), key=lambda x: x[1], reverse=True)
    picked = [i for i, s in ranked[:k] if s > 0.2]
    picked.sort()
    return picked

def _dedupe_text(items: List[str], max_items: int = 8) -> List[str]:
    seen = set()
    out = []
    for x in items:
        key = re.sub(r"\W+", "", x.lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
        if len(out) >= max_items:
            break
    return out