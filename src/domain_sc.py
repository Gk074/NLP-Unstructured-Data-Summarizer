from __future__ import annotations
from typing import Dict, List
import re

CATEGORIES: Dict[str, List[str]] = {
    "Demand/Forecast": ["demand", "forecast", "sales", "baseline", "promotions", "s&op", "plan"],
    "Supply/Supplier": ["supplier", "vendor", "lead time", "late", "eta", "shortage", "allocation"],
    "Capacity/Production": ["capacity", "plant", "line", "oee", "throughput", "shift", "bottleneck"],
    "Inventory/Service": ["inventory", "backorder", "fill rate", "dos", "coverage", "safety stock"],
    "Logistics/DC": ["dc", "warehouse", "inbound", "outbound", "lane", "carrier", "expedite", "freight"],
    "Risks/Issues": ["risk", "blocker", "concern", "delay", "constraint"]
}

def classify_topic(text: str) -> str:
    low = text.lower()
    scores = {cat: 0 for cat in CATEGORIES}
    for cat, kws in CATEGORIES.items():
        for kw in kws:
            if kw in low:
                scores[cat] += 1
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "General"