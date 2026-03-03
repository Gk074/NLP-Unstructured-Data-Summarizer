from __future__ import annotations
import os
from typing import Dict, Any, List

from src.schema import MoM
from src.eval.io import load_json
from src.eval.metrics import prf1
from src.eval.positional import evidence_position_stats
from src.eval.faithfulness import evidence_coverage

from run_mom import build_mom  # uses your Phase-3 pipeline


def extract_strings(mom: MoM) -> Dict[str, List[str]]:
    decisions = [d.decision for d in mom.decisions]

    actions = []
    for a in mom.action_items:
        parts = []
        if a.owner:
            parts.append(a.owner)
        parts.append(a.task)
        if a.due:
            parts.append(f"due {a.due}")
        actions.append(" | ".join(parts))

    return {"decisions": decisions, "action_items": actions}


def extract_gold_strings(gold: Dict[str, Any]) -> Dict[str, List[str]]:
    gd = [x["decision"] for x in gold.get("decisions", [])]

    ga = []
    for x in gold.get("action_items", []):
        parts = []
        if x.get("owner"):
            parts.append(x["owner"])
        parts.append(x["task"])
        if x.get("due"):
            parts.append(f"due {x['due']}")
        ga.append(" | ".join(parts))

    return {"decisions": gd, "action_items": ga}


def avg(xs: List[Dict[str, float]], k: str) -> float:
    return sum(x[k] for x in xs) / len(xs) if xs else 0.0


def main():
    base_t = "data/eval/transcripts"
    base_g = "data/eval/gold"

    files = sorted([f for f in os.listdir(base_t) if f.endswith(".txt")])
    if not files:
        raise RuntimeError("No eval transcripts found in data/eval/transcripts")

    all_action = []
    all_dec = []
    cov_list = []

    pos_accum = {"early": 0.0, "middle": 0.0, "late": 0.0}
    npos = 0

    for f in files:
        tid = f.replace(".txt", "")
        tpath = os.path.join(base_t, f)
        gpath = os.path.join(base_g, f"{tid}.gold.json")

        if not os.path.exists(gpath):
            raise RuntimeError(f"Missing gold file for {tid}: {gpath}")

        mom = build_mom(tpath, meeting_title=tid)
        pred = extract_strings(mom)

        gold_raw = load_json(gpath)
        gold = extract_gold_strings(gold_raw)

        d = prf1(pred["decisions"], gold["decisions"])
        a = prf1(pred["action_items"], gold["action_items"])

        all_dec.append(d)
        all_action.append(a)

        # Evidence coverage
        cov = evidence_coverage(mom)
        cov_list.append(cov["coverage"])

        # Positional distribution (requires meta.total_turns)
        if hasattr(mom, "meta") and mom.meta and mom.meta.total_turns:
            pos = evidence_position_stats(
                total_turns=mom.meta.total_turns,
                decisions=mom.decisions,
                actions=mom.action_items,
            )
            pos_accum["early"] += pos["early"]
            pos_accum["middle"] += pos["middle"]
            pos_accum["late"] += pos["late"]
            npos += 1

        print(f"\n== {tid} ==")
        print("Decisions PRF1:", d)
        print("Actions   PRF1:", a)
        print("Evidence coverage:", {"coverage": cov["coverage"], "num_items": cov["num_items"]})
        if npos:
            print("Evidence position:", pos)

    print("\n==== OVERALL ====")
    print("Decisions avg:", {k: avg(all_dec, k) for k in ["precision", "recall", "f1"]})
    print("Actions   avg:", {k: avg(all_action, k) for k in ["precision", "recall", "f1"]})

    if cov_list:
        print("Evidence coverage avg:", sum(cov_list) / len(cov_list))

    if npos:
        print("Evidence position avg:", {k: pos_accum[k] / npos for k in pos_accum})


if __name__ == "__main__":
    main()