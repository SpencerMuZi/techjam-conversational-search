#!/usr/bin/env python3
"""Search simple confidence gates from paired immediate/deferred evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def utility(session: dict) -> float:
    turn = session["first_hit_turn"] if session["first_hit_turn"] is not None else 11
    return 0.5 * float(session["hit"]) + 0.3 * session["reciprocal_rank"] - 0.02 * turn


def score(rows: list[dict]) -> float:
    return 0.22 + sum(utility(row) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("immediate")
    parser.add_argument("deferred")
    parser.add_argument("--scenario", default="buying")
    args = parser.parse_args()

    immediate = json.loads(Path(args.immediate).read_text())
    deferred = json.loads(Path(args.deferred).read_text())
    base = {row["sample_id"]: row for row in immediate["sessions"]}
    delayed = {row["sample_id"]: row for row in deferred["sessions"]}
    traces = immediate["score_traces"]

    scenarios = set(args.scenario.split(","))
    candidates = [
        sid for sid, row in base.items() if row["scenario_type"] in scenarios
    ]
    features = {}
    for sid in candidates:
        first = traces[sid][0]
        scores = first["scores"]
        features[sid] = {
            "top": scores[0] if scores else float("-inf"),
            "margin": first["margin_1_2"] if first["margin_1_2"] is not None else float("inf"),
        }

    top_values = sorted({features[sid]["top"] for sid in candidates})
    margin_values = sorted({features[sid]["margin"] for sid in candidates})
    best = []

    def evaluate(name, gate) -> None:
        chosen = [
            delayed[sid] if sid in features and gate(features[sid]) else base[sid]
            for sid in base
        ]
        gated = [sid for sid in candidates if gate(features[sid])]
        best.append((score(chosen), name, len(gated)))

    evaluate("never", lambda f: False)
    evaluate(f"always_{args.scenario}", lambda f: True)
    for threshold in top_values:
        evaluate(f"top<={threshold:.6f}", lambda f, t=threshold: f["top"] <= t)
        evaluate(f"top>={threshold:.6f}", lambda f, t=threshold: f["top"] >= t)
    for threshold in margin_values:
        evaluate(f"margin<={threshold:.6f}", lambda f, t=threshold: f["margin"] <= t)
        evaluate(f"margin>={threshold:.6f}", lambda f, t=threshold: f["margin"] >= t)
    for top in top_values:
        for margin in margin_values:
            evaluate(
                f"top<={top:.6f}&margin<={margin:.6f}",
                lambda f, s=top, m=margin: f["top"] <= s and f["margin"] <= m,
            )

    oracle = [
        delayed[sid]
        if sid in features and utility(delayed[sid]) > utility(base[sid])
        else base[sid]
        for sid in base
    ]
    print(f"baseline: {score(list(base.values())):.6f}")
    print(f"oracle per-session gate: {score(oracle):.6f}")
    for value, name, count in sorted(best, reverse=True)[:20]:
        print(f"{value:.6f}  gated={count:2d}  {name}")


if __name__ == "__main__":
    main()
