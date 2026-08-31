"""Run V4 through the official evaluator without altering the participant kit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "techjam-conversational-search-participant-kit" / "techjam-conversational-search-participant-kit"
sys.path.insert(0, str(KIT))

from agent import Agent  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402


def main() -> None:
    catalog = ROOT / "catalog.jsonl"
    samples = load_jsonl(KIT / "data" / "public_set.jsonl")
    identifiers, categories, products = catalog_index(catalog)
    result = evaluate(Agent(catalog), samples, identifiers, categories, products)
    Path(__file__).with_name("public_evaluation_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
