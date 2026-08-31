"""Headless multi-turn Agent demo.

Two modes:

* scripted   --message "..." --message "..."   (no evaluator dependency)
* replay     --sample public_0050              (replays one public dev session
                                                through the OFFICIAL customer
                                                simulator, for the required
                                                "one demonstrated multi-turn
                                                session" deliverable)

Replay mode imports read-only helpers from ``evaluator.local_evaluator`` purely
to reproduce the deterministic customer policy for display. It does not modify
the evaluator, and the ground-truth ``parent_asin`` is used only to annotate the
transcript -- it is never passed to the Agent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from starter.agent import Agent

DEFAULT_MESSAGES = [
    "I'm looking for Shoes Fashion Sneakers, but I'm still exploring.",
    "For that, what matters is: leather.",
    "For that, what matters is: cushioned comfort.",
]
DEFAULT_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort", "material"],
    "summary": "Prior purchases emphasize fit, comfort, and material.",
}

RULE = "-" * 72


def _title(products: dict, asin: str) -> str:
    p = products.get(asin) or {}
    return str(p.get("title") or asin)[:88]


def _print_turn(turn, user_message, response, ranked, target, pool_size, products):
    print(f"\n{RULE}\n  Turn {turn}")
    print(f"  CUSTOMER : {user_message}")
    ask = response.get("ask_attribute")
    print(f"  AGENT    : {response.get('message', '')}")
    if ask:
        print(f"             (ask_attribute = {ask})")
    raw_recs = response.get("recommendations") or []
    if ask and not raw_recs:
        print("  (holding recommendations for one turn -- asking for a stronger")
        print("   preference before committing to a ranking)")
    top = ranked[:5]
    if top:
        print("  Top matches:")
        for i, asin in enumerate(top, start=1):
            mark = "  <-- TARGET" if target and asin == target else ""
            print(f"    {i}. {asin}  {_title(products, asin)}{mark}")
    if pool_size is not None:
        print(f"  candidate pool: {pool_size}")


def run_scripted(args) -> None:
    agent = Agent(args.catalog)
    agent.reset("demo_session", DEFAULT_PROFILE)
    for turn, user_message in enumerate(args.messages or DEFAULT_MESSAGES, start=1):
        response = agent.respond("demo_session", user_message, turn, 10)
        print(f"\nTurn {turn}\nUSER: {user_message}\nAGENT:")
        print(json.dumps(response, indent=2))


def run_replay(args) -> None:
    from evaluator.local_evaluator import (
        MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply,
        initial_message, load_jsonl, materialize_hidden_fields, normalize_recommendations,
    )

    samples = {s["sample_id"]: s for s in load_jsonl(args.dataset)}
    if args.sample not in samples:
        raise SystemExit(f"unknown sample {args.sample!r}; e.g. public_0001 .. public_0200")
    sample = samples[args.sample]
    catalog_ids, categories, products = catalog_index(args.catalog)

    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    agent = Agent(args.catalog)
    agent.reset(args.sample, sample["user_profile"])  # only the anonymized profile

    print(RULE)
    print(f"  sample {args.sample}  |  scenario: {sample['scenario_type']}")
    print(f"  hidden target: {target}  {_title(products, target)}")
    print(f"  profile tags : {sample['user_profile'].get('preference_tags')}")

    disclosed, boundary_used = set(), False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
    hit_turn = best_rank = None

    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(args.sample, user_message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        pool = getattr(agent.retriever, "last_candidates", None)
        pool_size = len(pool) if pool else None
        _print_turn(turn, user_message, response, ranked, target, pool_size, products)

        if override_applied and target in ranked:
            hit_turn = turn
            best_rank = ranked.index(target) + 1
            break
        if turn == MAX_TURNS:
            break
        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            if str(override.get("new_value", "")):
                disclosed.add(str(override["new_value"]))
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )

    print(f"\n{RULE}")
    if hit_turn:
        print(f"  RESULT: target found at rank {best_rank}, turn {hit_turn}  (session ends)")
    else:
        print("  RESULT: target not in Top-10 after 10 turns  (miss)")
    print(RULE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a headless multi-turn Agent demo")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sample", help="replay a public dev session, e.g. public_0050")
    parser.add_argument("--message", action="append", dest="messages")
    args = parser.parse_args()
    run_replay(args) if args.sample else run_scripted(args)


if __name__ == "__main__":
    main()
