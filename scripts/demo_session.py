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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a headless multi-turn Agent demo")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--message", action="append", dest="messages")
    args = parser.parse_args()

    agent = Agent(args.catalog)
    session_id = "demo_session"
    agent.reset(session_id, {
        "purchase_frequency": "3-4 prior purchases",
        "average_prior_rating": 4.5,
        "rating_style": "usually positive",
        "preference_tags": ["fit", "comfort", "material"],
        "summary": "Prior purchases emphasize fit, comfort, and material.",
    })

    for turn, user_message in enumerate(args.messages or DEFAULT_MESSAGES, start=1):
        response = agent.respond(session_id, user_message, turn, 10)
        print(f"\nTurn {turn}\nUSER: {user_message}\nAGENT:")
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
