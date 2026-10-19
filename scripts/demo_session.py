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
    parser.add_argument("--mode", choices=["offline", "react"], default=None)
    parser.add_argument("--interactive", action="store_true", help="Chat until /quit; /reset starts a new session")
    parser.add_argument("--trace", action="store_true", help="Print actions and observations (no private reasoning)")
    args = parser.parse_args()

    agent = Agent(args.catalog, mode=args.mode)
    session_id = "demo_session"
    profile = {
        "purchase_frequency": "3-4 prior purchases",
        "average_prior_rating": 4.5,
        "rating_style": "usually positive",
        "preference_tags": ["fit", "comfort", "material"],
        "summary": "Prior purchases emphasize fit, comfort, and material.",
    }
    agent.reset(session_id, profile)

    def respond(user_message, turn):
        response = agent.respond(session_id, user_message, turn, 10)
        print(f"\nTurn {turn}\nUSER: {user_message}\nAGENT:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        if agent.last_error:
            print(f"Offline fallback: {agent.last_error}", file=sys.stderr)
        if args.trace:
            print(json.dumps(agent.last_trace, indent=2, ensure_ascii=False))

    if args.interactive:
        turn = 1
        while True:
            try:
                message = input("\nYou (/quit, /reset): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if message == "/quit":
                break
            if message == "/reset":
                agent.reset(session_id, profile)
                turn = 1
            elif message:
                respond(message, turn)
                turn += 1
        return

    for turn, user_message in enumerate(args.messages or DEFAULT_MESSAGES, start=1):
        respond(user_message, turn)


if __name__ == "__main__":
    main()
