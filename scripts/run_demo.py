"""Paced walkthrough of several public sessions for the demo video.

Runs each replay in turn and pauses for Enter between them, so a single
screen recording can cover the whole "demonstrated multi-turn session"
deliverable while you narrate.

    python scripts/run_demo.py                 # default 4 sessions
    python scripts/run_demo.py public_0072 public_0050
    python scripts/run_demo.py --no-pause      # run straight through
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT = [
    ("public_0021", "Browsing - vague start, first turn deferred, target rank 1 on turn 2"),
    ("public_0050", "Boundary - 'use your judgment', agent asks 'other', target rank 1 on turn 2"),
    ("public_0072", "Intent override - customer changes their mind on turn 3, target jumps to rank 1"),
    ("public_0082", "Buying - one broad constraint, deferred, target rank 1 on turn 2"),
]


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--no-pause"]
    pause = "--no-pause" not in sys.argv[1:]
    sessions = [(s, "") for s in args] if args else DEFAULT

    for i, (sample, note) in enumerate(sessions, start=1):
        print("\n" + "=" * 72)
        print(f"  DEMO {i}/{len(sessions)}   {sample}")
        if note:
            print(f"  {note}")
        print("=" * 72)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "demo_session.py"), "--sample", sample],
            cwd=ROOT, check=True,
        )
        if pause and i < len(sessions):
            try:
                input("\n  [Enter] for the next session ... ")
            except EOFError:
                pass


if __name__ == "__main__":
    main()
