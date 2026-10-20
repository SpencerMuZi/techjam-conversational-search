from __future__ import annotations

import re


def requires_budget_ceiling(state) -> bool:
    values = state.hard_slots.get("budget", [])
    # An approximate target price is not an upper bound. The original scorer
    # already handles its tolerance; do not turn it into an invented ceiling.
    return bool(values) and not all(re.search(r"\b(around|about|approximately)\b|左右|大约", value, re.I)
                                    for value in values)
