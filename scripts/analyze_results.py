from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact TechJam evaluation summary")
    parser.add_argument("results", nargs="?", default="results.json")
    args = parser.parse_args()
    result = json.loads(Path(args.results).read_text(encoding="utf-8"))

    print("overall")
    for key in ("sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score"):
        if key in result:
            print(f"  {key}: {result[key]}")
    print("scenarios")
    for scenario, metrics in result.get("scenario_metrics", {}).items():
        print(
            f"  {scenario:16} "
            f"hit={metrics['hit_rate_at_10']:.6f} "
            f"mrr={metrics['mrr']:.6f} "
            f"mttc={metrics['mttc']:.6f}"
        )


if __name__ == "__main__":
    main()
