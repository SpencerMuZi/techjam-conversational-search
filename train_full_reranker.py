"""Train the V4 reranker on all public development sessions.

Use `run_xgboost_cv.py` from the experiment folder to decide whether the model
is appropriate before running this final training step.
"""
from pathlib import Path

import train_reranker as experiment


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    catalog_path = root / "catalog.jsonl"
    samples = experiment.load_jsonl(experiment.KIT / "data" / "public_set.jsonl")
    _, categories, _ = experiment.catalog_index(catalog_path)
    products = experiment.load_catalog(catalog_path)
    agent = experiment.Agent(catalog_path)
    contexts = experiment.build_contexts(agent, samples, products, categories)
    ids = [sample["sample_id"] for sample in samples]
    model = experiment.fit_ranker(contexts, ids)
    output = Path(__file__).with_name("xgboost_reranker.json")
    model.save_model(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
