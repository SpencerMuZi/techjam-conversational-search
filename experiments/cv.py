"""Session-grouped cross-validation of reranking strategies.

For every fold we train on the rows from the training sessions (with negative
subsampling) and then run the *real* evaluator on the held-out sessions with the
trained reranker plugged in, so the reported numbers are Hit@10 / MRR / MTTC, not
classification accuracy.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from experiments.dataset import RerankDataset, _agent_with_index, build_dataset
from shopping_copilot.features import FEATURE_NAMES
from shopping_copilot.rerankers import build_reranker
from shopping_copilot.retrieval import CatalogIndex

METHODS = ("manual", "logistic", "forest", "lambdamart")
_FUSED_IDX = FEATURE_NAMES.index("rrf_fused_score")


def sample_training_rows(
    data: RerankDataset,
    session_mask: np.ndarray,
    n_hard_neg: int = 24,
    n_rand_neg: int = 8,
    seed: int = 0,
    include_mined: bool = False,
) -> np.ndarray:
    """Row indices for training: every positive + hard/random negatives per group.

    Groups (session,turn) without a positive are dropped – an unretrieved target
    is not something the reranker can learn to fix. ``include_mined`` adds every
    catalog-mined confusable negative for the surviving groups.
    """
    rng = np.random.default_rng(seed)
    in_split = session_mask[data.session_idx]
    mined = data.is_mined
    keep: list[int] = []
    for g in np.unique(data.group_id[in_split]):
        idx = np.where((data.group_id == g) & in_split)[0]
        pos = idx[data.y[idx] == 1]
        if pos.size == 0:
            continue
        keep.extend(pos.tolist())
        pool_neg = idx[(data.y[idx] == 0) & (mined[idx] == 0)]
        if pool_neg.size:
            order = np.argsort(-data.X[pool_neg, _FUSED_IDX], kind="stable")
            hard = pool_neg[order[:n_hard_neg]]
            rest = pool_neg[order[n_hard_neg:]]
            keep.extend(hard.tolist())
            if rest.size:
                pick = rng.choice(rest, size=min(n_rand_neg, rest.size), replace=False)
                keep.extend(np.asarray(pick, dtype=int).tolist())
        if include_mined:
            keep.extend(idx[mined[idx] == 1].tolist())
    return np.array(sorted(set(keep)), dtype=int)


@dataclass
class FoldResult:
    method: str
    fold: int
    hit_rate_at_10: float
    mrr: float
    mttc: float
    scenario: dict
    train_seconds: float
    infer_ms_per_turn: float
    roc_auc: float = float("nan")
    pr_auc: float = float("nan")


@dataclass
class Comparison:
    folds: list[FoldResult] = field(default_factory=list)
    full_fit: dict = field(default_factory=dict)
    importances: dict = field(default_factory=dict)

    def summary(self) -> dict:
        scen_names = sorted({s for f in self.folds for s in f.scenario})
        out: dict = {}
        for method in METHODS:
            rows = [f for f in self.folds if f.method == method]
            out[method] = {
                "hit_rate_at_10": _mean_std([r.hit_rate_at_10 for r in rows]),
                "mrr": _mean_std([r.mrr for r in rows]),
                "mttc": _mean_std([r.mttc for r in rows]),
                "train_seconds": _mean_std([r.train_seconds for r in rows]),
                "infer_ms_per_turn": _mean_std([r.infer_ms_per_turn for r in rows]),
                "roc_auc": _mean_std([r.roc_auc for r in rows if r.roc_auc == r.roc_auc]),
                "pr_auc": _mean_std([r.pr_auc for r in rows if r.pr_auc == r.pr_auc]),
                "scenario": {
                    scen: {
                        "hit_rate_at_10": _mean_std(
                            [r.scenario[scen]["hit_rate_at_10"] for r in rows if scen in r.scenario]
                        ),
                        "mrr": _mean_std(
                            [r.scenario[scen]["mrr"] for r in rows if scen in r.scenario]
                        ),
                    }
                    for scen in scen_names
                },
            }
        return out


def _mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "per_fold": []}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "per_fold": [float(v) for v in arr]}


def _timed_eval(agent, samples, cids, cats, prods) -> tuple[dict, float]:
    start = time.perf_counter()
    result = evaluate(agent, samples, cids, cats, prods)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    # Amortise wall time over the turns actually run (a hit ends the session early).
    total_turns = sum((s["first_hit_turn"] or 10) for s in result["sessions"])
    return result, elapsed_ms / max(total_turns, 1)


def run(catalog_path: str = "data/catalog.jsonl", dataset_path: str = "data/public_set.jsonl",
        n_splits: int = 5, seed: int = 0, checkpoint_path: str | None = None,
        skip_full_fit: bool = False) -> Comparison:
    index = CatalogIndex(catalog_path)
    samples = load_jsonl(dataset_path)
    cids, cats, prods = catalog_index(catalog_path)
    data = build_dataset(catalog_path, dataset_path, index=index)

    n_sessions = int(data.session_idx.max()) + 1
    comparison = Comparison()
    checkpoint = Path(checkpoint_path) if checkpoint_path else None

    def save_checkpoint() -> None:
        if checkpoint is None:
            return
        checkpoint.write_text(json.dumps({
            "folds": [asdict(f) for f in comparison.folds],
            "full_fit": comparison.full_fit,
            "importances": comparison.importances,
        }, indent=2) + "\n", encoding="utf-8")

    # Stratify folds by scenario so every fold carries ~2 boundary / ~6 override
    # sessions instead of a random count that makes per-scenario CV meaningless.
    session_scenario = np.empty(n_sessions, dtype=object)
    for idx, scen in zip(data.session_idx, data.scenario):
        session_scenario[idx] = scen
    session_ids = np.arange(n_sessions)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_sess, test_sess) in enumerate(
        splitter.split(session_ids, session_scenario, groups=session_ids)
    ):
        train_mask = np.zeros(n_sessions, dtype=bool)
        train_mask[train_sess] = True
        test_mask = ~train_mask
        rows = sample_training_rows(data, train_mask, seed=seed)
        X_tr, y_tr, g_tr = data.X[rows], data.y[rows], data.group_id[rows]
        # Held-out rows for a classification-quality sanity check (all candidates,
        # not subsampled) restricted to turns where the target was retrieved.
        test_rows = sample_training_rows(data, test_mask, n_hard_neg=10**9, n_rand_neg=0, seed=seed)
        X_te, y_te = data.X[test_rows], data.y[test_rows]
        test_samples = [samples[i] for i in test_sess]

        for method in METHODS:
            reranker = build_reranker(method, random_state=seed)
            t0 = time.perf_counter()
            if getattr(reranker, "trainable", False):
                if method == "lambdamart":
                    reranker.fit(X_tr, y_tr, g_tr)
                else:
                    reranker.fit(X_tr, y_tr)
            train_seconds = time.perf_counter() - t0

            roc = pr = float("nan")
            if y_te.sum() and len(set(y_te.tolist())) == 2:
                from sklearn.metrics import average_precision_score, roc_auc_score
                preds = np.asarray(reranker.score(X_te), dtype=float)
                roc = float(roc_auc_score(y_te, preds))
                pr = float(average_precision_score(y_te, preds))

            agent = _agent_with_index(index)
            agent.retriever.capture = False
            agent.retriever.reranker = reranker
            result, infer_ms = _timed_eval(agent, test_samples, cids, cats, prods)
            comparison.folds.append(FoldResult(
                method=method, fold=fold,
                hit_rate_at_10=result["hit_rate_at_10"], mrr=result["mrr"], mttc=result["mttc"],
                scenario={k: {"hit_rate_at_10": v["hit_rate_at_10"], "mrr": v["mrr"]}
                          for k, v in result["scenario_metrics"].items()},
                train_seconds=train_seconds, infer_ms_per_turn=infer_ms,
                roc_auc=roc, pr_auc=pr,
            ))
            print(f"  fold {fold} {method:11s} hit={result['hit_rate_at_10']:.4f} "
                  f"mrr={result['mrr']:.4f} mttc={result['mttc']:.3f}", flush=True)
        save_checkpoint()
        print(f"fold {fold} done", flush=True)

    if skip_full_fit:
        return comparison

    # Full-fit reference + feature importances (optimistically biased; flagged in report).
    full_rows = sample_training_rows(data, np.ones(n_sessions, dtype=bool), seed=seed)
    Xf, yf, gf = data.X[full_rows], data.y[full_rows], data.group_id[full_rows]
    for method in METHODS:
        reranker = build_reranker(method, random_state=seed)
        if getattr(reranker, "trainable", False):
            reranker.fit(Xf, yf, gf) if method == "lambdamart" else reranker.fit(Xf, yf)
        agent = _agent_with_index(index)
        agent.retriever.reranker = reranker
        res = evaluate(agent, samples, cids, cats, prods)
        comparison.full_fit[method] = {
            "hit_rate_at_10": res["hit_rate_at_10"], "mrr": res["mrr"], "mttc": res["mttc"],
            "technical_score": res["recommended_technical_score"],
        }
        comparison.importances[method] = _importance(method, reranker, data.feature_names)
        save_checkpoint()
        print(f"full-fit {method:11s} hit={res['hit_rate_at_10']:.4f} mrr={res['mrr']:.4f} "
              f"tech={res['recommended_technical_score']:.4f}", flush=True)
    return comparison


def _importance(method: str, reranker, names: list[str]) -> list[tuple[str, float]]:
    try:
        if method == "logistic":
            coef = reranker.estimator.named_steps["lr"].coef_[0]
            pairs = sorted(zip(names, (abs(c) for c in coef)), key=lambda x: -x[1])
        elif method == "forest":
            imp = reranker.estimator.feature_importances_
            pairs = sorted(zip(names, imp), key=lambda x: -x[1])
        elif method == "lambdamart":
            imp = reranker.model.feature_importances_
            total = sum(imp) or 1
            pairs = sorted(zip(names, (i / total for i in imp)), key=lambda x: -x[1])
        else:
            return []
        return [(n, float(v)) for n, v in pairs[:8]]
    except Exception as exc:  # pragma: no cover - diagnostics only
        return [("<error>", 0.0)]
