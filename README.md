# SlotState Shopping Agent V4

V4 extends the offline V3 dialogue-state and BM25 retrieval pipeline with an
optional XGBoost pairwise-ranking reranker. It preserves a complete V3 fallback
for environments that do not install XGBoost or do not provide the trained model.

## Files

- `agent.py`: required submission entry point exporting `Agent`.
- `v3_core.py`: deterministic slot-state, retrieval, and fallback implementation.
- `xgboost_reranker.json`: trained reranking model produced from public development data.
- `xgboost_cv_results.json`: leakage-safe five-fold development validation report.
- `train_reranker.py` and `train_full_reranker.py`: reproducible training utilities; do not run during official scoring.

## Runtime

Place `agent.py`, `v3_core.py`, and `xgboost_reranker.json` beside the official
participant kit's `starter/agent.py`, then install:

```bash
pip install -r requirements.txt
```

The required frozen catalog remains external and must not be committed. If the
model or XGBoost is absent, `Agent` automatically serves the V3 ranking instead
of failing.

## Ranking

V3 retrieves 300 candidates with SQLite FTS5/BM25. The optional XGBoost model
reranks them using only runtime-safe features: V3 score/rank, text and slot
coverage, budget fit, profile affinity, rating, rating count, price presence,
and conversation state. It does not receive target ASIN, sample ID, scenario
type, or difficulty as a feature.

## Validation policy

The model was investigated using five-fold, session-level cross validation:
each validation session was absent from its training fold. The model was
selected only because it improved all five validation folds over the same V3
candidate pool. Public results are development evidence, not a claim about the
organizer's private evaluation set.

The mean cross-validation TechnicalScore was 0.885662 for the XGBoost reranker,
compared with 0.817538 for the V3 ordering over the same candidate pools. The
cross-validation metrics are:

| Metric | V3 candidate ordering | V4 XGBoost reranker |
| --- | ---: | ---: |
| HitRate@10 | 0.940000 | **0.995000** |
| MRR | 0.592794 | **0.651540** |
| MTTC | 2.515000 | **1.365000** |
| Efficiency | 0.848500 | **0.963500** |
| TechnicalScore | 0.817538 | **0.885662** |
