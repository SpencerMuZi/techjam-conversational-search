# Shopping Copilot: Conversational Search Framework

An offline-first, runtime-adaptive shopping agent for the TechJam Conversational E-Commerce Search Challenge. The implementation keeps the official `starter.agent.Agent` interface while separating dialogue state, intent routing, context distillation, clarification, retrieval, and reranking into testable modules.

## Current public-set result

| Metric | Weak starter | Manual | Logistic | LambdaRank |
| --- | ---: | ---: | ---: | ---: |
| Hit Rate@10 | 0.1250 | 0.9450 | 0.9950 | **0.9950** |
| MRR | 0.068034 | 0.517153 | 0.763129 | **0.9950** |
| MTTC | 9.81 | 3.655 | 2.280 | **2.230** |
| Efficiency | 0.1190 | 0.7345 | 0.8720 | **0.8770** |
| TechnicalScore | 0.106710 | 0.774546 | 0.900839 | **0.971400** |

Official deterministic evaluator on the 200-session public development set;
development results, not a claim about the private 800-session score. Aggregate
snapshot in `docs/framework_results.json`.

The default learned reranker is a 36-feature LightGBM LambdaRank model trained
with every in-pool negative. The `0.9714` number is a full-fit public-development
score, not a private-set estimate: its session-grouped CV TechnicalScore is
`0.8832 ± 0.0236`, so it has a material overfitting gap. The smaller 11-feature
logistic model remains available as the more stable fallback. Select with
`SHOPPING_COPILOT_RERANKER=lambdarank|logistic|manual`; see
`docs/model_experiments.md`.

## What is implemented

- Buying/Browsing routing with explicit Intent Override detection.
- Per-session state for category, hard constraints, soft preferences, asked attributes, no-preference answers, and dialogue history.
- Runtime context distillation that rebuilds the retrieval query after every turn.
- Proactive clarification that prioritizes high-value attributes and avoids repeated questions.
- Multi-route retrieval over keyword, category, and accumulated constraints.
- A high-precision conjunctive constraint route that requires every informative
  term in one disclosed feature fragment, ordered by fragment specificity.
- Reciprocal Rank Fusion followed by structured constraint reranking, with an
  idf-weighted match bonus and a retrieval-rank prior so a strong BM25 hit is
  promoted toward rank 1 instead of being flattened by category peers.
- The head of each precise route is seeded straight into the reranker so a target
  BM25 ranked highly is never dropped by a thin fused score.
- A packaged LightGBM LambdaRank reranker over 36 shared features, with a ~2 KB
  standard-library logistic fallback and a manual-score fallback.
- A pluggable semantic retrieval interface with an offline no-op fallback.
- Exact compliance with the official Agent response contract.
- No API key, hosted model, model download, or network access is required at inference.

## Architecture

```text
user message
    |
    v
IntentRouter + SlotExtractor
    |
    v
ConversationState ---- override/no-preference handling
    |
    v
ContextBuilder
    |
    +--> keyword route
    +--> category route
    +--> constraint route
    +--> optional semantic route
              |
              v
        Reciprocal Rank Fusion
              |
              v
      structured reranking
              |
              +--> Top-10 recommendations
              +--> ClarificationPolicy
```

See `docs/architecture.md` for state transitions and extension points.

## Repository layout

```text
starter/agent.py                  official evaluator entry point
shopping_copilot/agent.py         orchestration
shopping_copilot/state.py         session state machine
shopping_copilot/intent.py        Buying/Browsing/Override router
shopping_copilot/slots.py         structured constraint extraction
shopping_copilot/context.py       dynamic context distillation
shopping_copilot/clarification.py question policy
shopping_copilot/retrieval.py     in-memory hybrid retrieval and reranking
shopping_copilot/features.py      shared candidate feature vector (36 features)
shopping_copilot/rerankers.py     Manual / logistic / forest / LambdaRank (experiments)
shopping_copilot/learned_reranker.py  packaged LambdaRank loader + linear fallback
shopping_copilot/reranker_lr.json 11-feature logistic-regression coefficients
shopping_copilot/reranker_lgbm.txt submitted high-capacity LambdaRank model
shopping_copilot/semantic.py      semantic retrieval adapter boundary
experiments/dataset.py            replay public sessions, capture rerank features
experiments/cv.py                 session-grouped CV harness (real evaluator per fold)
experiments/feature_select.py     L1 path + greedy forward feature selection
experiments/validate_subset.py    real-evaluator CV for feature subsets
experiments/weight_stability.py   coefficient stability across seeds x folds
experiments/hard_negatives.py     A/B: catalog-mined confusable negatives (rejected)
experiments/train_reranker.py     fit the shipped logistic model -> reranker_lr.json
scripts/analyze_results.py        compact metric reporting
scripts/compare_models.py         four-way reranker comparison report
scripts/demo_session.py           headless multi-turn demonstration
tests/                             evaluator and framework tests
```

## Setup

Python 3.10 or later is recommended. Installation includes the local LightGBM
runtime used by the high-accuracy ranker.

```bash
python3 -m pip install -e .
```

On macOS, LightGBM may also require `brew install libomp`. If LightGBM cannot be
loaded, the agent automatically uses the pure-Python logistic model.

Download `catalog.jsonl.gz` and `SHA256SUMS` from the official [Participant Kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit), verify it, and install the catalog:

```bash
shasum -a 256 -c SHA256SUMS
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Expected catalog row count: 50,000. The catalog is ignored by Git and must not be committed.

## Run tests

```bash
python3 -m unittest discover -v
```

## Reranker experiments (optional)

```bash
python3 -m pip install -r requirements-experiments.txt
python3 scripts/compare_models.py          # 4-way session-grouped CV comparison
python3 -m experiments.feature_select       # L1 path + greedy feature selection
python3 -m experiments.validate_subset      # real-evaluator CV for feature subsets
python3 -m experiments.weight_stability     # coefficient stability across seeds x folds
python3 -m experiments.hard_negatives       # A/B mined hard negatives
python3 -m experiments.train_reranker --features "<comma,separated>"   # refit shipped model
python3 -m scripts.tune_lambdarank           # capacity / negative-pool search
python3 -m scripts.cv_lambdarank_capacity    # real-evaluator grouped CV
```

`scripts/compare_models.py` compares the hand-tuned score against logistic
regression, random forest, and LightGBM LambdaRank; every fold is scored by the
real evaluator on held-out sessions. See `docs/model_experiments.md` for the
protocol and current results. Experiment scripts additionally need scikit-learn.

## Reproduce the public score

```bash
python3 -m evaluator.local_evaluator --output results.json
python3 scripts/analyze_results.py results.json
```

The first command builds the in-memory index once, evaluates 200 sessions, and writes per-session results plus aggregate metrics.

## Run a headless demo

```bash
python3 scripts/demo_session.py
```

The script demonstrates API responses without requiring a UI. It can also accept custom turns:

```bash
python3 scripts/demo_session.py \
  --message "I'm looking for Shoes Fashion Sneakers, but I'm still exploring." \
  --message "For that, what matters is: leather."
```

## Adding dense retrieval or a local reranker

`shopping_copilot.models.SemanticRetriever` is the adapter contract. A dense implementation only needs to expose:

```python
def search(self, query: str, limit: int) -> list[tuple[str, float]]:
    ...
```

Pass that implementation to `HybridRetriever`. Keep embeddings and generated indexes outside the catalog, document how they are produced, and retain the offline fallback for organizer environments without network access.

## Design choices

- Every turn returns recommendations and may ask one structured question; retrieval opportunities are never sacrificed just to ask.
- Intent Override clears intent-specific slots while preserving the stable category and anonymized profile.
- Profile tags are weak ranking signals, not hard filters, because they are broad aggregate preferences.
- Product popularity and rating are only tie-breakers. User constraints dominate the final rank.
- The frozen Amazon catalog and official evaluator are never modified.

## Limitations and next steps

- The current semantic route is an extension point rather than a bundled dense model.
- Rule-based slot extraction is aligned with the clean-text competition scope but is not a general natural-language parser.
- Public-set tuning may not transfer perfectly to private users and products; route weights should be validated with ablation tests.
- The submitted high-capacity LambdaRank model intentionally maximizes the public
  TechnicalScore and has a large full-fit/CV gap. For private-score robustness,
  compare it against `SHOPPING_COPILOT_RERANKER=logistic` before final submission.
- Candidate-count estimation currently reflects the fused retrieval pool, not an exact filtered-catalog count.
- The next iteration should compare a small local bi-encoder and cross-encoder against this offline framework, with latency and memory reported.

## Data attribution

The competition data is derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` for the required attribution and redistribution notes.

## Team contributions

Add participant names and a concise contribution breakdown before the final Devpost submission.
