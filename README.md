# Shopping Copilot: Conversational Search Framework

An offline-first, runtime-adaptive shopping agent for the TechJam Conversational E-Commerce Search Challenge. The implementation keeps the official `starter.agent.Agent` interface while separating dialogue state, intent routing, context distillation, clarification, retrieval, and reranking into testable modules.

## Current public-set result

| Metric | Weak starter | Framework |
| --- | ---: | ---: |
| Hit Rate@10 | 0.1250 | **0.9250** |
| MRR | 0.068034 | **0.473827** |
| MTTC | 9.81 | **3.835** |
| Efficiency | 0.1190 | **0.7165** |
| TechnicalScore | 0.106710 | **0.747948** |

These numbers come from the official deterministic evaluator on the 200-session public development set. They are development results, not a claim about the private 800-session score. The aggregate snapshot is stored in `docs/framework_results.json`.

## What is implemented

- Buying/Browsing routing with explicit Intent Override detection.
- Per-session state for category, hard constraints, soft preferences, asked attributes, no-preference answers, and dialogue history.
- Runtime context distillation that rebuilds the retrieval query after every turn.
- Proactive clarification that prioritizes high-value attributes and avoids repeated questions.
- Multi-route retrieval over keyword, category, and accumulated constraints.
- Reciprocal Rank Fusion followed by structured constraint reranking.
- A pluggable semantic retrieval interface with an offline no-op fallback.
- Exact compliance with the official Agent response contract.
- Standard-library-only default runtime: no API key, model download, or network access is required.

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
shopping_copilot/semantic.py      semantic retrieval adapter boundary
scripts/analyze_results.py        compact metric reporting
scripts/demo_session.py           headless multi-turn demonstration
tests/                             evaluator and framework tests
```

## Setup

Python 3.10 or later is recommended. The default implementation only uses the Python standard library.

```bash
python3 -m pip install -e .
```

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
- Candidate-count estimation currently reflects the fused retrieval pool, not an exact filtered-catalog count.
- The next iteration should compare a small local bi-encoder and cross-encoder against this offline framework, with latency and memory reported.

## Data attribution

The competition data is derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` for the required attribution and redistribution notes.

## Team contributions

Add participant names and a concise contribution breakdown before the final Devpost submission.
