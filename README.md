# Shopping Copilot

Shopping Copilot is a conversational retrieval system built for the
TechJam Conversational E-Commerce Search Challenge. It implements the official
`starter.agent.Agent` interface. The default `offline` mode runs without an API
key or network connection. `llm_parse` adds DeepSeek extraction to the existing
ranker and dialogue policy. `react` adds a conditional tool loop for insufficient
or conflicting results.

The agent keeps dialogue state across turns, distinguishes buying from browsing,
handles intent changes, retrieves candidates through several lexical routes, and
reranks the merged pool. The submitted default uses a small logistic model and an
adaptive first-turn deferral policy. Larger LambdaRank models are included for
comparison but are not enabled by default.

## DeepSeek ReAct mode

ReAct here means **model-directed action/observation iteration**, not the React
frontend framework. Compact LLM extraction preserves original phrases, then the
existing retrieval, Logistic ordering and clarification policy run. Informative
requests use this fast path; insufficient/conflicting results enter a model-led
tool loop. No additional runtime dependency is required for the HTTP adapter.

```text
User + state → LLM extraction + original words → retrieval + Logistic
                                                 ↓
                     sufficient → candidates + structured question
                     conflict → ReAct actions / observations → response
```

Configure your key locally (do not commit it), then start an interactive session:

```bash
cp .env.example .env
# Edit .env and set DEEPSEEK_API_KEY.
set -a
source .env
set +a
python3 scripts/demo_session.py --mode react --interactive --trace
```

Use `/reset` for a new conversation and `/quit` to exit. For example, start with
`帮我找一双红色真皮运动鞋，预算100以内`, then change a requirement with
`不要红色了，换黑色`. The model is instructed to translate search terms into English
for this catalog and answer in the user's language. Search candidates are not
guaranteed matches; the model must inspect evidence and disclose unknown facts.

`deepseek-flash` and the endpoint follow the current
[DeepSeek API documentation](https://api-docs.deepseek.com/); override
`DEEPSEEK_MODEL` for another tool-capable model available to your account.
The adapter uses the documented
[non-thinking tool-call mode](https://api-docs.deepseek.com/guides/tool_calls/).
ReAct does not require exposing private model reasoning.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SHOPPING_COPILOT_MODE` | `offline` | `offline`, `llm_parse`, or `react` |
| `DEEPSEEK_API_KEY` | empty | Read only when making LLM requests |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | HTTPS API root; `/v1` is also supported |
| `DEEPSEEK_MODEL` | `deepseek-flash` | Model with function calling |
| `SHOPPING_COPILOT_MAX_STEPS` | `8` | Maximum exploration decisions after extraction (1–20) |
| `SHOPPING_COPILOT_LLM_TIMEOUT` | `30` | Timeout in seconds for each HTTP attempt |
| `SHOPPING_COPILOT_LLM_MAX_TOKENS` | `1600` | Maximum output tokens per decision |
| `SHOPPING_COPILOT_LLM_FALLBACK` | `true` | On provider failure/step exhaustion, run the offline pipeline |
| `SHOPPING_COPILOT_RANKING_POLICY` | `logistic` | Preserve local order; `llm_filter` is experimental |
| `SHOPPING_COPILOT_QUERY_EXPANSION` | `false` | Enable automatic additional recall from LLM parsing |

`.env` loading is explicit; the library itself only reads environment variables.
The HTTP client retries a timeout, HTTP 429 or 5xx once, but does not retry
authentication failures. Extraction normally takes one call, with one validation
retry. If needed, up to eight additional exploration decisions run, with up to
four tool calls per decision. Timeout applies per attempt, not to the whole
turn. `usage` sums token counts from successfully parsed LLM responses,
including those preceding an offline fallback. Failed/malformed API responses
may still incur provider charges that cannot be counted locally.

`agent.last_trace` contains tool names and observations; `agent.last_error`
reports an offline fallback reason. Both describe the most recent call and stay
outside the official response schema. For live integration debugging, disable
fallback with `SHOPPING_COPILOT_LLM_FALLBACK=false` to surface `LLMError` directly.

See [ReAct implementation and limits](docs/react.md) for tool contracts, Python
configuration, failure handling, and tests. The metrics below describe the
**offline baseline only**. See [the paired small-batch experiment](docs/llm_optimization.md)
for live scores, sample selection and token use. To run a cached comparison:

```bash
python3 scripts/compare_llm_modes.py --env-file .env \
  --output artifacts/llm-ablation --samples 20 --filter-samples 5 \
  --workers 2 --max-api-calls 250
```

The same sampled sessions compare offline, LLM parsing, adaptive ReAct and
automatic expansion. Filtering is tested on a five-session subset. Identical
requests reuse a local response cache; replayed tokens are separated from new
API use. Original queries stay active, and explicit ReAct searches still add a
recall route when automatic expansion is off.

## Offline results

The table below reports the official deterministic evaluator on the 200-session
public development set.

| Metric | Starter | Manual score | Logistic + adaptive (default) | Precise LambdaRank | Wide LambdaRank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hit Rate@10 | 0.1250 | 0.9450 | 0.9950 | 0.9950 | **1.0000** |
| MRR | 0.068034 | 0.517153 | 0.791000 | 0.9950 | **0.9975** |
| MTTC | 9.81 | 3.655 | 2.510 | 2.130 | **1.690** |
| Efficiency | 0.1190 | 0.7345 | 0.8490 | 0.8870 | **0.9310** |
| TechnicalScore | 0.106710 | 0.774546 | 0.904600 | 0.973400 | **0.985450** |

These are development-set results, not estimates of the private score. The two
LambdaRank models fit the public set closely and show a clear train-validation
gap. In user-profile-grouped cross-validation, their TechnicalScores are lower
than their full-fit results. The default logistic model with adaptive deferral
scores `0.9054 ± 0.0159` under the same stricter protocol, which is why it is the
submission default. The saved aggregate output is in
[`docs/framework_results.json`](docs/framework_results.json), and the full
experiment record is in [`docs/model_experiments.md`](docs/model_experiments.md).

## How it works

Each turn passes through the following stages:

1. Classify the message as buying, browsing, or an intent override.
2. Extract the category, constraints, preferences, and negative requirements.
3. Update the session state and build a query from the current conversation.
4. Retrieve candidates through keyword, category, broad-constraint,
   conjunctive-constraint, and optional semantic routes.
5. Merge the route rankings with Reciprocal Rank Fusion.
6. Rerank the candidate pool with the configured model.
7. Return up to ten products and, when useful, one clarification question.

The conjunctive route is useful for long product-feature phrases: all informative
terms must occur in one catalog fragment. Precise-route leaders are also kept in
the rerank pool so a strong lexical match is not lost during fusion.

The default reranker is an 11-feature logistic model stored in
`shopping_copilot/reranker_lr.json`. It uses only the Python standard library at
runtime. Two packaged 36-feature LightGBM models can be selected explicitly for
experiments.

See [`docs/architecture.md`](docs/architecture.md) for the state transitions and
component boundaries.

## Setup

Python 3.10 or later is recommended.

```bash
python3 -m pip install -e .
```

On macOS, LightGBM may require OpenMP:

```bash
brew install libomp
```

Download `catalog.jsonl.gz` and `SHA256SUMS` from the official
[Participant Kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit),
then verify and unpack the catalog:

```bash
shasum -a 256 -c SHA256SUMS
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

The expected catalog size is 50,000 rows. `data/catalog.jsonl` is ignored by Git
and should remain outside the submission history.

## Run and verify

Run the test suite:

```bash
python3 -m unittest discover -v
```

Reproduce the public-set result:

```bash
python3 -m evaluator.local_evaluator --output results.json
python3 scripts/analyze_results.py results.json
```

Run a short multi-turn example:

```bash
python3 scripts/demo_session.py
```

Custom turns can be passed from the command line:

```bash
python3 scripts/demo_session.py \
  --message "I'm looking for Shoes Fashion Sneakers, but I'm still exploring." \
  --message "For that, what matters is: leather."
```

## Reranker selection

Choose a reranker with `SHOPPING_COPILOT_RERANKER`:

```bash
export SHOPPING_COPILOT_RERANKER=logistic   # default
export SHOPPING_COPILOT_RERANKER=manual
export SHOPPING_COPILOT_RERANKER=lambdarank
export SHOPPING_COPILOT_RERANKER=wide
```

`logistic` is the submission setting. `lambdarank` uses the 120-candidate model;
`wide` reranks 300 candidates. If a LightGBM model cannot be loaded, the agent
falls back to the logistic scorer and then to the manual score.

The first-turn policy is controlled separately:

```bash
export SHOPPING_COPILOT_DEFERRAL=adaptive   # default
export SHOPPING_COPILOT_DEFERRAL=none
export SHOPPING_COPILOT_DEFERRAL=all
```

## Experiment reproduction

Install the experiment dependencies before retraining or running cross-validation:

```bash
python3 -m pip install -r requirements-experiments.txt
```

The main experiment entry points are:

```bash
python3 scripts/compare_models.py
python3 -m experiments.feature_select
python3 -m experiments.validate_subset
python3 -m experiments.weight_stability
python3 -m experiments.hard_negatives
python3 -m scripts.cv_regularized_lambdarank
python3 -m scripts.cv_deferral_policy
```

The validation folds are grouped by session or user profile, depending on the
experiment, so turns from the same group do not appear in both training and
validation.

## Repository map

```text
starter/agent.py                       evaluator entry point
shopping_copilot/agent.py              turn orchestration and deferral policy
shopping_copilot/llm.py                DeepSeek HTTP adapter and configuration
shopping_copilot/react.py              model-directed tool loop and validation
shopping_copilot/state.py              session state
shopping_copilot/intent.py             intent routing
shopping_copilot/slots.py              constraint extraction
shopping_copilot/context.py            query construction
shopping_copilot/retrieval.py          retrieval, fusion, and reranking
shopping_copilot/features.py           shared 36-feature candidate vector
shopping_copilot/learned_reranker.py   packaged model loading and fallback
shopping_copilot/rerankers.py          training-time model implementations
experiments/                            dataset, feature, and validation tools
scripts/                                evaluation and training commands
tests/                                  evaluator and framework tests
```

## Known limits

- The semantic route is an interface only; no dense model is bundled.
- Constraint extraction is rule-based and tuned to the challenge's clean-text
  format rather than unrestricted shopping queries.
- Candidate counts describe the fused retrieval pool, not an exact catalog-wide
  filtered count.
- Public-set tuning may not transfer to private sessions. The default model was
  chosen using grouped validation rather than the highest public full-fit score.

## Data attribution

The challenge data is derived from Amazon Reviews 2023 by McAuley Lab, UCSD.
See [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) for attribution and redistribution
terms.
