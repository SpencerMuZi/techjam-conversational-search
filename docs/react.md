# DeepSeek parsing and adaptive ReAct

## Modes

| Mode | Extraction | Ranking | Dialogue control |
| --- | --- | --- | --- |
| `offline` | Original rules | Existing multi-route retrieval and ranker | Original clarification and first-turn deferral |
| `llm_parse` | LLM plus original-word evidence anchoring | Same ranking, no automatic expansion | Same verified policy |
| `react` | Same parsing | Same local ranking and hard-budget filter | Fast path when sufficient; ReAct when insufficient or conflicting |

This is an adaptive workflow with a conditional ReAct sub-loop. The fixed fast
path is not autonomous planning. Within exploration, the model reads tool
observations and selects subsequent actions, their order and repetition. There
is no separate Plan-and-Execute planner.

Extraction normally needs one compact API call, with one validation retry.
Mechanically extractable English category and requirement phrases anchor the
state instead of being replaced by model paraphrases. Quoted additions persist
across ordinary turns; explicit overrides clear abandoned needs. Unstructured
language still relies on LLM parsing. The model receives state and recent history,
never the evaluator's hidden target or intent card.

## Candidates plus a structured question

Both terminal tools support ranked candidates plus one follow-up. `finish`
accepts `ask_attribute` and `question` together. A question in `message` is
rejected. The renderer appends one canonical English/Chinese question based on
`ask_attribute`, preventing a sleeve-length question from being labeled `other`.
Declined attributes cannot be asked again.

Default `ranking_policy=logistic` returns up to ten eligible locally ranked
candidates even when the model proposes only two or three IDs. Model scores and
ordering do not replace local scores. `llm_filter` is an explicit experiment:
it may filter IDs, but still retains local score order. It is off by default.
Candidates are catalog-grounded, not proof of satisfying every constraint.

A strict hard ceiling requires numeric `max_price`. Unknown, invalid and excessive
prices are excluded, with backfill from a 50-candidate ranked pool. Approximate
prices such as `around $40` are not converted to invented ceilings. Category
changes and no-preference updates clear abandoned ceilings. Other semantic
constraints still depend on extraction and evidence checks.

## Original query and additional recall

The original message, original category and cumulative verbatim constraints
remain primary retrieval/ranking evidence. Optional translations or synonyms
form additional recall routes; route ranks enter weighted RRF. Heads of the
original precise routes and expansion routes reach the same rerank pool, using
the existing feature vector and packaged Logistic model.

Automatic parsing-time expansion defaults to off because the initial paired
sample showed no net benefit. Enable `SHOPPING_COPILOT_QUERY_EXPANSION=true` to
experiment. An explicit ReAct `search_products(query)` always adds its query as a
recall route while keeping the original context; disabling automatic expansion
does not disable purposeful ReAct searches.

## Exploration tools

| Tool | Action |
| --- | --- |
| `update_search_state` | Update hard/soft needs, category, intent, budget and no-preference markers |
| `search_products` | Add the selected search query, fuse and locally rerank |
| `get_product_details` | Read catalog evidence for up to five observed IDs |
| `ask_user` | Return eligible candidates and an attribute-based question |
| `finish` | Return eligible candidates and optionally one structured question |

Terminal tools must be called alone. Names, JSON arguments, bounds, IDs and
duplicates are validated. Validation errors become observations the model can
repair. Moving a slot into hard requirements while explicitly clearing its old
soft bucket is valid; conflicting nonempty hard/soft values are rejected.

The fast path reuses `ClarificationPolicy` and the original adaptive deferral.
The conflict check uses the existing lexical matcher on leading candidates; it
is a heuristic, not formal verification. A missing ceiling causes a question
rather than repeated searches. Sufficient results avoid extra LLM decisions.

## Configuration and failure handling

```python
import os
from starter.agent import Agent
from shopping_copilot.llm import LLMConfig

agent = Agent("data/catalog.jsonl", mode="react", llm_config=LLMConfig(
    api_key=os.environ["DEEPSEEK_API_KEY"], model="deepseek-flash",
    ranking_policy="logistic", expansion_enabled=False, fallback=False,
))
agent.reset("customer-1", {"summary": "", "preference_tags": []})
response = agent.respond("customer-1", "想买真皮运动鞋，预算100以内", 1, 10)
```

State changes run against a copy and commit once after a completed fast response
or terminal tool. Provider/protocol failure rolls back before optional offline
fallback. Disable fallback for live evaluation. The fallback rule parser has
limited open-language coverage. `last_trace` records parsing, retrieval, the
exploration gate, actions and policy output outside the official response schema.

The adapter follows [DeepSeek's non-thinking tool-call protocol](https://api-docs.deepseek.com/guides/tool_calls/).
It does not log API keys or private `reasoning_content`. Timeout is per HTTP
attempt, not a turn deadline. A turn may use up to two extraction calls plus
`max_steps` further decisions (default eight). Timeouts, 429 and 5xx responses
retry once; authentication errors do not. Usage sums parsed responses, including
those preceding a later failure. Unparsed provider charges may remain unknown.

State is in memory. Use each SQLite-based instance serially; benchmark processes
own separate instances. Exploration uses six recent history messages (1,600
characters each), structured state and truncated product evidence. No purchase,
payment or web-search action is included.

## Verification

```bash
python3 -m unittest discover -v
python3 scripts/compare_llm_modes.py --env-file .env \
  --output artifacts/llm-ablation --samples 20 --filter-samples 5 \
  --workers 2 --max-api-calls 250
```

The small benchmark fixes stratified sample IDs, reuses identical API responses,
preserves official scoring, and separates new usage from cached replay. It also
compares expansion/no-expansion at common retrieval contexts without extra API
calls; those diagnostics are not a replacement for full-session comparisons.
Raw outputs remain in ignored `artifacts/`. See the
[paired experiment report](llm_optimization.md) for results and limitations.
