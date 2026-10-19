# DeepSeek ReAct agent

## Control flow

`starter.agent.Agent` still exports `reset` and `respond`. Select `react` via
`SHOPPING_COPILOT_MODE` or the constructor. In this mode, each user turn creates
a bounded model/action/observation loop rather than running a fixed sequence.
The model sees recent dialogue, structured needs, the anonymized profile, and
previously displayed recommendations. It chooses one or more tools, receives
their results as `role=tool` messages associated with `tool_call_id`, then
chooses again. Retrieval is not mandatory before asking a question, and can be
repeated with different queries after inspecting results.

| Tool | Action |
| --- | --- |
| `update_search_state` | Set category/intent; replace or remove hard/soft attributes; record no preference; clear abandoned needs |
| `search_products` | Query existing BM25 routes → RRF → configured reranker; return up to 10 candidate summaries |
| `get_product_details` | Return more catalog evidence for up to 5 observed product IDs |
| `ask_user` | Finish the current turn with a question and allowed `ask_attribute` |
| `finish` | Finish with a message and ordered observed product IDs; scores come from the local reranker |

The last two tools are terminal and must be called alone. The controller does
not invent model decisions or run a predetermined search/details/question sequence.
Tests use a scripted client to verify orchestration; a real DeepSeek client makes
the live decisions. This is a tool-based ReAct implementation, without a
separate Plan-and-Execute planner or a frontend React application.

## State and validation

The model handles natural-language extraction through `update_search_state`.
Each supplied slot array replaces that attribute's previous values; an empty
array removes it. A hard/soft update also removes the opposite kind for the same
attribute. Categories changing or `clear=true` clear earlier needs, declined
attributes, and question bookkeeping. `clear=true` without a category also
clears the category. A no-preference update removes existing constraints for
the attribute and prevents a later question about it. Untouched attributes persist.

State updates invalidate observed recommendation IDs, requiring a fresh search.
The controller validates tool names, JSON arguments, allowed fields/types,
string/array bounds, duplicate IDs, observed IDs and `top_k`. A validation error
becomes an observation so the model can repair its action on the next decision.
Product IDs, rank scores and product facts originate in the local catalog/tools.
Prompt instructions ask the model to verify constraints; semantic correctness
of its prose and satisfaction of all constraints are not enforced by a formal
filter. Prices in the catalog may lack currency; the assistant must not invent it.

The active turn runs against a copy of session state. Only `ask_user` or `finish`
commits it. A provider/protocol failure or exhausted step limit discards partial
changes before the existing offline pipeline handles the original user message.
With fallback disabled, `LLMError` propagates and the session remains unchanged.
Fallback uses the baseline's rule parser, whose unrestricted-language coverage
is limited. The interactive demo explicitly prints the fallback reason.

## Python usage

```python
import os
from starter.agent import Agent
from shopping_copilot.llm import LLMConfig

agent = Agent(
    "data/catalog.jsonl",
    mode="react",
    llm_config=LLMConfig(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        max_steps=8,
        fallback=False,
    ),
)
agent.reset("customer-1", {"summary": "", "preference_tags": []})
response = agent.respond("customer-1", "想买红色真皮运动鞋", turn=1, top_k=10)
print(response)
print(agent.last_trace)
```

For testing or another provider adapter, inject `llm_client` implementing
`complete(messages, tools) -> LLMReply`. The supplied DeepSeek adapter uses HTTPS
Chat Completions, `tool_choice=required`, and `thinking.type=disabled` as described
in [DeepSeek's tool-call guide](https://api-docs.deepseek.com/guides/tool_calls/).
It keeps only the assistant content and function-call fields, and never returns
or logs `reasoning_content`. Actions/observations provide the execution trace.
Keys are excluded from configuration representations and provider response
bodies are not included in errors. Trace observations can contain conversation
and product data; store them only according to the application's logging policy.

## Verification and remaining limits

```bash
python3 -m unittest tests.test_react_agent -v
python3 -m unittest discover -v
python3 -m evaluator.local_evaluator --output results.json
```

The full suite needs the repository's runtime/experiment dependencies. The
ReAct tests alone use only the standard library and a three-product fixture.
They cover observation-dependent action selection, search repetition, asking
before search, multi-turn follow-ups, replacing needs, intent/category changes,
invalid JSON, unknown tools, hallucinated/stale IDs, terminal batches, token
usage, transport retries, missing credentials, rollback, and the step limit.

The public offline scores do not evaluate this LLM mode. Live extraction,
instruction robustness, response grounding, latency and cost require additional
evaluation with an actual account/key. The tool loop is bounded, but timeout is
per HTTP attempt rather than a global turn deadline. Prompt history is limited
to the last 12 messages (4,000 characters each), plus structured state; product
fields are truncated. State is in memory and not durable across process restarts.
Like the existing SQLite-based agent, use an instance serially; concurrent
service workers should own separate instances. No web search, purchase, payment,
or other external action tool is included.
