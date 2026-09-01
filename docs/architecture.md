# Agent architecture

## Request lifecycle

The evaluator creates one `Agent` instance and calls `reset` before each session.
`respond` then runs the following sequence for every turn:

1. `IntentRouter` classifies the message as buying, browsing, or an intent
   override.
2. `SlotExtractor` extracts the category and any positive or negative
   constraints.
3. `ConversationState` applies the new information to the session.
4. `ContextBuilder` combines the current message with the retained state.
5. `HybridRetriever` runs keyword, category, broad-constraint,
   conjunctive-constraint, and optional semantic searches.
6. Reciprocal Rank Fusion merges the route results. Leading items from precise
   routes are retained in the rerank pool.
7. The configured reranker scores the candidates. The default is the packaged
   11-feature logistic model; LightGBM LambdaRank and a manual score remain
   available as explicit alternatives.
8. The first-turn policy may defer a low-confidence recommendation list until
   one useful preference has been collected.
9. `ClarificationPolicy` selects at most one follow-up attribute, and the agent
   returns the response in the evaluator's required schema.

## Session state

| Input | State update |
| --- | --- |
| Buying signal | Store new constraints as requirements |
| Browsing signal | Store new constraints as preferences |
| Explicit intent override | Clear prior intent-specific slots, retain category and profile, then apply the new request |
| No-preference answer | Mark the requested attribute as unavailable so it is not asked again |
| Ordinary follow-up | Add compatible values without discarding earlier context |

The stable category and anonymized profile survive an intent override. Hard and
soft slots do not, because they belong to the abandoned request.

## Retrieval

- **Keyword:** searches the current message and distilled context.
- **Category:** keeps results inside the detected catalog category.
- **Broad constraint:** uses OR matching to preserve recall for short or generic
  constraints.
- **Conjunctive constraint:** requires the informative terms from one disclosed
  fragment to occur together and processes the most specific fragments first.
- **Semantic:** accepts an injected `SemanticRetriever`; the default
  `NullSemanticRetriever` keeps the submitted runtime offline and deterministic.

Buying and browsing use different route weights from `AgentConfig`. Buying gives
more weight to constraint precision, while browsing leaves more room for broader
discovery.

## Reranking and fallbacks

All learned rerankers consume features from `shopping_copilot/features.py`. The
features cover constraint matches, category overlap, route ranks and scores,
profile tags, product quality, turn number, and current intent.

The runtime fallback order is:

1. the model selected by `SHOPPING_COPILOT_RERANKER`;
2. the packaged logistic model if a LightGBM model cannot be loaded;
3. the manual structured score.

The semantic retriever and the learned reranker are injected behind small
interfaces, so either can be replaced without changing `starter/agent.py` or the
official evaluator.
