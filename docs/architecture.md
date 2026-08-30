# Agent Architecture

## Runtime flow

The evaluator creates an Agent once and calls `reset` for every isolated session. `respond` applies the following deterministic workflow:

1. Route the new message as Buying, Browsing, or Intent Override.
2. Extract category and structured constraints.
3. Apply the turn to the session state.
4. Distill the state into a route-specific `SearchContext`.
5. Retrieve independent keyword, category, broad-constraint, conjunctive-constraint,
   and semantic candidate lists. The conjunctive route requires every term in
   one disclosed fragment and processes more specific fragments first.
6. Fuse routes using Reciprocal Rank Fusion, then seed the head of each precise
   route (`AgentConfig.precise_seed`) directly into the rerank pool so a strong
   BM25 hit with a thin fused score is still reranked.
7. Rerank the candidate set with the packaged 36-feature LambdaRank model. If
   LightGBM is unavailable, load the 11-feature pure-Python logistic scorer; a
   manual structured score is the final fallback. Features cover hard/soft
   constraint satisfaction, category overlap, route ranks and scores, profile
   tags, product quality, turn, and buying/browsing intent.
8. Return Top-10 recommendations and, when useful, one structured clarification request.

## State transition rules

| Input condition | State action |
| --- | --- |
| Buying signal | Treat newly extracted constraints as hard constraints |
| Browsing signal | Treat newly extracted constraints as soft preferences |
| `Actually...ignore...` | Clear hard/soft intent slots, preserve category/profile, apply the new constraint |
| No preference response | Mark the last requested attribute unavailable and do not ask it again |
| Normal answer | Accumulate new values without deleting earlier compatible values |

## Retrieval routes

- Keyword route searches the current turn plus distilled context.
- Category route preserves catalog boundary precision.
- Constraint route emphasizes disclosed material, feature, color, style, size, use-case, and budget values.
- Conjunctive constraint search repairs a recall failure of broad OR search for
  long catalog feature fragments, while keeping generic one-word constraints on
  the broad route.
- Semantic route is dependency-injected and defaults to `NullSemanticRetriever` for a reproducible offline run.

Route weights are selected at runtime from `AgentConfig`: Buying emphasizes constraint precision; Browsing allocates more weight to semantic discovery.

## Safe extension points

- Replace `NullSemanticRetriever` with an in-memory embedding index. Roughly half
  of the remaining misses are targets whose disclosed constraints are pure
  boilerplate (`polyester`, `100% Polyester`, `Imported`) that no keyword route
  can separate; a dense route is the main lever left.
- Add a local cross-encoder after RRF and before the final structured score.
- Replace the clarification priority list with expected information gain computed over candidate metadata.
- Add per-source slot confidence and time decay for soft preferences.

None of these changes require modifying `starter/agent.py` or the official evaluator.
