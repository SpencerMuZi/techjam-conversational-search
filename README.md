# SlotState Shopping Agent

A lightweight, offline conversational shopping agent for the TechJam Conversational E-Commerce Search Challenge. The agent finds a hidden Amazon product by combining multi-turn constraint tracking with hybrid lexical retrieval and deterministic reranking.

## Problem

The challenge evaluates whether an agent can place the customer's hidden target `parent_asin` in its Top 10 recommendations within ten turns. The goal is to maximize HitRate@10 and MRR while reaching the correct product in as few turns as possible.

## Solution overview

The agent implements a slot-level dialogue state machine and a two-stage retrieval pipeline.

1. It captures the product category and explicit user constraints in independent slots: material, color, size, brand, budget, style, use case, and feature.
2. It treats a statement such as "ignore my earlier preference" as an intent override: the most recently supplied preference slot is removed and the replacement is inserted, while independent confirmed constraints remain active.
3. It retrieves 300 candidates from the frozen catalog with SQLite FTS5/BM25, using higher field weights for title and category.
4. It reranks candidates with fixed, interpretable weights:

```text
score = 0.55 * lexical_retrieval
      + 0.30 * constraint_coverage
      + 0.10 * budget_fit
      + 0.05 * profile_affinity
```

The agent asks the valid broad follow-up field `other`, allowing the customer simulator to disclose the next useful product detail while it continues returning ranked recommendations every turn.

## Public development result

The supplied `public_evaluation_results.json` was produced by the official local evaluator over all 200 public sessions.

| Metric | Result |
| --- | ---: |
| HitRate@10 | 0.915 |
| MRR | 0.586621 |
| MTTC | 3.0 |
| Efficiency | 0.8 |
| Recommended TechnicalScore | 0.793486 |

For Intent Override sessions, HitRate@10 is 0.73. This is the primary improvement over a whole-history approach.

## Repository structure

```text
agent.py                       Required Agent implementation
public_evaluation_results.json Full public-set evaluation output
requirements.txt               Runtime dependency declaration
```

## Setup

Requirements:

- Python 3.10 or newer
- The official TechJam participant kit and frozen `catalog.jsonl`

This implementation uses only the Python standard library (`sqlite3`, `json`, `re`, and `math`). It has no model download, external API, API key, or network requirement at scoring time.

```bash
git clone https://github.com/sandeloYisewtes/sandelo-nus-techjam-shopping-AI.git
git clone https://github.com/TechJam2026/techjam-conversational-search.git participant-kit
cp sandelo-nus-techjam-shopping-AI/agent.py participant-kit/starter/agent.py
```

Download the organizer-provided catalog release, verify its checksum, and place it at `participant-kit/data/catalog.jsonl`. The catalog itself is deliberately not included in this repository.

## Reproduce the public evaluation

From the participant-kit directory, run:

```bash
python -m evaluator.local_evaluator --catalog data/catalog.jsonl --output results.json
```

The expected aggregate result is shown above. Exact values can vary only if the official catalog or evaluator changes.

## Tools, APIs, libraries, and data

- Development tools: Python and the official TechJam local evaluator.
- Third-party APIs: none.
- Libraries/frameworks: Python standard library only; SQLite FTS5 is bundled with Python's `sqlite3` runtime.
- Data: the organizer-provided frozen 50,000-product `Clothing_Shoes_and_Jewelry` catalog and 200 public development sessions, derived from Amazon Reviews 2023.
- Attribution: Amazon Reviews 2023, McAuley Lab, UCSD. See the official participant kit's `DATA_ATTRIBUTION.md` for the upstream dataset terms.

## Limitations and next steps

- The slot extractor is deterministic and English keyword-based; paraphrases not covered by its rules may be missed.
- Retrieval is lexical rather than dense semantic retrieval, so it is weaker for synonyms and highly abstract requests.
- Intent overrides are modeled as replacement of the latest explicit preference. A learned or LLM-assisted resolver could identify the precise reference in more complex dialogue.
- Future work includes semantic embeddings, value-of-information question selection, and a validation protocol based on held-out sessions.

## Generalization safeguards

The agent never reads `ground_truth`, `scenario_type`, `difficulty_bucket`, or `sample_id`. It has no fitted parameters from the public target labels: the ranking coefficients are fixed, interpretable design choices. At runtime it uses only the allowed customer message, anonymized profile, and frozen catalog.

## Team contributions

Update this table before the final Devpost submission.

| Team member | Contribution |
| --- | --- |
| TBD | Algorithm design, implementation, evaluation, and documentation |
