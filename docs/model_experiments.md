# Reranking model comparison

5-fold cross-validation, folds split by `session_id` (never by row, so no turn from one session leaks across train/test). Each fold trains on the training sessions' captured candidate pools — every positive + the 24 hardest negatives by fused score + 8 random negatives per turn — and is then scored by the **real `evaluator.local_evaluator`** on the held-out sessions with the trained model plugged into `HybridRetriever.reranker`.

All four rerankers consume the identical 36-feature vector (`shopping_copilot/features.py`). `manual` is the shipped hand-tuned score expressed over those features; it reproduces the built-in result (Hit@10 0.9450 / MRR 0.5172 / tech 0.7745) exactly.

## Cross-validated ranking quality

| Method | Hit@10 (μ±σ) | MRR (μ±σ) | MTTC (μ±σ) | ROC-AUC | PR-AUC | train s | infer ms/turn |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| Manual weights | 0.9450 ± 0.0292 | 0.5172 ± 0.0680 | 3.6550 ± 0.4603 | 0.7427 ± 0.0130 | 0.0232 ± 0.0064 | 0.00 | 89.66 |
| Logistic regression | 0.9900 ± 0.0122 | 0.7271 ± 0.0286 | 2.4500 ± 0.4022 | 0.9894 ± 0.0036 | 0.5015 ± 0.0216 | 0.06 | 83.47 |
| Random forest | 0.9800 ± 0.0100 | 0.6947 ± 0.0380 | 2.5950 ± 0.3187 | 0.9870 ± 0.0064 | 0.3688 ± 0.0784 | 2.56 | 107.15 |
| LightGBM LambdaRank | 0.9800 ± 0.0187 | 0.6458 ± 0.0411 | 2.6150 ± 0.3434 | 0.9809 ± 0.0110 | 0.3852 ± 0.0450 | 0.94 | 90.43 |

### Per-fold MRR

| Method | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
| --- | --- | --- | --- | --- | --- |
| Manual weights | 0.5601 | 0.4895 | 0.6207 | 0.4945 | 0.4209 |
| Logistic regression | 0.7619 | 0.6853 | 0.7561 | 0.7209 | 0.7111 |
| Random forest | 0.7301 | 0.6321 | 0.6897 | 0.6831 | 0.7383 |
| LightGBM LambdaRank | 0.7019 | 0.5883 | 0.6445 | 0.6155 | 0.6788 |

### Per-scenario Hit@10 / MRR (CV mean)

| Method | boundary | browsing | buying | intent_override |
| --- | --- | --- | --- | --- |
| Manual weights | 0.900 / 0.347 | 0.978 / 0.544 | 0.970 / 0.481 | 0.860 / 0.609 |
| Logistic regression | 0.800 / 0.604 | 1.000 / 0.711 | 0.991 / 0.759 | 1.000 / 0.726 |
| Random forest | 0.800 / 0.511 | 0.977 / 0.632 | 0.991 / 0.731 | 1.000 / 0.812 |
| LightGBM LambdaRank | 1.000 / 0.532 | 0.980 / 0.614 | 0.979 / 0.697 | 1.000 / 0.641 |

## Full-fit reference (trained on all 200, scored on all 200)

Optimistically biased for the learned models — shown only to size the train/CV gap (overfitting signal).

| Method | Hit@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| Manual weights | 0.9450 | 0.5172 | 3.6550 | 0.7745 |
| Logistic regression | 0.9900 | 0.7375 | 2.4150 | 0.8879 |
| Random forest | 0.9900 | 0.7962 | 2.4000 | 0.9059 |
| LightGBM LambdaRank | 0.9950 | 0.8955 | 2.3500 | 0.9392 |

## Feature importance (top 8, full-fit)

- **Logistic regression**: `category_overlap_frac` (5.509), `soft_total` (5.262), `soft_hits` (4.334), `exact_phrase_hits` (2.512), `log_rating_number` (1.976), `hard_miss_count` (1.673), `full_coverage` (1.522), `hard_total` (1.441)
- **Random forest**: `log_rating_number` (0.364), `has_price` (0.176), `category_overlap_frac` (0.104), `category_overlap_count` (0.060), `kw_recip` (0.034), `avg_rating` (0.029), `kw_score` (0.029), `rrf_fused_score` (0.021)
- **LightGBM LambdaRank**: `log_rating_number` (0.182), `rrf_fused_score` (0.110), `cat_recip` (0.090), `kw_score` (0.078), `kw_recip` (0.062), `avg_rating` (0.060), `cat_score` (0.055), `constraint_score` (0.042)

## Overfitting signal (full-fit MRR − CV MRR)

| Method | CV MRR | Full-fit MRR | gap |
| --- | ---: | ---: | ---: |
| Manual weights | 0.5172 | 0.5172 | +0.0000 |
| Logistic regression | 0.7271 | 0.7375 | +0.0104 |
| Random forest | 0.6947 | 0.7962 | +0.1016 |
| LightGBM LambdaRank | 0.6458 | 0.8955 | +0.2497 |


## Feature selection

The 36-feature vector was over-specified (494 positives → ~14 per feature). L1
path + greedy forward selection (5-fold session-grouped CV, turn-level MRR proxy)
both converge on ~8–12 features; the extra 24 are net noise.

Real-evaluator 5-fold CV on candidate subsets:

| Subset | n | Hit@10 (μ±σ) | MRR (μ±σ) | MTTC (μ±σ) |
| --- | ---: | --- | --- | --- |
| full_36 | 36 | 0.9900 ± 0.0122 | 0.7271 ± 0.0286 | 2.450 ± 0.402 |
| greedy_8 | 8 | 0.9800 ± 0.0100 | 0.7436 ± 0.0263 | 2.520 ± 0.333 |
| greedy_12 | 12 | 0.9900 ± 0.0122 | 0.7589 ± 0.0257 | 2.425 ± 0.407 |
| **g11** (greedy_12 − `soft_idf_sum`) | **11** | **0.9900 ± 0.0122** | **0.7623 ± 0.0198** | **2.425 ± 0.407** |
| g9 (also − `soft_total`, `hard_total`) | 9 | 0.9800 ± 0.0100 | 0.7362 ± 0.0211 | 2.515 ± 0.354 |
| curated_10 (hand-picked) | 10 | 0.9750 ± 0.0158 | 0.6875 ± 0.0368 | 2.520 ± 0.315 |

### Coefficient stability (`experiments/weight_stability.py`)

25 refits (5 seeds × 5 scenario-stratified folds) of the greedy_12 model. The
eight features with |coef| > 1 all have |CV| = σ/|μ| ≤ 0.13 and 100 % sign
consistency. Two are weak:

| feature | mean coef | \|CV\| | sign consistency |
| --- | ---: | ---: | ---: |
| `soft_idf_sum` | −0.24 | 1.17 | 68 % — **sign flips across folds** |
| `hard_miss_count` | −0.63 | 0.54 | 100 % (direction stable, magnitude noisy) |

`soft_idf_sum` is dropped. `g9` shows that `soft_total` / `hard_total` — though
constant within a turn — still earn their place: removing them shifts the other
fitted coefficients and costs 0.026 MRR + 0.010 Hit@10.

### Shipped model

`shopping_copilot/reranker_lr.json` uses **g11** (11 features):

```
log_rating_number, category_overlap_frac, constraint_recip, hard_miss_count,
has_price, profile_hits, exact_phrase_hits, soft_total, hard_total, soft_hits,
cat_present
```

Official evaluator (full fit, 200 sessions):
**Hit@10 0.9900, MRR 0.7589, MTTC 2.410, Efficiency 0.8590, TechnicalScore 0.8945.**
Full-fit MRR 0.7589 vs CV MRR 0.7623 → train/CV gap ≈ 0, and CV variance is the
lowest of every subset tried (σ 0.020).

Reproduce: `python -m experiments.feature_select`,
`python -m experiments.validate_subset`, `python -m experiments.weight_stability`.

## Hard-negative mining (rejected)

Tested whether adding catalog-mined confusables to the training negatives helps
(`experiments/hard_negatives.py`): per target, BM25 on the target's own text, take
the 10 most similar items **not** already in the retrieved pool, label them 0.

| Arm | Hit@10 (μ±σ) | MRR (μ±σ) |
| --- | --- | --- |
| baseline (24 hard-by-fused + 8 random) | 0.9900 ± 0.0122 | 0.7623 ± 0.0198 |
| + 10 mined confusables / target | 0.9900 ± 0.0122 | 0.7620 ± 0.0293 |

No gain, and CV variance rises (σ 0.020 → 0.029). The pool's top-by-fused items
are already the hard confusables; mined items sit outside any retrieval route
(`constraint_recip` = 0, `rrf_fused_score` ≈ 0), a state the reranker never meets
at inference, so they add noise. The shipped model keeps the baseline negatives.

## High-capacity public-score model

The final public-score experiment adds conjunctive per-fragment constraint
retrieval, asks a broad `other` clarification before narrower attributes, and
trains LambdaRank against every candidate in each turn containing the target.
The selected model uses 1,000 estimators, up to 511 leaves, and 36 features.

| Metric | Full-fit public result |
| --- | ---: |
| Hit@10 | 0.9950 |
| MRR | 0.9950 |
| MTTC | 2.230 |
| Efficiency | 0.8770 |
| TechnicalScore | **0.971400** |

This is intentionally an optimistic, full-fit development result. A fresh
5-fold scenario-stratified GroupKFold run through the real evaluator gives:

| Metric | CV mean ± σ |
| --- | ---: |
| Hit@10 | 0.9950 ± 0.0100 |
| MRR | 0.7114 ± 0.0708 |
| MTTC | 2.385 ± 0.273 |
| TechnicalScore | 0.8832 ± 0.0236 |

The gap is large: the high-capacity model should be treated as a public-score
upper-bound experiment, not proof of private-set generalization. The packaged
logistic model remains the stability-oriented option and can be selected with
`SHOPPING_COPILOT_RERANKER=logistic`; `manual` disables learned reranking.

At runtime, the precise model's confidence-gated 300-pool Top-1 probe improves
its public result to TechnicalScore 0.973400 (MRR 0.995, MTTC 2.130) without
changing the packaged model weights. The CV table above measures the underlying
model without that public-tuned gate.

Reproduce the capacity search and grouped validation with:

```bash
python3 -m scripts.tune_lambdarank --extended
python3 -m scripts.cv_lambdarank_capacity
```

## Wide-pool efficiency model

The XGBoost V4 branch demonstrated that reranking all 300 retrieved candidates
can surface the purchased item earlier. Its published CV MTTC of 1.365 was not
directly comparable, because that experiment counted intent-override targets
before the override became active; the official evaluator result was MTTC 1.685,
Efficiency 0.9315, and TechnicalScore 0.908582.

The integrated approach keeps this useful 300-candidate idea but trains on the
framework's richer 36-feature vectors and uses the official evaluator semantics.
It also retains the precise, logistic, and manual modes instead of replacing the
full architecture with the isolated V4 submission files.

| Metric | Full-fit public | 5-fold CV mean ± σ |
| --- | ---: | ---: |
| Hit@10 | 1.0000 | 0.9950 ± 0.0100 |
| MRR | 0.9975 | 0.6181 ± 0.0280 |
| MTTC | 1.690 | 1.975 ± 0.151 |
| Efficiency | 0.9310 | — |
| TechnicalScore | **0.985450** | 0.8634 ± 0.0093 |

The wide model is an opt-in public-score mode. The CV gap is explicit; the
stability-oriented `logistic` model is the runtime default. Reproduce it with:

```bash
python3 -m scripts.train_wide_lambdarank
python3 -m scripts.cv_lambdarank_capacity \
  --pool-depth 300 --profile wide_pool --seed 2026
```

## Robustness hardening

The public data contains 200 sessions but only 125 unique user-profile payloads.
Session-only folds can therefore put the same profile in training and validation.
The hardened protocol uses scenario-stratified, **user-profile-grouped** outer
folds. LambdaRank capacity and early-stopping iteration are selected inside each
outer training fold; the outer fold is evaluated once by the real evaluator.

| Model | Hit@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| g11 Logistic | 0.9972 ± 0.0056 | 0.7706 ± 0.0580 | 2.239 ± 0.175 | **0.9050 ± 0.0202** |
| Regularized LambdaRank | 0.9235 ± 0.0662 | 0.6051 ± 0.0926 | 2.982 ± 0.886 | 0.8036 ± 0.0766 |

The regularized search compared 7/15-leaf trees, 11/12-feature subsets,
L1/L2 penalties, hard-negative subsampling, NDCG@1 early stopping, and a
rank-10 LambdaRank truncation level. Inner folds selected only 8–27 trees in
four of five outer folds, confirming that the former 800–1,000-tree models were
far beyond the data's supported capacity. Preventing memorization does not make
the small dataset sufficient for a nonlinear model; the linear model wins.

First-turn deferral was tested under the same profile-grouped folds:

| Policy | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: |
| Immediate | 0.7706 ± 0.0580 | 2.239 ± 0.175 | 0.9050 ± 0.0202 |
| Defer every first turn | 0.7892 ± 0.0433 | 2.804 ± 0.129 | 0.8993 ± 0.0133 |
| Existing adaptive gate | 0.7856 ± 0.0500 | 2.446 ± 0.181 | 0.9054 ± 0.0159 |
| Browsing-only | 0.7857 ± 0.0500 | 2.459 ± 0.169 | 0.9051 ± 0.0161 |

The adaptive gate slightly improves the mean while reducing fold variance and is
therefore paired with the stable logistic reranker as the runtime default. None
of the leakage-resistant variants reaches 0.95. At the observed Hit@10 and MTTC,
that target requires roughly 0.91 MRR;
additional independently labelled or fold-safe synthetic sessions are needed.

Reproduce:

```bash
python3 -m scripts.cv_regularized_lambdarank --output regularized_cv.json
python3 -m scripts.cv_deferral_policy --output deferral_cv.json
```
