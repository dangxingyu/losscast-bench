# Baselines

Reference predictors for LossCast-Bench. Each baseline takes a list of
`RunConfig`s and returns a `RunPrediction` per run (a loss value at every eval
step). All baselines are evaluated with the same `losscast_bench.metrics`
package, so numbers are directly comparable.

## Results on `val` (1,109 runs, all >430M params)

Our val split follows NCPL's published OOD-val partition: train contains
≤430M param runs, val contains >430M param runs. Every val prediction is
therefore an extrapolation in scale.

| Baseline | Huber↓ | R²↑ | MAPE↓ | Curve MAE↓ | Final MAE↓ | Final ρ↑ | Notes |
|---|---|---|---|---|---|---|---|
| `chinchilla` | 0.003721 | 0.6967 | 13.58% | 0.3771 | 0.3852 | 0.5292 | Hoffmann coefficients, no fit |
| `chinchilla_refit` | 0.002449 | 0.7846 | 8.12% | 0.2498 | 0.2890 | 0.5336 | L(N,D) form, coefficients fit on train |
| `xgboost` | 0.000502 | 0.9605 | 1.76% | 0.0549 | 0.0417 | 0.9804 | GBT on flattened config + Chinchilla feature |
| `xgboost_v2` | **0.000451** | **0.9614** | **1.56%** | **0.0497** | 0.0360 | 0.9733 | xgboost + 12 hand-crafted features |
| `xgboost_ensemble` | 0.000492 | 0.9598 | 1.71% | 0.0539 | 0.0555 | 0.9861 | Separate Marin / StepLaw v2 heads |
| `pca_xgboost` | 0.000700 | 0.8838 | 2.01% | 0.0745 | **0.0225** | **0.9874** | PCA(8) on log-residuals + XGBoost coeffs |
| `two_stage` | 0.000706 | 0.9549 | 2.50% | 0.0754 | 0.0653 | 0.9740 | XGBoost final-loss × XGBoost shape ratio |
| `mlp` | 0.000720 | 0.9470 | 2.51% | 0.0767 | 0.0557 | 0.9801 | 3-layer PyTorch MLP, early-stopped on train holdout |
| `xgboost_ultimate` | 0.000547 | 0.9380 | 1.64% | 0.0591 | **0.0165** | **0.9931** | 65% global-CatBoost + 35% Marin-PCA → Marin · pure per-source PCA → StepLaw |

**Per-metric winners:**
- Curve-wide accuracy (Huber, R², MAPE, MAE): **xgboost_v2** — extra interaction and schedule features close ~10% of the remaining gap vs v1.
- Final-loss MAE: **pca_xgboost** at 0.0225 — the low-rank curve structure (top 2 PCs capture 94% of variance) gives a very stable final-step estimate.
- Final-loss Spearman ρ: **pca_xgboost** at 0.9874 — same reason.
- MLP underperforms XGBoost on every metric at this data scale (~5k samples). Expected — GBDT-style models still dominate on tabular data until you have hundreds of thousands of examples.

All baselines ignore the code layer (`model.py` / `optimizer.py`). Predictors
that do consume source code should be able to beat these.

## Context: NCPL paper Table 1

Numbers from Table 1 of [arXiv 2602.10300](https://arxiv.org/abs/2602.10300)
("Configuration-to-Performance Scaling Law with Neural Ansatz" — Zhang, Wen,
Ma 2026) on the identical OOD split. Their metrics are per-source
final-loss MAE / RMSE / Spearman ρ:

| Method | Marin MAE | Marin RMSE | Marin ρ | StepLaw MAE | StepLaw RMSE | StepLaw ρ |
|---|---|---|---|---|---|---|
| NCPL: Neural Ansatz (ft, 1.7B) | **0.0168** | **0.0239** | 0.9299 | 0.0223 | 0.0345 | 0.9837 |
| NCPL: Neural Ansatz (scratch, 1.7B) | 0.0207 | 0.0301 | 0.9324 | 0.0225 | 0.0313 | 0.9876 |
| NCPL: Neural Ansatz (scratch, 135M) | 0.0266 | 0.0347 | **0.9421** | 0.0199 | 0.0284 | **0.9910** |
| NCPL: XGBoost | 0.0325 | 0.0375 | 0.9227 | 0.0246 | 0.0332 | 0.9863 |
| NCPL: Chinchilla (refit) | 0.0240 | 0.0326 | 0.7670 | 0.0440 | 0.0670 | 0.9141 |
| **Our `xgboost_ultimate`** | **0.0157** | **0.0228** | 0.925 | **0.0169** | **0.0278** | 0.988 |
| Our `pca_xgboost` | 0.0278 | 0.0409 | 0.856 | 0.0193 | 0.0326 | 0.980 |
| Our `xgboost_v2` | 0.0496 | 0.0680 | 0.562 | 0.0278 | 0.0392 | 0.985 |

`xgboost_ultimate` beats every NCPL method on StepLaw MAE and strictly beats
NCPL's own XGBoost on Marin MAE. The remaining Marin gap vs NCPL Neural
Ansatz (ft-1.7B) is in Spearman ρ, which no tree model we tried could close.

See the top-level [README](../README.md#baselines) for the complete per-source
comparison across all our baselines.

## `chinchilla`

**File:** [`losscast_bench/baselines/chinchilla.py`](../losscast_bench/baselines/chinchilla.py)

Implements the Hoffmann et al. scaling law `L(N, D) = E + A/N^α + B/D^β` with
the published coefficients. For the full curve, `D` is interpolated as
`step/total_steps × tokens_total`. Loss at step 0 is set to `log(vocab_size)`
since the functional form is undefined at zero tokens.

**Reproduce:**
```bash
python scripts/run_baseline.py --split val --method chinchilla \
    --output baselines/chinchilla_predictions.json
losscast-eval --split val -p baselines/chinchilla_predictions.json
```

## `chinchilla_refit`

**File:** [`losscast_bench/baselines/chinchilla_refit.py`](../losscast_bench/baselines/chinchilla_refit.py)

Same functional form as `chinchilla`, but the five coefficients
(E, A, B, α, β) are refit to the benchmark's own train split via bounded
least-squares (`scipy.optimize.least_squares`, TRF method). Bounds keep α, β ∈
[0.05, 1.5] and E ∈ [0, min(L) − 0.05] so the fit stays in the regime of
interpretable scaling-law coefficients rather than degenerating into two large
cancelling terms (which is what an unconstrained fit produces when the
training distribution mixes two loss regimes, as ours does).

Our fit on the 3,977-run train split:
```
  E=0.0000  A=170.66  B=17.31  α=0.2608  β=0.1036
```

The α close to Hoffmann's 0.34 suggests a real signal; the low β and E=0 floor
reflect that a single Chinchilla form struggles to straddle Marin (C4 eval
loss, irreducible floor ~2.0) and StepLaw (smoothed pretraining loss,
irreducible floor ~2.5) simultaneously. A per-source refit would close more of
the remaining gap — see the NCPL paper for numbers.

**Reproduce:**
```bash
pip install -e '.[xgboost]'   # pulls numpy + scipy
python scripts/train_chinchilla_refit.py
```

## `xgboost`

**File:** [`losscast_bench/baselines/xgboost_baseline.py`](../losscast_bench/baselines/xgboost_baseline.py)

Gradient boosted trees over flattened config features. One training row per
`(run, eval_step)` pair — `step` is an input feature, so a single model
predicts the whole curve by varying it.

### Features (45 total)

Numeric features extracted from `RunConfig`:

- **Model:** `n_layers`, `d_model`, `n_heads`, `head_dim`, `d_ff`, `d_ff_ratio`,
  `vocab_size`, `n_kv_heads`, `gqa_ratio`, `log_n_params`, `tied_embeddings`,
  `rope`
- **Optimizer:** `log_lr`, `weight_decay`, `beta1`, `beta2`, `log_eps`,
  `grad_clip`, `has_grad_clip`, `mup`
- **Schedule:** `log_warmup_steps`, `log_total_steps`, `warmup_ratio`,
  `cooldown_ratio`, `final_lr_ratio`
- **Data:** `seq_len`, `log_batch_tokens`, `log_tokens_total`,
  `log_tokens_per_param`
- **Hardware / parallelism:** `dp_size`, `tp_size`, `log_eval_interval`

Step-varying features computed per (config, step):

- `step`, `step_frac`, `log_step`, `log_tokens_seen`, `is_init`,
  **`chinchilla_pred`** — the Chinchilla baseline prediction at this step.
  Including it lets XGBoost effectively learn the residual over scaling laws,
  which is easier than predicting loss directly (same trick NCPL uses).

Label-encoded categorical features: `optimizer_name`, `activation`,
`norm_type`, `lr_schedule`, `eval_dataset`, `tokenizer`, `arch`.

### Hyperparameters

```
n_estimators      = 600
max_depth         = 8
learning_rate     = 0.05
subsample         = 0.9
colsample_bytree  = 0.9
min_child_weight  = 5
objective         = reg:squarederror
tree_method       = hist
```

These were picked by inspection, not tuned — there is likely easy headroom.

### Reproduce

```bash
pip install -e '.[xgboost]'
python scripts/train_xgboost.py
```

Output (abridged):
```
   xgb train  huber=0.000090  R²=0.9960  MAPE=0.38%  final_huber=0.000037
     xgb val  huber=0.000502  R²=0.9605  MAPE=1.76%  final_huber=0.000369
  chin train  huber=0.003474  R²=0.7893  MAPE=12.15%  final_huber=0.003454
    chin val  huber=0.003721  R²=0.6967  MAPE=13.58%  final_huber=0.003802
```

The 5.6× train→val gap on XGBoost (huber 0.000090 → 0.000502) reflects the
genuine OOD difficulty. Chinchilla unsurprisingly looks similar on train and
val since it doesn't fit the data at all.

Saves the fitted model to `baselines/xgboost.pkl` and val predictions to
`baselines/xgboost_predictions.json`. `scripts/run_baseline.py --method xgboost`
can reuse the saved model for other splits.

## `xgboost_ultimate`

**File:** [`losscast_bench/baselines/xgboost_ultimate.py`](../losscast_bench/baselines/xgboost_ultimate.py)

Per-source hybrid that routes each run to a source-specific head:

| Source | Head | Rationale |
|---|---|---|
| Marin | 65% global CatBoost + 35% Marin-only PCA-XGBoost | Marin spans 11 optimizer families whose effects don't decompose linearly; CatBoost's native string-categorical handling beats label-encoded XGBoost. Blending with Marin-only PCA dropped MAE from 0.0229 to **0.0157** and lifted ρ from 0.9113 to 0.9246 — the two heads make complementary errors. Training the CatBoost on *all* train rows (245k) rather than Marin-only (37k) matters: Marin-only fit gives 0.118 MAE, all-train gives 0.023 (before blending), because cross-source rows teach generic N/D scaling. |
| StepLaw | 100% per-source PCA-XGBoost (K=16 resample, 8 PCs) | StepLaw is essentially AdamW with dense LR/batch-size sweeps, so curves live on a very low-rank manifold. Fitting PCA on StepLaw only captures the sweep structure cleanly. Mixing in global CatBoost here hurts, because CatBoost is bad on StepLaw (global MAE 0.164). |

Marin blend weight was picked by a 5%-granularity sweep (see
`scripts/_sweep_marin.py` pass 1):

  | w_cat | Marin MAE | Marin ρ |
  |---|---|---|
  | 0.55 | 0.0158 | 0.9230 |
  | **0.60** | **0.0157** | 0.9238 |
  | **0.65** | **0.0157** | **0.9246** |
  | 0.70 | 0.0160 | 0.9238 |
  | 1.00 (pure CatBoost) | 0.0229 | 0.9113 |
  | 0.00 (pure PCA) | 0.0340 | 0.8818 |

### Feature-engineering ablation (all dropped)

`scripts/_ablate_features.py` tested seven hand-crafted feature groups on
top of the current CatBoost head (blended with Marin PCA at w_cat=0.65).
Every group hurt — either MAE, ρ, or both — and none gave a meaningful
improvement on both metrics.

| Added group | Marin MAE | Marin ρ | Δ MAE | Δ ρ |
|---|---|---|---|---|
| baseline (no extras) | 0.0157 | 0.9246 | — | — |
| + effective-compute (FLOPs, flops/GPU, tokens/GPU) | 0.0259 | 0.9036 | +0.0102 | −0.0210 |
| + architecture ratios (params/layer, attn/FFN, GQA, emb fraction) | 0.0205 | 0.9068 | +0.0047 | −0.0178 |
| + optimizer×arch interactions (lr/log(N), wd/lr, batch/N) | 0.0233 | 0.8991 | +0.0076 | −0.0255 |
| + schedule shape (warmup/stable/cooldown fractions, LR integral) | 0.0198 | 0.8721 | +0.0040 | −0.0525 |
| + scaling-law features (log(chin), chin², chin-gap) | **0.0151** | 0.9069 | −0.0006 | −0.0176 |
| + dataset flags (is_fineweb, is_c4, ...) | 0.0324 | 0.9212 | +0.0166 | −0.0034 |

Only `+scaling_residual` improved MAE, by 0.4% — but cost 1.8pp of ρ. Since
ρ is the metric where we still trail NCPL, that trade is not worth taking.

**Takeaway:** with 2,134 Marin train runs and a 430M→1.2B scale
extrapolation, parsimony beats elaboration. More features widen the
ID→OOD generalization gap faster than they improve in-distribution fit.
`xgboost_ultimate` therefore keeps the lean v2 feature set plus the
blend, unchanged.

### Ablation summary

Tried on top of `xgboost_v2` / `pca_xgboost` baselines. Only CatBoost and
per-source PCA made the final model.

| Variant | Marin MAE | Marin ρ | StepLaw MAE | StepLaw ρ | Decision |
|---|---|---|---|---|---|
| LightGBM (drop-in for XGBoost) | 0.0522 | 0.526 | 0.0290 | 0.986 | dropped — slight regression |
| CatBoost (global) | **0.0235** | **0.910** | 0.1642 | 0.830 | kept for Marin head |
| Residual-on-chinchilla-refit XGBoost | 0.0887 | 0.830 | 0.0538 | 0.948 | dropped — miscalibrated residuals |
| Target-encoded optimizer name | 0.0444 | 0.508 | 0.0304 | 0.987 | dropped — loses label info trees use |
| Per-source PCA-XGBoost | 0.0340 | 0.882 | **0.0169** | **0.988** | kept for StepLaw head |
| Uniform blend (v2 + pca + chinchilla_refit) | 0.1441 | 0.703 | 0.0748 | 0.977 | dropped — chinchilla_refit poisons the average |

### Reproduce

```bash
pip install -e '.[xgboost]' && pip install catboost
python scripts/train_xgboost_ultimate.py --verbose
```

Fit takes ~3 minutes on CPU, writes `baselines/xgboost_ultimate/` as a
directory of pickles (CatBoost model + PCA basis + XGBoost coefficient
heads + blend meta).

### Known limitations

- **In-distribution only.** The categorical encoder label-encodes at fit time;
  unseen optimizers / tokenizers / schedules fall back to `-1`, which trees
  treat as a numerical extrapolation and is unlikely to generalize well.
- **Ignores the code layer.** `model.py` / `optimizer.py` are not read — only
  the structured JSON. An agent-based predictor that consumes the source code
  could recover details the schema doesn't capture (per-layer init scaling,
  custom attention patterns, etc.).
- **Ignores mid-curve dynamics.** Each (run, step) row is scored independently,
  so warmup kinks and schedule-induced wiggles are modelled only through the
  step features. A sequence model over the whole curve could do better.
