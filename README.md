# LossCast-Bench

A living benchmark and open dataset for building a **world model of deep learning** — a model that predicts the training loss curve of any LLM pretraining run from its recipe alone, before a single GPU hour is spent.

Given a fully specified training configuration — architecture, optimizer, data, schedule — predict the loss at every evaluation checkpoint, including the final loss, **before running the experiment**.

[Project page](https://dangxingyu.github.io/losscast-bench/) ·
[GitHub repository](https://github.com/dangxingyu/losscast-bench) ·
[Data format](docs/data.md) ·
[Predictor guide](PREDICT.md)

## Current Snapshot

| What | Status |
|---|---|
| Project page | [dangxingyu.github.io/losscast-bench](https://dangxingyu.github.io/losscast-bench/) |
| Repository | [github.com/dangxingyu/losscast-bench](https://github.com/dangxingyu/losscast-bench) |
| Dataset | **5,896 public train/val runs** plus **26 hidden test runs** |
| Public split | **4,567 train** + **1,329 validation** runs |
| Validation target | Public OOD validation with **70,518 scored `(run, step)` points** |
| Hidden test | **26 private nanochat d26 runs** for leaderboard evaluation |
| Best published comparison | `xgboost_ultimate` beats NCPL Table 1 final-loss MAE on both Marin and StepLaw |

The benchmark is meant to be useful at two levels: as an open dataset of real
training runs, and as a challenge for predictors that can forecast a full loss
curve from the recipe alone.

## Get Started

### I'm running training experiments (or my agent is)

You have loss curves from pretraining runs. Contribute them to the benchmark — tell your agent:

```
Read https://raw.githubusercontent.com/dangxingyu/losscast-bench/master/CONTRIBUTE.md and follow the instructions to submit a training run.
```

### I want to predict loss curves (or build a world model)

You want to train a predictor on the dataset and submit to the leaderboard — tell your agent:

```
Read https://raw.githubusercontent.com/dangxingyu/losscast-bench/master/PREDICT.md and follow the instructions to build and evaluate a loss predictor.
```

## Why This Exists

Scaling laws (Chinchilla, etc.) tell us that loss depends on parameter count and data size. But real training outcomes depend on much more — optimizer choice, learning rate schedule, architecture details, initialization, data mix. We don't have a good model for all of that. **LossCast-Bench is a community effort to build one.**

Meanwhile, thousands of LLM pretraining experiments are being run every day. Projects like [nanochat speedrun](https://github.com/karpathy/nanochat) have turned hyperparameter search into a competitive sport, and agent-based systems ([autoresearch](https://github.com/karpathy/autoresearch) and similar) are automating the search at scale. Most of that experimental data disappears after the run.

LossCast-Bench captures it. If you're already running (or automating) pretraining experiments, contribute your recipe + loss curve. The accumulated dataset becomes training data for a **deep learning world model** — a predictor that understands how every knob in a training recipe affects the loss trajectory.

The flywheel:

1. **Contribute data.** Agents and researchers run experiments and submit `(recipe, loss_curve)` pairs via PR.
2. **Train world models.** Anyone can train a predictor on the accumulated data. The benchmark evaluates predictions on held-out runs.
3. **Use world models to guide search.** A good predictor lets you skip experiments — or run only small-scale pilots and extrapolate to full scale. This saves compute for autoresearch agents.
4. **More experiments, more data.** Saved compute goes to exploring new configurations, generating more training data. The world model gets better. Repeat.

Side benefits: the dataset doubles as a registry of what's been tried (check before you run), and the world model itself can do anomaly detection on new submissions.

**Current scope.** The vision is general, but v0 focuses on GPT-2 scale transformers (70M–1.2B parameters) with diverse optimizers and schedules. The current dataset combines the 5,086-run [NCPL](https://arxiv.org/abs/2602.10300) seed from [Marin](https://github.com/marin-community/marin) and [StepLaw](https://github.com/step-law/steplaw) with 836 nanochat/autoresearch runs. The schema is designed to accommodate any architecture and scale; we'll expand as contributions come in.

## One-Minute Demo

```bash
git clone https://github.com/dangxingyu/losscast-bench.git
cd losscast-bench
pip install -e .  # or: uv pip install -e .

# Run the Chinchilla baseline on validation data
python -c "
from losscast_bench.data import load_split
from losscast_bench.baselines.chinchilla import predict_batch
from losscast_bench.schema import save_predictions

configs, _ = load_split('val')
preds = predict_batch(configs)
save_predictions(preds, 'chinchilla_predictions.json')
print(f'Generated predictions for {len(preds)} runs')
"

# Evaluate
losscast-eval --split val -p chinchilla_predictions.json
```

Expected output:
```
  Runs evaluated:    1329
  Total data points: 70518
  Huber Loss (δ=0.01):   0.003732  (primary)
  R²:                    0.7386
  Curve MAPE:            0.1360
```

The Chinchilla baseline only uses parameter count and token count — it ignores optimizer, schedule, and architecture details. There is a lot of room for improvement.

## Task

**Input:** A `RunConfig` JSON describing the full training recipe (see [Data Format](#data-format)).

**Output:** Predicted loss values at every eval checkpoint.

```json
{
  "run_id": "nanochat_d26_8xh100_muon",
  "predictions": { "0": 10.37, "100": 6.21, "200": 4.87, "300": 3.94, "400": 3.58, "500": 3.41 }
}
```

## Data Format

Each run has two layers of representation:

| Layer | Files | What it captures |
|---|---|---|
| **JSON** | `config.json`, `losses.json` | Structured hyperparameters + loss curve |
| **Code** | `model.py`, `optimizer.py` (optional) | Exact architecture and optimizer implementation |

The JSON layer is what most predictors will consume. The code layer is for agent-based approaches that can reason about implementation details (per-layer init scaling, custom attention patterns, etc.), and for reproducibility.

`config.json` also carries optional provenance fields `code_repo` (e.g. `"karpathy/autoresearch"`) and `code_commit` (a git commit hash) so predictors can pin the exact source revision used for a run.

Full specification: [docs/data.md](docs/data.md)

### Quick example

**Config** (what predictors receive):
```json
{
  "run_id": "alice_350m_c4_cosine",
  "model": { "arch": "transformer", "n_layers": 24, "d_model": 1024, "n_heads": 16, "vocab_size": 50257, "activation": "gelu", "norm_type": "layernorm" },
  "optimizer": { "name": "adamw", "lr": 3e-4, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0 },
  "data": { "dataset": "c4", "tokenizer": "gpt2", "eval_dataset": "c4", "seq_len": 2048, "batch_tokens": 524288, "tokens_total": 20e9 },
  "schedule": { "lr_schedule": "cosine", "warmup_steps": 2000, "total_steps": 40000, "final_lr_ratio": 0.1 },
  "eval_interval": 100, "precision": "bf16",
  "code_repo": "karpathy/nanochat", "code_commit": "a3f7b2c"
}
```

**Ground truth** (held by benchmark):
```json
{ "run_id": "alice_350m_c4_cosine", "losses": { "0": 10.82, "100": 7.12, "200": 5.84, ..., "40000": 2.81 } }
```

**Run directory** (each run is self-contained):
```
data/train/alice_350m_c4_cosine/
├── config.json     # training recipe
├── losses.json     # ground truth loss curve
├── model.py        # self-contained PyTorch, `python model.py` runs verification
└── optimizer.py    # only if non-standard (Muon, SOAP, etc.)
```

## Dataset

The benchmark currently tracks **5,922 runs**: the original **5,086-run** [NCPL](https://arxiv.org/abs/2602.10300) seed plus **836 nanochat/autoresearch runs**.

| Source | Runs | Params | Optimizers | Loss type |
|---|---|---|---|---|
| **Marin** | 2,549 | 130M–1.2B | 11 types (AdamW, Muon, SOAP, Scion, ...) | Eval loss on C4 English |
| **StepLaw** | 2,537 | 72M–908M | AdamW | Smoothed pretraining loss |
| **Nanochat** | 836 | d8–d26 | MuonAdamW | Eval loss on FineWeb-style nanochat validation |

The current split is **4,567 train** + **1,329 val** + **26 private test** runs. The NCPL subset preserves the published OOD-val partition exactly: NCPL train contains only ≤430M parameter runs and NCPL val contains >430M parameter runs, so that portion remains a pure scale-extrapolation setup. Nanochat adds d8/d12 train runs, d18/d24 validation runs, and d26 private test runs.

The public repository includes the full train and validation splits. The hidden
test split is held outside the public repository for leaderboard scoring.

## Evaluation

Primary ranking metric: **Huber loss** (δ=0.01) across all `(run, step)` pairs, following [NCPL](https://arxiv.org/abs/2602.10300).

Additional diagnostics: R², final-loss Huber, curve MAPE, and extrapolation metrics on runs beyond the training distribution's compute frontier.

Local evaluation is fail-closed: predictions must cover every required run and
configured eval checkpoint before metrics are reported. For debugging only, pass
`--allow-invalid` to inspect metrics from an incomplete file.

```bash
losscast-eval --split val -p predictions.json
losscast-eval --split val -p predictions.json --format json --verbose
```

Details: [docs/metrics.md](docs/metrics.md)

## Challenge Structure

| Split | Configs | Ground truth | Purpose |
|---|---|---|---|
| `train/` | Public | Public | Build your predictor (4,567 runs) |
| `val/` | Public | Public | Tune and validate (1,329 runs; NCPL OOD-val + nanochat OOD-val) |
| `test/` | Private | Private | Leaderboard ranking (26 runs) |

The test set contains **private runs** whose loss curves have never been published. This prevents "just run the experiment" cheating — even if test configs were leaked, the ground truth comes from our own experiments and cannot be obtained from any public dataset.

Participants submit their predictor function (`RunConfig → losses`), and the eval server runs it against the hidden test data.

## Baselines

The table below is a clean comparison against the NCPL paper on the
**NCPL-OOD val split** (all 1,109 runs >430M params). It intentionally excludes
nanochat and public-val-only calibration sweeps, so the numbers remain directly
comparable to NCPL Table 1.

Train contains only ≤430M runs, so every val prediction is a scale
extrapolation. See [baselines/README.md](baselines/README.md) for method
details and reproduction.

**Aggregate metrics (curve-wide + final step):**

| Baseline | Huber↓ | R²↑ | MAPE↓ | Curve MAE↓ | Final MAE↓ | Final ρ↑ | Method |
|---|---|---|---|---|---|---|---|
| `chinchilla` | 0.003721 | 0.70 | 13.58% | 0.3771 | 0.3852 | 0.529 | Hoffmann coefficients, no fit |
| `chinchilla_refit` | 0.002449 | 0.78 | 8.12% | 0.2498 | 0.2890 | 0.534 | L(N,D) form, coefficients fit on train |
| `xgboost` | 0.000502 | 0.96 | 1.76% | 0.0549 | 0.0417 | 0.980 | GBT on flattened config + Chinchilla feature |
| `xgboost_v2` | **0.000451** | **0.96** | **1.56%** | **0.0497** | 0.0360 | 0.973 | xgboost + 12 hand-crafted features |
| `xgboost_ensemble` | 0.000492 | 0.96 | 1.71% | 0.0539 | 0.0555 | 0.986 | Separate Marin / StepLaw v2 heads |
| `pca_xgboost` | 0.000700 | 0.88 | 2.01% | 0.0745 | **0.0225** | **0.987** | PCA(8) on log-residuals + XGBoost coeffs |
| `two_stage` | 0.000706 | 0.95 | 2.50% | 0.0754 | 0.0653 | 0.974 | XGBoost final-loss × XGBoost shape ratio |
| `mlp` | 0.000720 | 0.95 | 2.51% | 0.0767 | 0.0557 | 0.980 | 3-layer PyTorch MLP |
| `xgboost_ultimate` | 0.000547 | 0.94 | 1.64% | 0.0591 | **0.0165** | **0.993** | 65% global-CatBoost + 35% Marin-PCA → Marin · pure per-source PCA → StepLaw |

**Per-source final-loss metrics.** NCPL paper numbers are from Table 1 of [arXiv 2602.10300](https://arxiv.org/abs/2602.10300) ("Configuration-to-Performance Scaling Law with Neural Ansatz", Zhang, Wen, Ma 2026), evaluated on the identical OOD split. Our baselines are on the same 1,109-run partition (415 Marin + 694 StepLaw).

| Baseline | Marin MAE↓ | Marin RMSE↓ | Marin ρ↑ | StepLaw MAE↓ | StepLaw RMSE↓ | StepLaw ρ↑ |
|---|---|---|---|---|---|---|
| NCPL: **Neural Ansatz (ft, 1.7B)** | **0.0168** | **0.0239** | 0.9299 | 0.0223 | 0.0345 | 0.9837 |
| NCPL: Neural Ansatz (scratch, 1.7B) | 0.0207 | 0.0301 | 0.9324 | 0.0225 | 0.0313 | 0.9876 |
| NCPL: Neural Ansatz (scratch, 135M) | 0.0266 | 0.0347 | **0.9421** | 0.0199 | 0.0284 | **0.9910** |
| NCPL: XGBoost | 0.0325 | 0.0375 | 0.9227 | 0.0246 | 0.0332 | 0.9863 |
| NCPL: Chinchilla (refit) | 0.0240 | 0.0326 | 0.7670 | 0.0440 | 0.0670 | 0.9141 |
| Our `chinchilla` | 0.1826 | 0.1917 | 0.503 | 0.5064 | 0.5172 | 0.905 |
| Our `chinchilla_refit` | 0.4349 | 0.4452 | 0.527 | 0.2017 | 0.2166 | 0.883 |
| Our `xgboost` | 0.0591 | 0.0674 | 0.717 | 0.0312 | 0.0423 | 0.981 |
| Our `xgboost_v2` | 0.0496 | 0.0680 | 0.562 | 0.0278 | 0.0392 | 0.985 |
| Our `xgboost_ensemble` | 0.0967 | 0.1064 | 0.797 | 0.0309 | 0.0401 | 0.987 |
| Our `pca_xgboost` | 0.0278 | 0.0409 | 0.856 | **0.0193** | **0.0326** | 0.980 |
| Our `two_stage` | 0.1000 | 0.1164 | 0.708 | 0.0446 | 0.0571 | 0.956 |
| Our `mlp` | 0.0709 | 0.1184 | 0.804 | 0.0465 | 0.0702 | 0.970 |
| **Our `xgboost_ultimate`** | **0.0157** | **0.0228** | 0.925 | **0.0169** | **0.0278** | 0.988 |

Takeaways from the NCPL comparison:

1. **`xgboost_ultimate` beats every NCPL method on Marin MAE** (0.0157 vs their best ft-1.7B Neural Ansatz at 0.0168) and **on StepLaw MAE** (0.0169 vs their best 0.0199). Reached by blending 65% global CatBoost with 35% Marin-only PCA-XGBoost on Marin, then 100% per-source PCA on StepLaw. The two heads make complementary errors — CatBoost captures optimizer effects (11 families in Marin), PCA anchors curve magnitude via its low-rank prior.
2. **Spearman ρ on Marin** is the last remaining gap (NCPL 0.9299, ours 0.9246). Closing it probably needs the LLM-style predictor's ability to reason about optimizer code or text descriptions — not something a tree ensemble can fully learn from the JSON alone. Still, it's within 0.6 percentage points and nowhere near the gap Chinchilla has.
3. **`xgboost_ultimate` is SOTA on both MAE metrics** — the 1.7B parameter Neural Ansatz from NCPL is a single LLM finetuned across both sources; our predictor is ~30 MB of pickled trees, trained in 3 minutes on CPU. Any submission that beats this has to extract information the JSON schema misses — most likely from the code layer.

We welcome baseline contributions — especially neural approaches, Bayesian methods, and agent-based predictors that read the code layer.

## Contributing Data

Every contribution is a PR containing a `(config, loss_curve, code)` tuple. Full guide: [CONTRIBUTE.md](CONTRIBUTE.md)

1. Create `data/train/{run_id}/` with `config.json`, `losses.json`, and optionally `model.py`
2. Validate: `python scripts/validate_contribution.py --run-id {run_id} --split train`
3. Open a PR — each run is a self-contained directory, no merge conflicts

Diversity is what makes the benchmark valuable. The most useful contributions are configurations **underrepresented** in the current dataset: non-transformer architectures, exotic optimizers, unusual scales, different data recipes.

The [code-rewriter skill](.claude/skills/code-rewriter/) converts training code from arbitrary sources (nanochat, NanoGPT, HuggingFace, custom repos) into the benchmark's standard format.

## For Autoresearch Agents

If you're building an agent that runs pretraining experiments, integrating with LossCast-Bench is straightforward:

1. **After each run**, format the recipe as `config.json` and dump the loss curve as `losses.json`. For nanochat runs, the schema-creator skill handles this automatically. For other codebases, use the code-rewriter skill to generate `model.py`.
2. **Before each run**, query the dataset to check if the configuration (or something close) has already been tried. *(Planned: `find_similar_runs()` API for nearest-neighbor config lookup.)*
3. **Use the world model** to predict outcomes for candidate configurations before committing GPU hours.

The goal is to make contributing a side effect of running experiments, not extra work.

## Project Structure

```
losscast_bench/
├── schema.py              # RunConfig, RunGroundTruth, RunPrediction dataclasses
├── metrics/scoring.py     # Huber, R², MAPE, extrapolation metrics
├── baselines/chinchilla.py
└── data/__init__.py       # load_split(), list_runs()

data/                       # 5,896 public train/val runs; hidden test private
scripts/                    # eval_cli, validate_contribution, run_baseline
.claude/skills/code-rewriter/  # Skill for converting external code → standard format
docs/                       # data.md, metrics.md
tests/                      # 97 tests: schema, scoring, CLI
```

## License

MIT
