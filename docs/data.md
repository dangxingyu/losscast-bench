# LossCast-Bench Data Specification

LossCast-Bench is a **living benchmark** — its dataset grows over time through community contributions. Anyone who has run a training experiment with a well-specified recipe can contribute their data to help the community build better loss predictors.

This document is the single authoritative reference for data formats, code templates, and contribution guidelines.

## Overview

Each training run in the benchmark has two layers of representation:

| Layer | Format | Contains | Required? |
|---|---|---|---|
| **JSON config** | `{run_id}/config.json` | Structured hyperparameters (arch, optimizer, data, schedule) | Always |
| **Code** | `{run_id}/model.py` | Exact architecture definition + weight initialization | Recommended |

The JSON config is the primary input for loss predictors. The code layer captures details that JSON can't express (per-layer initialization, custom attention patterns, etc.) and enables agent-based approaches that can reason about code directly.

## Dataset Structure

Each run is a self-contained directory. No shared JSON files — this avoids merge conflicts when multiple contributors submit simultaneously.

```
data/
├── train/                              # Public training data
│   ├── alice_125m_c4_cosine/
│   │   ├── config.json                 # Training recipe
│   │   ├── losses.json                 # Ground truth loss curve
│   │   ├── model.py                    # Architecture + init (recommended)
│   │   └── optimizer.py                # Only for non-standard optimizers
│   └── bob_350m_pile_wsd/
│       ├── config.json
│       ├── losses.json
│       ├── model.py
│       └── optimizer.py
├── val/                                # Public validation data
│   └── ...                             # Same per-run structure
└── test/                               # Private leaderboard data
```

The `train/` and `val/` splits are public (config + losses + code). The `test/` split is private leaderboard data in the maintainer repository; public releases may omit it. Participants submit a predictor function, not predictions for specific configs.

The current public JSON data is stored as normal text files for transparent
diffs and review. Large raw training artifacts, fitted models, and hidden test
outputs should stay outside the public repository.

## JSON Format

### config.json (per-run)

Each run directory contains its own `config.json`:

```json
{
  "run_id": "sample_125m_c4_cosine",
  "model": {
    "arch": "transformer",
    "n_layers": 12,
    "d_model": 768,
    "n_heads": 12,
    "head_dim": null,
    "d_ff": null,
    "vocab_size": 32000,
    "activation": "swiglu",
    "norm_type": "rmsnorm",
    "rope": true,
    "n_kv_heads": null,
    "tied_embeddings": false
  },
  "optimizer": {
    "name": "adamw",
    "lr": 3e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "grad_clip": 1.0,
    "mup": false
  },
  "data": {
    "dataset": "c4",
    "tokenizer": "gpt2",
    "seq_len": 2048,
    "batch_tokens": 524288,
    "tokens_total": 10e9,
    "mix": null,
    "eval_dataset": "c4"
  },
  "schedule": {
    "lr_schedule": "cosine",
    "warmup_steps": 2000,
    "total_steps": 50000,
    "cooldown_steps": 0,
    "final_lr_ratio": 0.1
  },
  "eval_interval": 500,
  "hardware": "8xA100-80G",
  "precision": "bf16",
  "dp_size": 1,
  "tp_size": 1
}
```

A blank template is available at `templates/config.json`.

### losses.json (per-run)

Each run directory contains its own `losses.json`:

```json
{
  "500": 5.51,
  "1000": 4.08,
  "5000": 3.11,
  "10000": 2.85,
  "25000": 2.68,
  "50000": 2.59
}
```

Keys are string representations of step numbers. Steps are determined by `eval_interval` — every multiple of `eval_interval` up to `total_steps` must have an entry. Loss values should be cross-entropy loss in nats, measured on the validation split of `eval_dataset`.

### predictions.json (Submission Format)

Predictions are still submitted as a single file with all runs:

```json
{
  "runs": [
    {
      "run_id": "sample_125m_c4_cosine",
      "predictions": {
        "500": 5.43,
        "1000": 4.12,
        "5000": 3.15,
        "10000": 2.89,
        "25000": 2.71,
        "50000": 2.62
      }
    }
  ]
}
```

Validation rules:

- Every run in the split must have a corresponding prediction entry
- Every eval step (derived from `eval_interval`) must have a corresponding prediction
- Predictions for steps outside the configured eval grid are rejected
- Evaluation aborts on validation errors unless `--allow-invalid` is passed for local debugging
- All values must be non-negative floats
- Values above 20.0 trigger a warning (likely an error)

## JSON Field Reference

### model

| Field | Type | Description |
|---|---|---|
| `arch` | string | `transformer`, `mamba`, `rwkv`, `retnet` |
| `n_layers` | int | Number of layers |
| `d_model` | int | Hidden dimension |
| `n_heads` | int or null | Attention heads (null for non-attention archs) |
| `head_dim` | int or null | Per-head dimension (null = d_model / n_heads) |
| `d_ff` | int or null | FFN dimension (null defaults to 4 × d_model) |
| `vocab_size` | int | Vocabulary size (default 32000) |
| `activation` | string | `swiglu`, `gelu`, `relu` |
| `norm_type` | string | `rmsnorm`, `layernorm` |
| `rope` | bool | Whether RoPE positional encoding is used |
| `n_kv_heads` | int or null | KV heads for GQA (null = MHA) |
| `tied_embeddings` | bool | Whether input/output embeddings are tied |

### optimizer

| Field | Type | Description |
|---|---|---|
| `name` | string | `adamw`, `adam`, `sgd`, `lion`, `muon`, `soap` |
| `lr` | float | Peak learning rate |
| `weight_decay` | float | Weight decay coefficient |
| `beta1` | float | First moment decay |
| `beta2` | float | Second moment decay |
| `eps` | float | Epsilon for numerical stability |
| `grad_clip` | float or null | Gradient clipping norm (null = no clipping) |
| `mup` | bool | Whether muP / spectral scaling is used |

### data

| Field | Type | Description |
|---|---|---|
| `dataset` | string | Dataset name, e.g. `c4`, `pile`, `refinedweb`, `fineweb`, `dolma`, `custom_mix` |
| `tokenizer` | string | HuggingFace tokenizer name (e.g. `gpt2`, `meta-llama/Llama-2-7b-hf`) |
| `seq_len` | int | Sequence length |
| `batch_tokens` | int | Batch size in tokens (= batch_size × seq_len) |
| `tokens_total` | float | Total training tokens |
| `mix` | dict or null | Data mix weights, e.g. `{"web": 0.7, "code": 0.2, "books": 0.1}` |
| `eval_dataset` | string | Dataset whose validation split is used to measure loss (required) |

### schedule

| Field | Type | Description |
|---|---|---|
| `lr_schedule` | string | `cosine`, `linear`, `constant`, `wsd`, `inverse_sqrt` |
| `warmup_steps` | int | Warmup steps |
| `total_steps` | int | Total training steps |
| `cooldown_steps` | int | Cooldown steps (for WSD schedule) |
| `final_lr_ratio` | float | final_lr / peak_lr |

### Top-level

| Field | Type | Description |
|---|---|---|
| `run_id` | string | Unique identifier for the run |
| `eval_interval` | int | Loss is recorded every this many steps (default 500) |
| `hardware` | string or null | Hardware description, e.g. `8xA100-80G` |
| `precision` | string | `fp32`, `bf16`, `fp16` |
| `dp_size` | int | Data parallelism degree |
| `tp_size` | int | Tensor parallelism degree |

## Code Format

The code layer provides the exact model definition and weight initialization as runnable Python. This captures details the JSON can't express — per-layer init scaling, custom attention patterns, non-standard components, etc.

### model.py (required for code contributions)

A self-contained PyTorch file defining the model architecture and initialization. Must follow the standard template (`templates/model.py`):

```python
class Model(nn.Module):
    def __init__(self, config: dict):
        # Architecture definition
        ...
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Exact weight initialization logic (per-layer if needed)
        ...

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Forward pass: input_ids → logits
        ...
```

Requirements:

- Pure PyTorch — no framework dependencies (HuggingFace, etc.)
- Self-contained — all components defined in this single file
- Runnable — `python model.py` executes a built-in verification that checks the model builds, forward pass produces correct shapes, and no NaN/Inf in output
- `MODEL_CONFIG` dict at the bottom must match the JSON config's model fields

A complete example is in `templates/example/model.py` (125M transformer with SwiGLU, RMSNorm, RoPE, and GPT-2 style residual scaling init).

### optimizer.py (only for non-standard optimizers)

Only needed when using an optimizer not fully described by the JSON config fields (e.g., SOAP, Muon, Schedule-Free). Standard optimizers (adamw, adam, sgd, lion) don't need this file.

Must follow the standard template (`templates/optimizer.py`):

```python
class CustomOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, **kwargs):
        ...

    @torch.no_grad()
    def step(self, closure=None, step_idx: int = 0):
        # Full update rule, including periodic operations.
        # Use step_idx for any step-dependent logic
        # (e.g., SOAP's preconditioner update every N steps).
        ...
```

The key design choice: `step()` receives `step_idx` so all periodic logic lives inside the optimizer. This keeps the training loop standardized — contributors don't need to expose it.

Requirements:

- Pure PyTorch
- `step(closure, step_idx)` interface
- Runnable — `python optimizer.py` runs a built-in verification (200 steps, checks loss decreases, no NaN)
- `OPTIMIZER_DEFAULTS` dict must match the JSON config's optimizer fields

## Contributing Data

### What to contribute

Each contribution is a **(config, ground_truth, code)** tuple:

1. **JSON config** — fully specified, following the schema above
2. **Ground truth losses** — validation loss (cross-entropy, nats) at each eval step, measured on a fixed held-out set
3. **Code** (recommended) — `model.py` and optionally `optimizer.py`, following the standard templates

### Naming convention

Use the format `{contributor}_{params}_{dataset}_{schedule}` for `run_id`:

- `alice_350m_pile_cosine`
- `labx_1b_dolma_wsd`
- `chen_70m_refinedweb_linear`

### eval_interval

We recommend `eval_interval: 500` (the default). This means loss is recorded at steps 500, 1000, 1500, ..., up to `total_steps`. If you need a coarser or finer resolution, adjust accordingly — but keep in mind that more frequent evaluations give predictors more signal to learn from.

### Loss measurement

Report cross-entropy loss in nats on a fixed validation set. Ensure consistency across all checkpoints — same eval set, same number of eval batches, no gradient updates during evaluation.

### What makes a good contribution

Diversity is what makes the benchmark valuable:

- Diverse architectures (Mamba, RWKV, RetNet — not just transformers)
- Diverse scales (10M to 10B+ parameters)
- Diverse data recipes (different datasets, mixes, sequence lengths)
- Diverse optimizers and schedules (especially non-standard ones)
- Unusual configurations (these test generalization and are especially valuable)

### Quality control

Contributions are reviewed via pull request with manual review. PRs are tied to GitHub identities, which makes sabotage traceable.

Before submitting, validate locally:
```bash
python scripts/validate_contribution.py --run-id {your_run_id} --split train
```

This checks schema compliance, model.py syntax, loss curve sanity, and internal consistency. Reviewers will additionally assess plausibility of the loss curve for the given configuration.

### How to submit

1. Fork the repository
2. Create `data/train/{run_id}/` with `config.json`, `losses.json`, `model.py` (and `optimizer.py` if needed)
3. Verify:
   ```bash
   python data/train/{run_id}/model.py            # model verification
   python scripts/validate_contribution.py --run-id {run_id} --split train
   ```
4. Open a PR using the "Data Contribution" template. Fill in the training setup details, link to training logs or WandB if available.

## Loading Data Programmatically

```python
from losscast_bench.data import load_split, list_splits

# See which splits are available
print(list_splits())  # e.g. ["train", "val", "test"]

# Load a split
configs, ground_truths = load_split("val")

# Load everything needed for evaluation
from losscast_bench.data import load_split_predictions
configs, gts, preds = load_split_predictions("val", "my_predictions.json")

# Evaluate
from losscast_bench.metrics import evaluate
result = evaluate(preds, gts, configs)
print(result.summary())
```
