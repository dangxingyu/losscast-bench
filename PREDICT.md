# Building a Loss Predictor for LossCast-Bench

You are an agent (or human) that wants to predict LLM pretraining loss curves. Follow these steps to get started, build a predictor, evaluate it, and submit to the leaderboard.

## Prerequisites

```bash
git clone https://github.com/xingyudang/losscast-bench.git
cd losscast-bench
pip install -e .
```

## Understanding the Task

You receive a `RunConfig` — a JSON object describing a full training recipe (model architecture, optimizer, data, schedule). You must predict the training loss at every eval checkpoint.

```python
from losscast_bench.data import load_split

configs, ground_truths = load_split("train")
example = configs[0]
print(example.run_id)           # e.g. "alice_350m_c4_cosine"
print(example.model.arch)       # e.g. "transformer"
print(example.model.n_layers)   # e.g. 24
print(example.eval_steps)       # [500, 1000, 1500, ..., 50000]
```

Your predictor must output a loss value for **every step** in `config.eval_steps`.
Submissions that miss required runs or eval checkpoints fail validation and are
not scored by default.

## Available Data

### Structured data (JSON)

```python
from losscast_bench.data import load_split

# Training set: build your predictor on this
train_configs, train_gt = load_split("train")

# Validation set: tune hyperparameters, check generalization
val_configs, val_gt = load_split("val")
```

Key fields in each `RunConfig`:
- `model`: arch, n_layers, d_model, n_heads, head_dim, d_ff, vocab_size, activation, norm_type, rope, n_kv_heads, tied_embeddings
- `optimizer`: name, lr, weight_decay, beta1, beta2, eps, grad_clip, mup
- `data`: dataset, tokenizer, seq_len, batch_tokens, tokens_total, mix, eval_dataset
- `schedule`: lr_schedule, warmup_steps, total_steps, cooldown_steps, final_lr_ratio
- Top-level: eval_interval, hardware, precision, dp_size, tp_size
- Derived: `model.n_params_approx` (approximate non-embedding parameter count)

### Code layer (optional, for agent-based approaches)

Each run directory may include `model.py` and `optimizer.py` source files:
```bash
ls data/train/
# alice_350m_c4_cosine/  bob_1b_pile_wsd/  ...

cat data/train/alice_350m_c4_cosine/model.py
# Self-contained PyTorch model with exact architecture + init
```

These files contain implementation details the JSON can't capture: per-layer init scaling, custom attention patterns, non-standard activations. If your predictor can reason about code, this is additional signal.

## Building a Predictor

Your predictor is a function: `RunConfig → dict[int, float]` (step → predicted loss).

### Baseline: Chinchilla scaling law

```python
from losscast_bench.baselines.chinchilla import predict_batch
from losscast_bench.data import load_split
from losscast_bench.schema import save_predictions

val_configs, _ = load_split("val")
preds = predict_batch(val_configs)
save_predictions(preds, "chinchilla_preds.json")
```

This baseline uses `L(N,D) = E + A/N^α + B/D^β` and ignores optimizer, schedule, and architecture details. You should beat it.

### Writing your own predictor

```python
from losscast_bench.schema import RunConfig, RunPrediction

def predict_run(config: RunConfig) -> RunPrediction:
    predictions = {}
    for step in config.eval_steps:
        # Your prediction logic here
        # Inputs available: config.model.*, config.optimizer.*, config.data.*, config.schedule.*
        predicted_loss = your_model(config, step)
        predictions[step] = predicted_loss
    return RunPrediction(run_id=config.run_id, predictions=predictions)

def predict_batch(configs: list[RunConfig]) -> list[RunPrediction]:
    return [predict_run(c) for c in configs]
```

### Approaches to consider

- **Parametric scaling laws**: extend Chinchilla with optimizer/schedule terms
- **Neural predictors**: train a network that takes config features → loss curve
- **Gaussian processes / Bayesian**: model uncertainty over loss trajectories
- **Few-shot / in-context learning**: use an LLM to predict loss given examples
- **Code-reading agents**: parse model.py to extract features the JSON misses
- **Test-time training**: given a partially observed curve (small-scale run), fit and extrapolate

## Evaluating Locally

```bash
# Evaluate on validation set (loads configs + ground truth from per-run directories)
losscast-eval --split val -p my_predictions.json

# Or specify files explicitly
losscast-eval -p my_predictions.json -g ground_truth.json -c configs.json

# With extrapolation scoring
losscast-eval --split val -p my_predictions.json --extrap-threshold 50e9

# Per-run breakdown
losscast-eval --split val -p my_predictions.json --format table --verbose

# JSON output for programmatic use
losscast-eval --split val -p my_predictions.json --format json -o results.json

# Debug only: inspect metrics for an incomplete file
losscast-eval --split val -p partial_predictions.json --allow-invalid
```

### Metrics

| Metric | What it measures |
|---|---|
| **Huber (δ=0.01)** | Primary ranking metric. Robust to outliers. |
| **R²** | Explained variance across all (run, step) pairs. |
| **Final Huber** | Accuracy on final loss only. |
| **Curve MAPE** | Mean absolute percentage error across full curve. |
| **Extrap Huber** | Accuracy on runs beyond training compute frontier. |

See `docs/metrics.md` for full details.

## Submitting to Leaderboard

The test set is fully hidden (configs and ground truth not published). You submit your **predictor function**, not predictions for specific configs.

Submission format (details TBD as eval server is set up):
```python
# Your submission must expose this interface:
def predict(config: dict) -> dict:
    """
    Input: a RunConfig as a dict (same schema as config.json entries)
    Output: {"step": loss, ...} for every eval step
    """
    ...
```

The eval server will:
1. Load hidden test configs
2. Call your `predict()` function on each config
3. Score predictions against hidden ground truth
4. Report metrics on the leaderboard

## Tips

- The training set is small but growing. Overfitting is a real risk — use the val set.
- `model.n_params_approx` is a useful derived feature, but don't rely on it alone.
- Loss curves have structure: they decrease roughly as a power law, with warmup phase and schedule-dependent cooldown.
- The test set includes distributional shift: architectures, optimizers, and scales not seen in training. Generalization matters more than memorization.
- If you're using an LLM as your predictor, the code layer (model.py) contains rich signal that isn't in the JSON.
