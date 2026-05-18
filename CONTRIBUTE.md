# Contributing Data to LossCast-Bench

You are an agent (or human) that has just finished running an LLM pretraining experiment. Follow these steps to contribute your training recipe and loss curve to the benchmark.

## Prerequisites

```bash
git clone https://github.com/xingyudang/losscast-bench.git
cd losscast-bench
pip install -e .
```

## Step 1: Choose a run ID

Format: `{contributor}_{params}_{dataset}_{schedule}`

Examples: `alice_350m_c4_cosine`, `labx_1b_dolma_wsd`, `nanochat_d26_fw_muon`

Pick a unique ID that doesn't already exist:
```bash
python -c "
from losscast_bench.data import list_runs, list_splits
existing = set()
for split in list_splits():
    existing.update(list_runs(split))
print('Existing run IDs:', sorted(existing))
"
```

## Step 2: Create config.json

Create your run directory and config file. Every field below must be filled in from your training code.

```bash
mkdir -p data/train/{YOUR_RUN_ID}
```

Write `data/train/{YOUR_RUN_ID}/config.json`:
```json
{
  "run_id": "{YOUR_RUN_ID}",

  "model": {
    "arch": "transformer",
    "n_layers": null,
    "d_model": null,
    "n_heads": null,
    "head_dim": null,
    "d_ff": null,
    "vocab_size": null,
    "activation": "swiglu",
    "norm_type": "rmsnorm",
    "rope": true,
    "n_kv_heads": null,
    "tied_embeddings": false
  },

  "optimizer": {
    "name": "adamw",
    "lr": null,
    "weight_decay": null,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "grad_clip": 1.0,
    "mup": false
  },

  "data": {
    "dataset": null,
    "tokenizer": null,
    "seq_len": null,
    "batch_tokens": null,
    "tokens_total": null,
    "mix": null,
    "eval_dataset": null
  },

  "schedule": {
    "lr_schedule": "cosine",
    "warmup_steps": null,
    "total_steps": null,
    "cooldown_steps": 0,
    "final_lr_ratio": 0.1
  },

  "eval_interval": 500,
  "hardware": null,
  "precision": "bf16",
  "dp_size": 1,
  "tp_size": 1
}
```

Fill in every `null` from your training config. Key things to get right:

- `batch_tokens` = per_device_batch_size × seq_len × gradient_accumulation_steps × num_gpus
- `lr` = **peak** learning rate
- `tokens_total` = batch_tokens × total_steps
- `d_ff`: set explicitly if not 4 × d_model
- `n_kv_heads`: set if using GQA/MQA, otherwise null (= MHA)
- `head_dim`: set if not d_model / n_heads, otherwise null
- `tokenizer`: **required** — HuggingFace tokenizer name (e.g., `"gpt2"`, `"meta-llama/Llama-2-7b-hf"`). Losses are only comparable across runs that use the same tokenizer + eval_dataset combination.
- `eval_dataset`: **required** — which dataset's validation split you measure loss on (e.g., `"fineweb"`, `"c4"`). Must be a standard dataset with a well-known validation split.
- For non-standard optimizers (Muon, SOAP, etc.), use a descriptive name like `"muon_adamw"`

## Step 3: Write model.py

There are two options for providing model code:

**Option A: Rewritten model (recommended).** A standardized, self-contained model.py that follows the template below. This passes automated validation and is easiest for predictors to consume.

**Option B: Raw source code.** If your training code is already clean PyTorch (e.g., nanochat's `gpt.py`), you can copy it directly as model.py. This preserves every implementation detail exactly but will skip some automated checks. Use this when rewriting would lose important architectural nuances.

### Option A: Rewritten model (default)

Create `data/train/{YOUR_RUN_ID}/model.py`. This file must be:

- **Self-contained**: all components (RMSNorm, RoPE, Attention, FFN, etc.) defined in this single file
- **Pure PyTorch**: no HuggingFace, no flash-attn, no xformers, no triton
- **Runnable**: `python model.py` executes a built-in verification

Required structure:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Components: norms, position encodings, attention, FFN, etc.
# ...

class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # Build architecture from config dict
        ...
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Exact weight initialization matching your training code
        # Pay attention to: per-layer scaling, residual projection scaling, embedding init
        ...

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len) -> logits: (batch, seq_len, vocab_size)
        ...
        return logits

MODEL_CONFIG = {
    # Must match the "model" section of your config.json exactly
    "arch": "transformer",
    "n_layers": ...,
    "d_model": ...,
    # ... all fields
}

# ── Verification (copy this block exactly) ──────────────────────────────
if __name__ == "__main__":
    print("Building model...")
    model = Model(MODEL_CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    batch_size, seq_len = 2, 128
    x = torch.randint(0, MODEL_CONFIG.get("vocab_size", 32000), (batch_size, seq_len))
    print(f"Forward pass with input shape {tuple(x.shape)}...")
    with torch.no_grad():
        logits = model(x)
    print(f"Output shape: {tuple(logits.shape)}")
    assert logits.shape == (batch_size, seq_len, MODEL_CONFIG.get("vocab_size", 32000)), \
        f"Expected (batch, seq, vocab), got {tuple(logits.shape)}"
    assert not torch.isnan(logits).any(), "NaN in output"
    assert not torch.isinf(logits).any(), "Inf in output"
    print("model.py verification PASSED")
```

If your training code uses HuggingFace `transformers`, nanochat, NanoGPT, or another framework, see `.claude/skills/code-rewriter/SKILL.md` for conversion guidance.

### Option B: Raw source code

If your model code is already clean PyTorch (no heavy framework abstractions), you can copy it directly:

```bash
cp /path/to/your/model_code.py data/train/{YOUR_RUN_ID}/model.py
cp /path/to/your/optimizer_code.py data/train/{YOUR_RUN_ID}/optimizer.py  # if applicable
```

Raw code will skip the `class Model` / `MODEL_CONFIG` validation checks, but the schema validation for config.json and losses.json still applies. This is the recommended path for nanochat contributions — use `scripts/nanochat_to_bench.py` to generate config.json and copy `gpt.py` / `optim.py` as-is.

## Step 4: Write optimizer.py (only if non-standard)

Skip this step if you're using standard AdamW/Adam/SGD/Lion.

If your optimizer has custom update rules (Muon, SOAP, hybrid Muon+AdamW, Schedule-Free, etc.), create `data/train/{YOUR_RUN_ID}/optimizer.py`:

```python
import torch
from torch.optim import Optimizer
import math

class CustomOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, **kwargs):
        defaults = dict(lr=lr, **kwargs)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, step_idx: int = 0):
        # Full update rule. Use step_idx for any periodic operations
        # (preconditioner updates, warmup inside optimizer, etc.)
        ...

OPTIMIZER_DEFAULTS = {
    # Must match the "optimizer" section of your config.json
    "name": "...",
    "lr": ...,
    "weight_decay": ...,
    # ... all fields
}

# ── Verification (copy this block exactly) ──────────────────────────────
if __name__ == "__main__":
    from model import Model, MODEL_CONFIG
    print("Building model + optimizer...")
    model = Model(MODEL_CONFIG)
    optimizer = CustomOptimizer(model.parameters(), **{
        k: v for k, v in OPTIMIZER_DEFAULTS.items() if k != "name"
    })

    x = torch.randint(0, MODEL_CONFIG.get("vocab_size", 32000), (2, 128))
    losses = []
    for step in range(200):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            x.view(-1),
        )
        loss.backward()
        optimizer.step(step_idx=step)
        optimizer.zero_grad()
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"  step {step}: loss={loss.item():.4f}")

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert not any(math.isnan(l) for l in losses), "NaN in losses"
    print("optimizer.py verification PASSED")
```

## Step 5: Add your loss curve

Your ground truth is the validation loss (cross-entropy, nats) at every eval checkpoint.

Format: `{ "step_number": loss_value, ... }` for every multiple of `eval_interval` up to `total_steps`.

```python
# Helper: convert a CSV/list of (step, loss) to the right format
import json

# Replace with your actual data
raw_losses = [
    (500, 5.51), (1000, 4.08), (1500, 3.65), (2000, 3.41),
    # ... every eval_interval steps through total_steps
]

losses_dict = {str(step): round(loss, 4) for step, loss in raw_losses}

# Save as losses.json in your run directory
with open(f"data/train/{RUN_ID}/losses.json", "w") as f:
    json.dump(losses_dict, f, indent=2)
```

## Step 6: Validate

```bash
# Model builds and passes shape checks
python data/train/{YOUR_RUN_ID}/model.py

# Optimizer (if present) builds and loss decreases
python data/train/{YOUR_RUN_ID}/optimizer.py

# Full validation: schema, loss curve sanity, consistency
python scripts/validate_contribution.py --run-id {YOUR_RUN_ID} --split train
```

All three must pass before submitting.

## Step 7: Submit PR

```bash
git checkout -b contribute/{YOUR_RUN_ID}
git add data/train/{YOUR_RUN_ID}/
git commit -m "Add training run {YOUR_RUN_ID}"
git push -u origin contribute/{YOUR_RUN_ID}
```

Open a PR using the "Data Contribution" template. Include:
- Hardware and framework used
- Link to training logs / WandB if available
- Any notes about the run (e.g., diverged and restarted, non-standard eval setup)
