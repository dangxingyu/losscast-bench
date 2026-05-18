---
name: schema-creator
description: |
  Create losscast-bench config.json + losses.json from training run artifacts.
  Accepts: WandB run URL, training logs, source code, CLI args, or checkpoint metadata.
  Identifies the source codebase (nanochat, Marin, StepLaw, HuggingFace, custom) and
  extracts the right fields. Trigger on: "create config from this run", "extract schema",
  "make a config.json", "add this run to the benchmark", "convert this training run".
---

# Schema Creator

Create a losscast-bench run directory (`config.json` + `losses.json` + optional source code)
from whatever training artifacts the user provides.

## Inputs

The user gives you one or more of:

| Input | What to extract |
|-------|-----------------|
| WandB run URL or ID | Config from `run.config`, losses from `run.history()` |
| Training script + CLI args | Parse argparse defaults + overrides |
| Source code (model, optimizer) | Architecture details, optimizer setup |
| Training logs (JSON/CSV/stdout) | Loss values at each step |
| Checkpoint metadata JSON | Config snapshot, step count |
| A text description | Best-effort config extraction |

## Step 1: Identify the Source Codebase

### nanochat (karpathy/nanochat)

**Signals:** imports from `nanochat.*`, `GPTConfig`, `MuonAdamW`, args like `--depth`, `--aspect-ratio`

**Action:** Run the converter:
```bash
python scripts/nanochat_to_bench.py \
  --wandb-run {RUN_PATH} \
  --nanochat-dir {NANOCHAT_DIR} \
  -o {OUTPUT_DIR}
```

Or from CLI args:
```bash
python scripts/nanochat_to_bench.py \
  --args "depth=12 aspect_ratio=64 device_batch_size=32 num_iterations=5000" \
  --nanochat-dir {NANOCHAT_DIR} \
  --losses {LOSSES_PATH} \
  -o {OUTPUT_DIR}
```

For model code, ask the user which option they prefer:
- **Raw code** (recommended for nanochat): copy `gpt.py` → `model.py`, `optim.py` → `optimizer.py` as-is. Preserves every implementation detail.
- **Rewritten code**: use the code-rewriter skill to produce standardized `class Model` + `MODEL_CONFIG` format. Passes stricter validation.

If using raw code, pass `--nanochat-dir` to copy automatically.

If the user provides a git commit/tag for the exact code version, use `--git-ref`.

### Marin (stanford-mercury/optimizer-scaling)

**Signals:** WandB project `stanford-mercury/optimizer-scaling`, run names like `sweep-130m-2B-mudam...`

**Action:** Architecture is determined by model size using this lookup:

| model_size (MB) | n_layers | d_model | n_heads | d_ff | n_params |
|-----------------|----------|---------|---------|------|----------|
| ~135 | 32 | 512 | 8 | 1365 | 130M |
| ~302 | 32 | 768 | 12 | 2048 | 300M |
| ~537 | 32 | 1024 | 16 | 2730 | 520M |
| ~1209 | 32 | 1536 | 24 | 4096 | 1.2B |

All Marin runs use: `activation: "swiglu"`, `norm_type: "rmsnorm"`, `rope: true`,
`vocab_size: 32000`, `eval_dataset: "c4_en"`, `tokenizer: "EleutherAI/gpt-neox-20b"`.
Extract eval losses from `eval/paloma/c4_en/loss` in WandB history.

### Other / Custom

**Action:** Read the code. Build config.json field by field. If the code uses framework
dependencies (HuggingFace, flash-attn, etc.), use the **code-rewriter** skill to produce
standalone model.py. Ask the user for anything you can't determine.

## Step 2: Build config.json

Reference schema: `templates/config.json`

**Required** (error if missing):
- `run_id`, `model.{n_layers, d_model, n_heads, vocab_size}`
- `optimizer.{name, lr}`, `data.{dataset, tokenizer, eval_dataset, seq_len}`
- `schedule.total_steps`

**Derived** (compute from other fields):
- `d_ff` — usually `4 * d_model`; for SwiGLU: `round(8/3 * d_model)`
- `head_dim` — `d_model / n_heads`
- `batch_tokens` — `device_batch_size × seq_len × n_gpus`
- `tokens_total` — `batch_tokens × total_steps`

**Ask the user if** you can't determine `eval_dataset` or `tokenizer`.

## Step 3: Extract losses.json

Format: `{"step": loss, ...}` with string keys.

**nanochat:** eval metric is BPB (bits-per-byte) logged as `val/bpb`. The converter
handles this via `--wandb-run` or `--losses`.

**Sanity checks:** loss should decrease, initial ≈ `ln(vocab_size)`, final ≈ 2.5–4.0, no NaN.

## Step 4: Validate

```bash
python scripts/validate_contribution.py --run-dir {OUTPUT_DIR}
```

## Step 5: Summarize

Tell the user what was created, what was guessed, what's missing, and next steps.
