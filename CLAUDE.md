# LossCast-Bench

A benchmark for predicting LLM pretraining loss curves from training recipes.

## Project Structure

```
data/{split}/{run_id}/       # Per-run directories
  config.json                # Training configuration (model, optimizer, data, schedule)
  losses.json                # Loss curve: {"step": loss_value, ...}
  model.py                   # Self-contained PyTorch model (optional for data contributions)
  optimizer.py               # Custom optimizer if non-standard (optional)

raw_data/                    # Unprocessed source data (not in benchmark format yet)
  marin/                     # Marin (Fantastic Optimizers) runs from WandB
  steplaw/                   # StepLaw runs (not yet downloaded)

scripts/                     # CLI tools: eval, baseline, download, validate
losscast_bench/              # Python package: data loading, metrics, baselines
```

## Key APIs

- `from losscast_bench.data import load_split, list_runs, list_splits`
- `load_split("val")` returns `(configs, ground_truths)` by scanning `data/val/*/`
- CLI: `losscast-eval --split val -p predictions.json`

## Data Schema Decisions

### eval_dataset field (required in config.json)
Identifies what loss is being measured. Losses are only comparable across runs with the same `tokenizer` + `eval_dataset`.

- Marin data: `eval_dataset: "c4_en"` — uses `eval/paloma/c4_en/loss` from WandB (true eval loss on C4 English validation set)
- StepLaw data: `eval_dataset: "train"` — uses pretraining loss (no separate eval set)
- General: use the name of the validation dataset, e.g. `"fineweb"`, `"c4"`, `"pile"`

### losses.json format
- Keys are step numbers as strings, values are loss (cross-entropy, nats)
- Steps should be at every `eval_interval` through `total_steps`
- Example: `{"0": 12.21, "1000": 4.29, "2000": 3.99, ...}`

### Run ID convention
Format: `{contributor}_{params}_{dataset}_{schedule}`
For Marin conversions: `marin_{size}_{tokens}_{optimizer}` (e.g., `marin_130m_2b_mudam`)

## Data Sources

### Marin (Fantastic Optimizers paper)
- WandB: `stanford-mercury/optimizer-scaling`
- 130M–1.2B params, multiple optimizers (AdamW, Muon, Mudam, Shampoo, etc.)
- Eval: Paloma benchmark with multiple domains; we use `eval/paloma/c4_en/loss`
- Training data: C4/Dolma mix, GPT-NeoX tokenizer

### StepLaw
- WandB: `billzid/predictable-scale`
- 215M–1B params, AdamW only, fine-grained LR/batch-size sweeps
- Loss: pretraining loss (no separate eval set)

### NCPL Paper (Zhang, Wen, Ma 2025)
- Published dataset: `huggingface.co/datasets/zhqwqwq/NCPL-Pretraining-Logs`
- 3,225 train + 796 ID val + 1,109 OOD val runs from both Marin and StepLaw
- OOD split = models >430M params

## Contributing

See CONTRIBUTE.md for the full pipeline. Two Claude skills help with data contribution:

- **schema-creator** (`.claude/skills/schema-creator/`): Creates config.json + losses.json from training artifacts (WandB runs, logs, CLI args, source code). For nanochat runs, uses `scripts/nanochat_to_bench.py` under the hood and copies raw source code as-is.
- **code-rewriter** (`.claude/skills/code-rewriter/`): Converts framework-coupled code (HuggingFace, custom repos) into standalone PyTorch model.py / optimizer.py. NOT needed for nanochat — its code is already clean PyTorch.
