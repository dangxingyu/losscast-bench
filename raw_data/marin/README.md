# Marin Raw Data

Raw WandB exports from the Marin (Fantastic Optimizers) project: `stanford-mercury/optimizer-scaling`.

Downloaded via `scripts/download_marin_runs.py`.

## Per-Run Files

Each subdirectory is one WandB run:

| File | Contents |
|------|----------|
| `wandb_meta.json` | Run ID, URL, state, creation date, **parsed fields** (model_size, tokens, optimizer, lr, wd, etc.) |
| `wandb_config.json` | Raw WandB config (often empty for these runs — config is encoded in the run name) |
| `wandb_summary.json` | Final metrics snapshot: all eval losses, train loss, timing info |
| `wandb_history.json` | Full step-by-step training history (large file, may be missing for timed-out runs) |
| `eval_losses.json` | Extracted eval + train losses at each logged step (derived from history) |

## Key Fields in eval_losses.json

Each entry has `_step` plus loss values. Most steps have NaN for eval (eval only runs periodically):

- `eval/paloma/c4_en/loss` — **This is what we use** as the target loss for losscast-bench (`eval_dataset: "c4_en"`)
- `train/loss` — Training loss at that step
- `eval/paloma/*/loss` — Eval loss on other Paloma domains (wikitext, redpajama, falcon, etc.)

## Run Name Encoding

Run names encode the config: `sweep-{size}-{tokens}-{optimizer}{hash}lr{lr}-...`

Examples:
- `sweep-130m-2B-mudam093d8elr0.008-wd0.2-minlr0-warmup1000-sb10.95-7dbc49`
- `sweep-300m-6B-lbs-muon-2xlr` (less structured manual runs)

The `wandb_meta.json > parsed` section has the extracted fields.

## Converting to losscast-bench Format

To convert a run to the benchmark format, you need to:

1. Extract non-NaN `eval/paloma/c4_en/loss` values from `eval_losses.json` → `losses.json`
2. Build `config.json` from `wandb_meta.json > parsed` + `wandb_summary.json` fields
3. Many config fields (n_layers, d_model, n_heads, etc.) must be inferred from model_size using Marin's known architecture configs
4. See `CONTRIBUTE.md` for the target format and `data/val/sample_*/config.json` for examples

## Known Issues

- 3 of 10 runs have no `wandb_history.json` / `eval_losses.json` (WandB API timeout on large runs)
- `wandb_config.json` is often empty — config is parsed from run name instead
- Marin uses GPT-NeoX tokenizer but some run names don't specify this
