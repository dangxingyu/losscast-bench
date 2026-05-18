## Data Contribution

**Run ID:** `{run_id}`

### Training setup

- **Model:** {arch}, {n_params} params, {n_layers}L / {d_model}D / {n_heads}H
- **Optimizer:** {name}, peak LR {lr}
- **Data:** {dataset}, {tokens_total} tokens, seq_len {seq_len}
- **Schedule:** {lr_schedule}, {total_steps} steps
- **Hardware:** {hardware}, {precision}
- **Framework/codebase:** {e.g., nanochat, NanoGPT, custom PyTorch, HuggingFace transformers}

### Files

- [ ] `data/train/{run_id}/config.json` — training recipe
- [ ] `data/train/{run_id}/losses.json` — ground truth loss curve
- [ ] `data/train/{run_id}/model.py` — model builds and passes verification (`python model.py`)
- [ ] `data/train/{run_id}/optimizer.py` — (if non-standard optimizer) builds and passes verification

### Loss measurement

- **Eval dataset:** {which dataset's validation split, e.g. fineweb, c4}
- **Tokenizer:** {HuggingFace tokenizer name, e.g. gpt2}
- **Metric:** cross-entropy loss in nats
- **Eval frequency:** every {eval_interval} steps
- **Consistency:** same eval set and eval batch count at every checkpoint

### Validation

```bash
python data/train/{run_id}/model.py
python scripts/validate_contribution.py --run-id {run_id} --split train
```

- [ ] Model verification passes
- [ ] Schema validation passes
- [ ] Loss curve looks plausible (attach a plot if possible)

### Notes

{Any additional context: training logs, WandB link, known quirks, etc.}
