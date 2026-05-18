# LossCast-Bench v0 Release Plan

## What's Done

- [x] Schema design (config.json, losses.json, model.py, optimizer.py)
- [x] Python package (losscast_bench: schema, data loading, metrics, Chinchilla baseline)
- [x] CLI tools (losscast-eval, validate_contribution, run_baseline, nanochat_to_bench)
- [x] NCPL seed data: 5,086 runs, split by NCPL OOD rule (3,977 train ≤430M + 1,109 val >430M)
- [x] Skills (schema-creator, code-rewriter)
- [x] Test suite (77 tests passing)
- [x] Docs (README, PREDICT.md, CONTRIBUTE.md, DESIGN.md, docs/*)
- [x] Chinchilla baseline (NCPL-OOD val): Huber=0.0037, R²=0.70, MAPE=13.6%
- [x] XGBoost baseline (NCPL-OOD val): Huber=0.0005, R²=0.96, MAPE=1.76%

## What's Left

### Phase 1: Nanochat Data Collection [needs GPU]

Collect (config, loss_curve, raw_code) triples from nanochat/autoresearch runs.
This gives us data from a different ecosystem than NCPL, and includes the code layer.

**Option A: Patch autoresearch to save loss curves**
- Fork autoresearch, add periodic val_bpb logging to a JSON file
- Let it run overnight → 50-100+ diverse (config, loss_curve) pairs automatically
- Each run also has a snapshot of train.py (the code IS the architecture)
- Fastest way to get diverse nanochat data

**Option B: Run nanochat manually with WandB**
- Run `scripts/base_train.py --run=losscast_d{N}` for d=4,8,12,16,24
- Vary: matrix_lr, weight_decay, warmdown_ratio, embedding_lr
- ~20-30 runs, each takes minutes (small models) to hours (d24)
- More controlled, less diverse

**Option C: Both** (recommended)
- Manual runs for controlled coverage (5-10 runs across sizes)
- Autoresearch for volume and diversity (50-100 runs at d=8)

**Deliverable:** `data/` populated with nanochat runs that have config.json + losses.json + model.py

**Data split strategy:**
- Nanochat runs with published code → train/val (public)
- Nanochat runs with unpublished code variants → test (private)
- autoresearch runs where the agent modified train.py → perfect test data (code changes are the signal)

### Phase 2: Baseline Predictors [after Phase 1, ~1-2 days]

Train two predictors to establish non-trivial baselines:

#### 2a. XGBoost on config features
- Input: flatten config.json → feature vector
  - Numeric: log(n_params), log(tokens_total), lr, wd, beta1, beta2, warmup/total ratio, batch_tokens, seq_len, d_model, n_layers, d_ff
  - Categorical (one-hot): optimizer name, activation, norm_type, lr_schedule, eval_dataset
- Target: Chinchilla residual at each step (loss - chinchilla_prediction)
- One model predicts loss at step t given (config_features, t/total_steps)
- Train on NCPL data, evaluate on val
- Expected: should beat Chinchilla significantly on ID, since it learns optimizer effects

#### 2b. LLM predictor on config + raw code
- Input: config.json serialized as text + model.py source code (truncated to fit context)
- Model: fine-tune a small LM (e.g., Qwen2.5-1.5B or SmolLM2-360M) to predict loss curve
- Following NCPL approach: predict Chinchilla residual, numeric MLP for numbers
- Train on NCPL data + nanochat data (with code)
- This is the "fancy" baseline that uses the code layer

**Deliverable:** Two new baselines in `losscast_bench/baselines/`, results in README

#### Implementation location
```
losscast_bench/baselines/
├── chinchilla.py          # exists
├── xgboost_predictor.py   # new: config-feature XGBoost
└── llm_predictor.py       # new: LLM on config+code
```

Training scripts:
```
scripts/
├── train_xgboost.py       # Train XGBoost baseline on train split
└── train_llm.py           # Fine-tune LLM baseline
```

### Phase 3: Test Split [parallel with Phase 2]

Populate `data/test/` with private runs for leaderboard evaluation.

**Source:** autoresearch runs where the agent modified the architecture
- The code changes are unpublished → can't be found in any public dataset
- Run the conversion: `scripts/nanochat_to_bench.py` + copy modified train.py as model.py
- Target: 50-100 private runs

**Important:** test data must never be committed to the public repo.
Store separately (eval server, private bucket, etc.).

### Phase 4: find_similar_runs() API [1 day]

Implement the config nearest-neighbor search promised in README.

```python
from losscast_bench.data import find_similar_runs
similar = find_similar_runs(my_config, split="train", top_k=5)
```

Plus CLI: `losscast-similar --config config.json --top-k 5`

See `losscast-dev/PLANS.md` for detailed design.

### Phase 5: Release Polish [1 day]

- [ ] pyproject.toml: add optional deps (`dev`, `predict`, `convert`)
- [ ] Verify end-to-end: fresh clone → pip install → run baseline → eval
- [ ] Update README baseline table with XGBoost + LLM results
- [ ] Tag v0.1.0
- [ ] Write release announcement

## Timeline

```
Phase 1 (data)     ████████░░░░░░  [needs GPU, 1-3 days]
Phase 2 (baselines)     ░░░░████████  [after data, 1-2 days]
Phase 3 (test)     ████████████░░  [parallel with 1&2, needs GPU]
Phase 4 (API)              ░░████  [1 day, any time]
Phase 5 (polish)              ░░██  [1 day, after all above]
                   ──────────────────
                   ~1 week total with GPU access
```

## Open Questions

1. **GPU access** — where to run nanochat? Lambda, modal.com, local cluster?
2. **Eval server** — GitHub Actions workflow? Hosted service? For v0 maybe just a script that people run locally against their own test data?
3. **LLM predictor size** — Qwen2.5-1.5B (matches NCPL) vs smaller (faster iteration)?
4. **autoresearch integration** — should we contribute a losscast-bench logging hook to autoresearch upstream?
