# Development Plans

## Plan 1: Private Test Split

### Goal
Populate `data/test/` with private runs whose loss curves have never been published, enabling a real leaderboard.

### Strategy
Run nanochat-style pretraining experiments ourselves. These runs should cover:

1. **In-distribution**: Configs similar to train/val (130M–900M, AdamW/Muon, cosine/linear schedule) to test interpolation ability
2. **OOD - scale**: Configs at sizes not well-represented (>1B or <50M)
3. **OOD - optimizer**: Optimizers not in NCPL (Schedule-Free, Prodigy, etc.)
4. **OOD - architecture**: Non-standard choices (different activation, different norm, GQA at small scale, etc.)
5. **OOD - schedule**: WSD, cyclical LR, very long warmup, no warmup at all

### Minimum viable test set
~50-100 runs is enough for v0:
- 30 ID runs (sanity check — predictor should do well here)
- 20 OOD-optimizer runs
- 20 OOD-scale runs
- 10-20 OOD-architecture runs

### Execution
1. Set up nanochat training on available GPUs
2. Run each experiment, collect config + loss curve
3. Use code-rewriter skill to generate model.py for each
4. Place in `data/test/` (gitignored, only on eval server)
5. Run Chinchilla baseline on test set to establish floor

### Data flow
```
Run experiment → collect losses → code-rewriter → validate → data/test/{run_id}/
```

Test data is NEVER committed to the public repo. It lives only on the eval server.

### Open questions
- Where does the eval server run? (GitHub Actions? A hosted service?)
- How do participants submit their predictor function? (PR with predict.py? API endpoint?)
- How to prevent overfitting to the test set over time? (Periodic refresh with new runs)


## Plan 2: find_similar_runs() API

### Goal
Let agents and humans check whether a config (or something close) has already been tried before running an experiment. This is the "registry" function mentioned in the README.

### Interface

```python
from losscast_bench.data import find_similar_runs

# Find runs most similar to a proposed config
similar = find_similar_runs(
    config=my_config,          # RunConfig or dict
    split="train",             # which split to search
    top_k=5,                   # how many to return
)
# Returns: list of (run_id, similarity_score, RunConfig, Optional[RunGroundTruth])
```

### Similarity metric
Compute distance in a normalized feature space. Features and their weights:

| Feature | Weight | Normalization |
|---------|--------|---------------|
| `log(n_params)` | 1.0 | Divide by log(max_params) |
| `log(tokens_total)` | 1.0 | Divide by log(max_tokens) |
| `optimizer.name` | 0.5 | 1.0 if same, 0.0 if different |
| `optimizer.lr` | 0.3 | Log-scale, normalize by range |
| `optimizer.weight_decay` | 0.2 | Linear normalize |
| `schedule.lr_schedule` | 0.3 | 1.0 if same, 0.0 if different |
| `schedule.warmup_steps / total_steps` | 0.2 | Linear |
| `data.eval_dataset` | 0.5 | 1.0 if same, 0.0 if different |
| `model.activation` | 0.2 | 1.0 if same, 0.0 if different |

Distance = weighted Euclidean over these features. Similarity = 1 / (1 + distance).

### Implementation plan
1. Add `find_similar_runs()` to `losscast_bench/data/__init__.py`
2. Precompute feature vectors for all runs on first call (cache in memory)
3. Add CLI command: `losscast-similar --config config.json --top-k 5`
4. Add to PREDICT.md: "Check what's already been tried before running"

### Stretch goals
- `losscast-similar --config config.json --predict`: also show what the best available predictor says the loss curve would be
- Semantic similarity for optimizer names (muon ≈ mudam > adamw)
- Cluster visualization of the dataset in feature space
