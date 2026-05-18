# LossCast-Bench Evaluation Metrics

## Primary Metric: Huber Loss

The primary ranking metric is mean Huber loss across all (run, step) pairs:

$$\text{Huber}(y, \hat{y}; \delta) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta \cdot (|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

Default δ = 0.01, following NCPL. This is robust to outliers while still penalizing large errors.

## Full Metric Suite

Submissions are ranked by a single leaderboard. All metrics are computed from the same submission; there is no separate "track" — participants always predict at all eval steps.

| Metric | Scope | Description |
|---|---|---|
| **Huber** | All points | Mean Huber loss over all (run, step) pairs. **Primary ranking metric.** |
| **R²** | All points | Coefficient of determination. Measures explained variance. |
| **Final Huber** | Final step only | Huber loss on the last checkpoint of each run. |
| **Final R²** | Final step only | R² on final loss predictions. |
| **Curve MAPE** | All points | Mean absolute percentage error across the full curve. |
| **Extrap Huber** | High-compute runs | Huber on runs exceeding a compute threshold. Tests generalization beyond training distribution. |
| **Extrap R²** | High-compute runs | R² on extrapolation set. |

The **primary ranking** uses the overall Huber loss (all points). Final-loss and curve-level metrics are reported alongside it to give participants diagnostic insight into where their model excels or struggles.

## Extrapolation Evaluation

Runs in the test set include configurations with significantly more compute than anything in the training set. The extrapolation metrics specifically measure performance on these out-of-distribution runs.

The compute threshold is defined as `tokens_total > max(tokens_total in training set)`.

## Usage

```bash
# Evaluate against a data split (loads configs + ground truth from per-run directories)
losscast-eval --split val -p predictions.json

# Or specify files explicitly
losscast-eval -p predictions.json -g ground_truth.json -c configs.json

# With extrapolation scoring
losscast-eval --split val -p predictions.json --extrap-threshold 50e9

# Per-run breakdown
losscast-eval --split val -p predictions.json --format table --verbose

# Machine-readable JSON output
losscast-eval --split val -p predictions.json --format json

# Save results to file
losscast-eval --split val -p predictions.json -o results.json
```
