#!/usr/bin/env python3
"""End-to-end test: load configs from per-run dirs → run Chinchilla baseline → evaluate."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from losscast_bench.data import load_split
from losscast_bench.schema import validate_submission
from losscast_bench.baselines.chinchilla import predict_batch
from losscast_bench.metrics import evaluate


def main():
    # 1. Load configs and ground truths from per-run directories
    configs, gts = load_split("val")
    print(f"Loaded {len(configs)} configs")
    for c in configs:
        print(f"  {c.run_id}: {c.model.arch} d={c.model.d_model} L={c.model.n_layers} "
              f"~{c.model.n_params_approx / 1e6:.0f}M params, {c.data.tokens_total / 1e9:.0f}B tokens")

    # 2. Run Chinchilla baseline
    predictions = predict_batch(configs)
    print(f"\nGenerated {len(predictions)} predictions")
    for p in predictions:
        print(f"  {p.run_id}: final_loss={p.final_loss:.4f}")

    # 3. Validate
    errors = validate_submission(predictions, configs)
    if errors:
        print(f"\nValidation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print(f"\nValidation: PASSED")

    # 4. Evaluate
    assert gts is not None, "Ground truth not available for val split"
    print(f"\nLoaded {len(gts)} ground truths")

    result = evaluate(predictions, gts, configs)
    print(f"\nEvaluation results:")
    for k, v in result.summary().items():
        print(f"  {k}: {v}")

    # 5. Sanity checks
    assert result.n_runs == 4, f"Expected 4 runs, got {result.n_runs}"
    assert result.n_points == 400, f"Expected 400 points, got {result.n_points}"
    assert result.r2 > 0.5, f"R² too low: {result.r2}"
    assert result.huber < 1.0, f"Huber too high: {result.huber}"

    print("\n ALL TESTS PASSED")


if __name__ == "__main__":
    main()
