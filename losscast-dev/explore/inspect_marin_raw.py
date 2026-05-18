#!/usr/bin/env python3
"""
Explore raw_data/marin/ WandB exports to understand what we have.

Usage:
    python losscast-dev/explore/inspect_marin_raw.py
"""

import json
import math
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "raw_data" / "marin"


def inspect_run(run_dir: Path):
    """Inspect a single raw marin run directory."""
    info = {"name": run_dir.name}

    # wandb_meta.json — parsed fields
    meta_path = run_dir / "wandb_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        info["parsed"] = meta.get("parsed", {})
        info["state"] = meta.get("state")

    # eval_losses.json — check what we have
    eval_path = run_dir / "eval_losses.json"
    if eval_path.exists():
        losses = json.loads(eval_path.read_text())
        info["n_history_rows"] = len(losses)

        # Count non-NaN c4_en eval points
        c4_key = "eval/paloma/c4_en/loss"
        c4_points = [
            (row["_step"], row[c4_key])
            for row in losses
            if c4_key in row and not (isinstance(row[c4_key], float) and math.isnan(row[c4_key]))
            and row[c4_key] != "NaN"
        ]
        info["c4en_eval_points"] = len(c4_points)
        if c4_points:
            info["c4en_steps_range"] = (c4_points[0][0], c4_points[-1][0])
            info["c4en_loss_range"] = (
                round(min(p[1] for p in c4_points), 4),
                round(max(p[1] for p in c4_points), 4),
            )

        # Train loss
        train_key = "train/loss"
        train_points = [
            row for row in losses
            if train_key in row and not (isinstance(row[train_key], float) and math.isnan(row[train_key]))
            and row[train_key] != "NaN"
        ]
        info["train_loss_points"] = len(train_points)
    else:
        info["eval_losses"] = "MISSING"

    # Check which files exist
    info["files"] = [f.name for f in run_dir.iterdir()]

    return info


def main():
    if not RAW_DIR.exists():
        print(f"No raw data at {RAW_DIR}")
        return

    runs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(runs)} raw marin runs in {RAW_DIR}\n")

    for run_dir in runs:
        info = inspect_run(run_dir)
        parsed = info.get("parsed", {})
        print(f"--- {info['name'][:80]} ---")
        print(f"  Size: {parsed.get('model_size', '?')}, Tokens: {parsed.get('tokens', '?')}, "
              f"Optimizer: {parsed.get('optimizer', '?')}, LR: {parsed.get('lr', '?')}")

        if "c4en_eval_points" in info:
            print(f"  C4-EN eval points: {info['c4en_eval_points']}, "
                  f"steps: {info.get('c4en_steps_range', '?')}, "
                  f"loss: {info.get('c4en_loss_range', '?')}")
            print(f"  Train loss points: {info['train_loss_points']}")
        else:
            print(f"  eval_losses.json: {info.get('eval_losses', 'present')}")
        print(f"  Files: {info['files']}")
        print()


if __name__ == "__main__":
    main()
