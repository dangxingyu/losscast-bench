#!/usr/bin/env python3
"""
Convert raw_data/marin/ WandB exports → losscast-bench format.

This is for the directly-downloaded WandB data (via scripts/download_marin_runs.py),
NOT the NCPL HuggingFace dataset. Use ncpl_to_bench.py for that.

Usage:
    python losscast-dev/convert/marin_raw_to_bench.py --output data/staging/
    python losscast-dev/convert/marin_raw_to_bench.py --dry-run
"""

import argparse
import json
import math
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "raw_data" / "marin"

# Same arch table as ncpl_to_bench.py
MARIN_ARCHS = {
    130: {"n_layers": 32, "d_model": 512, "n_heads": 8, "d_ff": 1365, "vocab_size": 32000},
    300: {"n_layers": 32, "d_model": 768, "n_heads": 12, "d_ff": 2048, "vocab_size": 32000},
    520: {"n_layers": 32, "d_model": 1024, "n_heads": 16, "d_ff": 2730, "vocab_size": 32000},
    1200: {"n_layers": 32, "d_model": 1536, "n_heads": 24, "d_ff": 4096, "vocab_size": 32000},
}


def match_arch(n_params: float) -> dict:
    """Match by approximate param count in millions."""
    size_m = round(n_params / 1e6)
    best = min(MARIN_ARCHS.keys(), key=lambda k: abs(k - size_m))
    return MARIN_ARCHS[best].copy()


def convert_run(run_dir: Path, output_dir: Path, dry_run: bool = False) -> bool:
    """Convert a single raw Marin run."""
    meta_path = run_dir / "wandb_meta.json"
    eval_path = run_dir / "eval_losses.json"

    if not meta_path.exists():
        print(f"  SKIP {run_dir.name}: no wandb_meta.json")
        return False
    if not eval_path.exists():
        print(f"  SKIP {run_dir.name}: no eval_losses.json")
        return False

    meta = json.loads(meta_path.read_text())
    parsed = meta.get("parsed", {})

    # Extract c4_en eval losses
    raw_losses = json.loads(eval_path.read_text())
    c4_key = "eval/paloma/c4_en/loss"

    points = []
    for row in raw_losses:
        step = row.get("_step", 0)
        val = row.get(c4_key)
        if val is not None and val != "NaN":
            val = float(val)
            if not math.isnan(val):
                points.append((step, val))

    if len(points) < 3:
        print(f"  SKIP {run_dir.name}: only {len(points)} valid c4_en eval points")
        return False

    # Build run_id
    size = parsed.get("model_size", "unk").lower()
    tokens = parsed.get("tokens", "unk").lower()
    opt = parsed.get("optimizer", "unk")
    run_id = f"marin_{size}_{tokens}_{opt}"

    # Architecture
    n_params = parsed.get("n_params", 130e6)
    arch = match_arch(n_params)

    # Losses
    losses_dict = {str(s): round(l, 6) for s, l in points}

    # Eval interval
    if len(points) >= 2:
        intervals = [points[i+1][0] - points[i][0] for i in range(min(10, len(points)-1))]
        eval_interval = max(set(intervals), key=intervals.count)
    else:
        eval_interval = 500

    config = {
        "run_id": run_id,
        "model": {
            "arch": "transformer",
            **arch,
            "activation": "swiglu",
            "norm_type": "rmsnorm",
            "rope": True,
        },
        "optimizer": {
            "name": opt,
            "lr": parsed.get("lr"),
            "weight_decay": parsed.get("wd", 0.1),
            "beta1": 0.9,
            "beta2": parsed.get("beta2", 0.95),
            "eps": 1e-8,
            "grad_clip": 1.0,
        },
        "data": {
            "dataset": "dolma",
            "tokenizer": "EleutherAI/gpt-neox-20b",
            "eval_dataset": "c4_en",
            "seq_len": 2048,
            "batch_tokens": 524288,  # TODO: extract from WandB if available
            "tokens_total": parsed.get("tokens_total"),
        },
        "schedule": {
            "lr_schedule": "cosine",
            "warmup_steps": parsed.get("warmup", 0),
            "total_steps": points[-1][0],  # best guess from last eval step
            "cooldown_steps": 0,
            "final_lr_ratio": 0.0,
        },
        "eval_interval": eval_interval,
        "precision": "bf16",
    }

    if dry_run:
        print(f"\n  [{run_id}]")
        print(f"    arch: {arch['n_layers']}L d={arch['d_model']}")
        print(f"    opt: {opt} lr={parsed.get('lr')}")
        print(f"    c4_en points: {len(points)}, loss: {min(l for _, l in points):.3f} - {max(l for _, l in points):.3f}")
        return True

    out = output_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out / "losses.json", "w") as f:
        json.dump(losses_dict, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/staging/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)

    runs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(runs)} raw marin runs\n")

    ok, skip = 0, 0
    for run_dir in runs:
        if convert_run(run_dir, output, args.dry_run):
            ok += 1
        else:
            skip += 1

    print(f"\nDone: {ok} converted, {skip} skipped")


if __name__ == "__main__":
    main()
