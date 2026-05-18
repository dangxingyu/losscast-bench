#!/usr/bin/env python3
"""
Assign runs from data/staging/ into train/val splits.

All NCPL data is public, so it goes into train + val only.
Test split is reserved for private runs (your own experiments).

Two modes are supported:

  ncpl-ood  (default, recommended):
    Val = runs with n_params > 430M (matches NCPL's published OOD-val rule
    exactly). Train = runs with n_params ≤ 430M. Deterministic, leakage-proof
    against any predictor that trained on NCPL's own train + ID-val pool
    (which by construction contains only ≤430M runs).

  group-stratified:
    Legacy mode — groups runs by (source, optimizer, N_bucket, D_bucket) and
    does a group-level 80/20 random split with seed=42. This re-randomizes
    NCPL's split and risks leakage against NCPL-trained predictors.

Usage:
    python losscast-dev/splits/build_splits.py --input data/staging/ --output data/ --dry-run
    python losscast-dev/splits/build_splits.py --input data/staging/ --output data/
    python losscast-dev/splits/build_splits.py --mode group-stratified ...
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


TRAIN_RATIO = 0.8
SEED = 42
NCPL_OOD_THRESHOLD = 430e6  # NCPL paper's OOD cutoff: n_params > 430M → OOD-val


def estimate_params(config: dict) -> float:
    """Estimate parameter count from config."""
    # Prefer explicit n_params if present (set during conversion)
    if config.get("n_params"):
        return config["n_params"]

    m = config.get("model", {})
    d = m.get("d_model", 512)
    n_layers = m.get("n_layers", 12)
    d_ff = m.get("d_ff") or 4 * d
    n_heads = m.get("n_heads", 8)
    head_dim = m.get("head_dim") or d // n_heads
    vocab = m.get("vocab_size", 32000)

    attn_params = 4 * d * (n_heads * head_dim) * n_layers
    act = m.get("activation", "gelu")
    ffn_mult = 3 if "swiglu" in act.lower() or "silu" in act.lower() else 2
    ffn_params = ffn_mult * d * d_ff * n_layers
    emb_params = vocab * d

    return attn_params + ffn_params + emb_params


def bucket_key(config: dict) -> tuple:
    """Create grouping key: (source, optimizer, size_bucket_M, tokens_bucket_B).

    Includes source (marin/steplaw) so groups don't mix data sources.
    """
    # Infer source from eval_dataset
    eval_ds = config.get("data", {}).get("eval_dataset", "")
    source = "marin" if eval_ds == "c4_en" else "steplaw"

    opt = config.get("optimizer", {}).get("name", "unknown")

    params = estimate_params(config)
    size_bucket = round(params / 50e6) * 50  # nearest 50M

    tokens = config.get("data", {}).get("tokens_total", 0) or 0
    tokens_bucket = round(tokens / 1e9)  # nearest 1B

    return (source, opt, size_bucket, tokens_bucket)


def load_staging(staging_dir: Path) -> list[tuple[str, dict]]:
    """Load all (run_id, config) pairs from staging directory."""
    runs = []
    for run_dir in sorted(staging_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        runs.append((run_dir.name, config))
    return runs


def assign_splits_group_stratified(
    runs: list[tuple[str, dict]],
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
) -> dict[str, list[str]]:
    """Legacy mode: group-stratified 80/20 random split."""
    rng = random.Random(seed)

    groups = defaultdict(list)
    for run_id, config in runs:
        key = bucket_key(config)
        groups[key].append(run_id)

    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)

    train_ids = []
    val_ids = []
    n_train_groups = max(1, int(len(group_keys) * train_ratio))

    for i, key in enumerate(group_keys):
        if i < n_train_groups:
            train_ids.extend(groups[key])
        else:
            val_ids.extend(groups[key])

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
    }


def assign_splits_ncpl_ood(
    runs: list[tuple[str, dict]],
    threshold: float = NCPL_OOD_THRESHOLD,
) -> dict[str, list[str]]:
    """NCPL-aligned mode: val = runs with n_params > threshold.

    Matches NCPL's published OOD-val partition exactly. A predictor trained on
    NCPL's own train + ID-val pool (which contains only ≤430M runs by
    construction) cannot have seen any run now in our val set — so this mode is
    leakage-proof against any honest NCPL-trained predictor.
    """
    train_ids = []
    val_ids = []
    for run_id, config in runs:
        n_params = estimate_params(config)
        if n_params > threshold:
            val_ids.append(run_id)
        else:
            train_ids.append(run_id)
    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
    }


def assign_splits(
    runs: list[tuple[str, dict]],
    mode: str = "ncpl-ood",
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
    threshold: float = NCPL_OOD_THRESHOLD,
) -> dict[str, list[str]]:
    if mode == "ncpl-ood":
        return assign_splits_ncpl_ood(runs, threshold=threshold)
    if mode == "group-stratified":
        return assign_splits_group_stratified(runs, train_ratio=train_ratio, seed=seed)
    raise ValueError(f"Unknown split mode: {mode!r}")


def print_split_stats(split_name: str, run_ids: list[str], runs: list[tuple[str, dict]]):
    """Print summary stats for a split."""
    run_id_set = set(run_ids)
    configs = [config for run_id, config in runs if run_id in run_id_set]
    if not configs:
        return

    params = [estimate_params(c) for c in configs]
    print(f"\n--- {split_name} ({len(run_ids)} runs) ---")
    print(f"  Params range: {min(params)/1e6:.0f}M - {max(params)/1e6:.0f}M")

    opts = defaultdict(int)
    sources = defaultdict(int)
    for c in configs:
        opts[c.get("optimizer", {}).get("name", "?")] += 1
        ed = c.get("data", {}).get("eval_dataset", "?")
        sources["marin" if ed == "c4_en" else "steplaw"] += 1

    print(f"  Sources: {dict(sources)}")
    print(f"  Optimizers: {dict(sorted(opts.items(), key=lambda x: -x[1]))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/staging/", help="Staging directory with all runs")
    parser.add_argument("--output", default="data/", help="Output data directory")
    parser.add_argument("--mode", choices=["ncpl-ood", "group-stratified"], default="ncpl-ood",
                        help="Split strategy (see module docstring)")
    parser.add_argument("--threshold", type=float, default=NCPL_OOD_THRESHOLD,
                        help="OOD parameter-count threshold (ncpl-ood mode only)")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO,
                        help="Train fraction (group-stratified mode only)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="RNG seed (group-stratified mode only)")
    parser.add_argument("--clean", action="store_true",
                        help="Remove existing split directories before writing")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    staging = Path(args.input)
    output = Path(args.output)

    runs = load_staging(staging)
    print(f"Loaded {len(runs)} runs from {staging} (mode={args.mode})")

    splits = assign_splits(
        runs,
        mode=args.mode,
        train_ratio=args.train_ratio,
        seed=args.seed,
        threshold=args.threshold,
    )

    for split_name, run_ids in splits.items():
        print(f"  {split_name}: {len(run_ids)} runs")

    for split_name, run_ids in splits.items():
        print_split_stats(split_name, run_ids, runs)

    if args.dry_run:
        return

    # Clean existing split dirs if requested
    if args.clean:
        for split_name in splits.keys():
            split_dir = output / split_name
            if split_dir.exists():
                print(f"Removing existing {split_dir}/ ({sum(1 for _ in split_dir.iterdir())} runs)")
                shutil.rmtree(split_dir)

    # Copy runs to split directories
    for split_name, run_ids in splits.items():
        split_dir = output / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for run_id in run_ids:
            src = staging / run_id
            dst = split_dir / run_id
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

    # Save split manifest
    manifest = {k: v for k, v in splits.items()}
    meta = {
        "mode": args.mode,
        "total_runs": len(runs),
        "note": "test split is reserved for private runs, not included here",
    }
    if args.mode == "ncpl-ood":
        meta["ncpl_ood_threshold"] = args.threshold
        meta["rationale"] = (
            "Val = runs with n_params > threshold, matching NCPL's published "
            "OOD-val partition. Leakage-proof against NCPL-trained predictors."
        )
    else:
        meta["train_ratio"] = args.train_ratio
        meta["seed"] = args.seed
    manifest["_meta"] = meta
    with open(output / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote splits to {output}/ and saved split_manifest.json")


if __name__ == "__main__":
    main()
