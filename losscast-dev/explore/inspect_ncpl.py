#!/usr/bin/env python3
"""
Explore the NCPL HuggingFace dataset to understand structure before conversion.

Usage:
    pip install datasets
    python losscast-dev/explore/inspect_ncpl.py
"""

from datasets import load_dataset


def inspect_marin():
    print("=" * 60)
    print("MARIN CONFIG")
    print("=" * 60)

    ds = load_dataset("zhqwqwq/NCPL-Pretraining-Logs", "marin", split="train")
    print(f"Rows: {len(ds)}")
    print(f"Features: {list(ds.features.keys())}")
    print()

    # Show one example
    row = ds[0]
    print("--- Example row ---")
    for k, v in row.items():
        if isinstance(v, list) and len(v) > 5:
            print(f"  {k}: [{v[0]}, {v[1]}, ..., {v[-1]}] (len={len(v)})")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                if isinstance(vv, list) and len(vv) > 3:
                    print(f"    {kk}: [len={len(vv)}]")
                else:
                    print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    # Summary stats
    print("\n--- Summary ---")
    conf_keys = list(ds[0]["conf"].keys()) if "conf" in ds.features else []
    print(f"Config keys: {conf_keys}")

    import collections

    # Model size distribution (key has space: "model size")
    size_key = "model size" if "model size" in ds[0]["conf"] else "model_size"
    sizes = [row["conf"][size_key] for row in ds]
    size_counts = collections.Counter([round(s / 1e6) for s in sizes])
    print(f"Model sizes (MB, count): {dict(sorted(size_counts.items()))}")

    # Optimizer distribution
    opts = [row["conf"]["optimizer"] for row in ds]
    opt_counts = collections.Counter(opts)
    print(f"Optimizers: {dict(opt_counts.most_common())}")

    # LR schedule distribution
    sched_key = "lr schedule" if "lr schedule" in ds[0]["conf"] else "lr_schedule"
    scheds = [row["conf"][sched_key] for row in ds]
    sched_counts = collections.Counter(scheds)
    print(f"LR schedules: {dict(sched_counts.most_common())}")

    # Data size distribution
    data_key = "data size" if "data size" in ds[0]["conf"] else "data_size"
    data_sizes = [round(row["conf"][data_key] / 1e9, 1) for row in ds]
    data_counts = collections.Counter(data_sizes)
    print(f"Data sizes (GB, count): {dict(sorted(data_counts.items()))}")

    # Loss curve lengths
    if "c4en_steps" in ds.features:
        lengths = [len(row["c4en_steps"]) for row in ds]
        print(f"Loss curve lengths: min={min(lengths)}, max={max(lengths)}, median={sorted(lengths)[len(lengths)//2]}")

    return ds


def inspect_steplaw():
    print("\n" + "=" * 60)
    print("STEPLAW CONFIG")
    print("=" * 60)

    ds = load_dataset("zhqwqwq/NCPL-Pretraining-Logs", "steplaw", split="train")
    print(f"Rows: {len(ds)}")
    print(f"Features: {list(ds.features.keys())}")
    print()

    row = ds[0]
    print("--- Example row ---")
    for k, v in row.items():
        if isinstance(v, list) and len(v) > 5:
            print(f"  {k}: [{v[0]}, {v[1]}, ..., {v[-1]}] (len={len(v)})")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                if isinstance(vv, list) and len(vv) > 3:
                    print(f"    {kk}: [len={len(vv)}]")
                else:
                    print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    # Summary stats
    print("\n--- Summary ---")

    import collections

    size_key = "model size" if "model size" in ds[0]["conf"] else "model_size"
    sizes = [row["conf"][size_key] for row in ds]
    size_counts = collections.Counter([round(s / 1e6) for s in sizes])
    print(f"Model sizes (MB, count): {dict(sorted(size_counts.items()))}")

    opts = [row["conf"]["optimizer"] for row in ds]
    opt_counts = collections.Counter(opts)
    print(f"Optimizers: {dict(opt_counts.most_common())}")

    if "smoothed_loss_steps" in ds.features:
        lengths = [len(row["smoothed_loss_steps"]) for row in ds]
        print(f"Loss curve lengths: min={min(lengths)}, max={max(lengths)}, median={sorted(lengths)[len(lengths)//2]}")

    return ds


if __name__ == "__main__":
    marin_ds = inspect_marin()
    steplaw_ds = inspect_steplaw()
