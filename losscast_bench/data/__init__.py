"""
Data loading utilities for LossCast-Bench.

Scans per-run directories under data/{split}/ to load configs and ground truths.

Usage:
    from losscast_bench.data import load_split, list_splits

    configs, ground_truths = load_split("val")
    configs, ground_truths = load_split("train")
    configs, ground_truths = load_split("test")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..schema import (
    RunConfig,
    RunGroundTruth,
    RunPrediction,
    load_predictions,
)


# Default data directory: <project_root>/data/
SPLITS_DIR = Path(__file__).resolve().parent.parent.parent / "data"

VALID_SPLITS = ("train", "val", "test")


def _resolve_split_dir(split: str, data_dir: Optional[str | Path] = None) -> Path:
    """Resolve the directory for a given split name."""
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Unknown split {split!r}. Must be one of: {', '.join(VALID_SPLITS)}"
        )
    base = Path(data_dir) if data_dir else SPLITS_DIR
    split_dir = base / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}. "
            f"Available splits: {', '.join(list_splits(data_dir))}"
        )
    return split_dir


def list_splits(data_dir: Optional[str | Path] = None) -> list[str]:
    """List available data splits that have at least one run directory."""
    base = Path(data_dir) if data_dir else SPLITS_DIR
    found = []
    for name in VALID_SPLITS:
        split_dir = base / name
        if split_dir.exists() and any(
            (d / "config.json").exists() for d in split_dir.iterdir() if d.is_dir()
        ):
            found.append(name)
    return found


def list_runs(split: str, data_dir: Optional[str | Path] = None) -> list[str]:
    """List all run IDs in a split."""
    split_dir = _resolve_split_dir(split, data_dir)
    return sorted(
        d.name for d in split_dir.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    )


def load_run(run_dir: Path) -> tuple[RunConfig, Optional[RunGroundTruth]]:
    """Load a single run from its directory."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")

    with open(config_path) as f:
        config_data = json.load(f)
    config = RunConfig.from_dict(config_data)

    losses_path = run_dir / "losses.json"
    gt = None
    if losses_path.exists():
        with open(losses_path) as f:
            losses_data = json.load(f)
        gt = RunGroundTruth(
            run_id=config.run_id,
            losses={int(k): v for k, v in losses_data.items()},
        )

    return config, gt


def load_split(
    split: str,
    data_dir: Optional[str | Path] = None,
) -> tuple[list[RunConfig], Optional[list[RunGroundTruth]]]:
    """
    Load configs and (if available) ground truths for a split.

    Scans all subdirectories of data/{split}/ for config.json + losses.json.

    Returns:
        (configs, ground_truths) — ground_truths is None if no losses.json
        files exist in this split.
    """
    split_dir = _resolve_split_dir(split, data_dir)

    configs = []
    ground_truths = []

    for run_dir in sorted(split_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "config.json").exists():
            continue
        config, gt = load_run(run_dir)
        configs.append(config)
        if gt is not None:
            ground_truths.append(gt)

    return configs, ground_truths if ground_truths else None


def load_split_predictions(
    split: str,
    predictions_path: str | Path,
    data_dir: Optional[str | Path] = None,
) -> tuple[list[RunConfig], list[RunGroundTruth], list[RunPrediction]]:
    """
    Load configs, ground truths, and a predictions file for evaluation.

    Convenience wrapper for the typical evaluation workflow:
        configs, gts, preds = load_split_predictions("val", "my_preds.json")
        result = evaluate(preds, gts, configs)

    Raises FileNotFoundError if ground truth is not available for this split.
    """
    configs, ground_truths = load_split(split, data_dir)
    if ground_truths is None:
        raise FileNotFoundError(
            f"Ground truth not available for split {split!r}. "
            f"Cannot evaluate without ground truth."
        )
    predictions = load_predictions(str(predictions_path))
    return configs, ground_truths, predictions
