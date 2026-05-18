#!/usr/bin/env python3
"""Train MLP baseline on the benchmark's train split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from losscast_bench.baselines.mlp_baseline import MLPPredictor
from losscast_bench.data import load_split
from losscast_bench.metrics.scoring import evaluate
from losscast_bench.schema import save_predictions


def _fmt(name, r):
    s = r.summary()
    return (
        f"{name:>14s}  huber={s['huber']:.6f}  R²={s['r2']:.4f}  "
        f"MAPE={s['curve_mape']*100:.2f}%  MAE={s['curve_mae']:.4f}  "
        f"final_MAE={s['final_mae']:.4f}  ρ={s['final_spearman']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-out", default="baselines/mlp_baseline.pt")
    parser.add_argument("--preds-out", default="baselines/mlp_predictions.json")
    parser.add_argument("--hidden", default="256,256,128")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    train_cfgs, train_gt = load_split("train")
    val_cfgs, val_gt = load_split("val")
    print(f"train: {len(train_cfgs)} runs, val: {len(val_cfgs)} runs")

    hidden_dims = tuple(int(x) for x in args.hidden.split(","))
    predictor = MLPPredictor(
        hidden_dims=hidden_dims,
        max_epochs=args.max_epochs,
        lr=args.lr,
    ).fit(train_cfgs, train_gt, verbose=args.verbose)
    predictor.save(args.model_out)

    val_preds = predictor.predict_batch(val_cfgs)
    train_preds = predictor.predict_batch(train_cfgs)
    save_predictions(val_preds, args.preds_out)

    print("─" * 120)
    print(_fmt("mlp train", evaluate(train_preds, train_gt, train_cfgs)))
    print(_fmt("mlp val", evaluate(val_preds, val_gt, val_cfgs)))
    print("─" * 120)

    report = {
        "train": evaluate(train_preds, train_gt, train_cfgs).summary(),
        "val": evaluate(val_preds, val_gt, val_cfgs).summary(),
    }
    Path(args.preds_out).with_suffix(".report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
