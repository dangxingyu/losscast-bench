#!/usr/bin/env python3
"""
Fit Chinchilla coefficients on the train split and evaluate on val.

Usage:
    python scripts/train_chinchilla_refit.py
    python scripts/train_chinchilla_refit.py --fit-out baselines/chinchilla_refit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from losscast_bench.baselines.chinchilla_refit import ChinchillaRefitPredictor
from losscast_bench.data import load_split
from losscast_bench.metrics.scoring import evaluate
from losscast_bench.schema import save_predictions


def _fmt(name: str, res) -> str:
    s = res.summary()
    return (
        f"{name:>22s}  "
        f"huber={s['huber']:.6f}  R²={s['r2']:.4f}  "
        f"MAPE={s['curve_mape']*100:.2f}%  "
        f"final_MAE={s['final_mae']:.4f}  "
        f"final_RMSE={s['final_rmse']:.4f}  "
        f"ρ={s['final_spearman']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-out", default="baselines/chinchilla_refit.json",
                        help="Where to save the fitted coefficients")
    parser.add_argument("--preds-out", default="baselines/chinchilla_refit_predictions.json",
                        help="Where to save val predictions")
    args = parser.parse_args()

    print("Loading train/val...")
    train_cfgs, train_gt = load_split("train")
    val_cfgs, val_gt = load_split("val")
    print(f"  train: {len(train_cfgs)} runs, val: {len(val_cfgs)} runs")

    print("Fitting Chinchilla coefficients on train...")
    predictor = ChinchillaRefitPredictor().fit_from(train_cfgs, train_gt)
    fit = predictor.fit
    print(
        f"  fit: E={fit.E:.4f}  A={fit.A:.2f}  B={fit.B:.2f}  "
        f"α={fit.alpha:.4f}  β={fit.beta:.4f}"
    )

    print(f"Saving coefficients to {args.fit_out}...")
    predictor.save(args.fit_out)

    print("Predicting on train + val...")
    train_preds = predictor.predict_batch(train_cfgs)
    val_preds = predictor.predict_batch(val_cfgs)
    save_predictions(val_preds, args.preds_out)
    print(f"  wrote {args.preds_out}")

    print()
    print("─" * 110)
    print(_fmt("chin_refit train", evaluate(train_preds, train_gt, train_cfgs)))
    print(_fmt("chin_refit val", evaluate(val_preds, val_gt, val_cfgs)))
    print("─" * 110)

    report = {
        "fit": fit.to_dict(),
        "train": evaluate(train_preds, train_gt, train_cfgs).summary(),
        "val": evaluate(val_preds, val_gt, val_cfgs).summary(),
    }
    report_path = Path(args.preds_out).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
