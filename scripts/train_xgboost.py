#!/usr/bin/env python3
"""
Train the XGBoost baseline on the train split, evaluate on val, and save the
fitted model + predictions.

Usage:
    python scripts/train_xgboost.py
    python scripts/train_xgboost.py --model-out baselines/xgboost.pkl --preds-out xgb_preds.json
    python scripts/train_xgboost.py --n-estimators 1000 --max-depth 10 --verbose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from losscast_bench.baselines.chinchilla import predict_batch as chinchilla_predict
from losscast_bench.baselines.xgboost_baseline import XGBoostPredictor
from losscast_bench.data import load_split
from losscast_bench.metrics.scoring import evaluate
from losscast_bench.schema import save_predictions


def _fmt_result(name: str, res) -> str:
    s = res.summary()
    line = (
        f"{name:>12s}  "
        f"huber={s['huber']:.6f}  "
        f"R²={s['r2']:.4f}  "
        f"MAPE={s['curve_mape']*100:.2f}%  "
        f"final_huber={s['final_huber']:.6f}  "
        f"runs={s['n_runs']}"
    )
    return line


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost baseline for LossCast-Bench")
    parser.add_argument("--model-out", default="baselines/xgboost.pkl",
                        help="Where to save the fitted model")
    parser.add_argument("--preds-out", default="baselines/xgboost_predictions.json",
                        help="Where to save val predictions")
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-child-weight", type=float, default=5.0)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--verbose", action="store_true",
                        help="Print XGBoost training progress")
    args = parser.parse_args()

    print("Loading data...")
    train_configs, train_gt = load_split("train")
    val_configs, val_gt = load_split("val")
    print(f"  train: {len(train_configs)} runs, val: {len(val_configs)} runs")

    print("Fitting XGBoost...")
    predictor = XGBoostPredictor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
    )
    predictor.fit(
        train_configs, train_gt,
        val_configs=val_configs, val_ground_truths=val_gt,
        verbose=args.verbose,
    )

    print(f"Saving model to {args.model_out}...")
    predictor.save(args.model_out)

    print("Predicting on val...")
    xgb_val_preds = predictor.predict_batch(val_configs)
    save_predictions(xgb_val_preds, args.preds_out)
    print(f"  wrote {args.preds_out}")

    # Also predict on train to show train/val gap
    xgb_train_preds = predictor.predict_batch(train_configs)

    # Chinchilla reference on the same splits
    chinchilla_val_preds = chinchilla_predict(val_configs)
    chinchilla_train_preds = chinchilla_predict(train_configs)

    print("\n" + "─" * 68)
    print(f"{'':>12s}  {'metrics vs. ground truth':s}")
    print("─" * 68)
    print(_fmt_result("xgb train",
                     evaluate(xgb_train_preds, train_gt, train_configs)))
    print(_fmt_result("xgb val",
                     evaluate(xgb_val_preds, val_gt, val_configs)))
    print(_fmt_result("chin train",
                     evaluate(chinchilla_train_preds, train_gt, train_configs)))
    print(_fmt_result("chin val",
                     evaluate(chinchilla_val_preds, val_gt, val_configs)))
    print("─" * 68)

    # Dump a small JSON report alongside the predictions
    val_eval = evaluate(xgb_val_preds, val_gt, val_configs)
    chin_eval = evaluate(chinchilla_val_preds, val_gt, val_configs)
    report_path = Path(args.preds_out).with_suffix(".report.json")
    report_path.write_text(json.dumps({
        "xgboost_val": val_eval.summary(),
        "chinchilla_val": chin_eval.summary(),
        "hyperparameters": predictor._config(),
    }, indent=2))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
