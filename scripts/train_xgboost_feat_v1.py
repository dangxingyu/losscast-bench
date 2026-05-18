#!/usr/bin/env python3
"""Train the xgboost_feat_v1 hybrid baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from losscast_bench.baselines.xgboost_feat_v1 import XGBoostFeatV1Predictor
from losscast_bench.data import load_split
from losscast_bench.metrics.scoring import (
    compute_mae,
    compute_rmse,
    compute_spearman,
    evaluate,
)
from losscast_bench.schema import save_predictions


def _fmt(name, result):
    s = result.summary()
    return (
        f"{name:>22s}  huber={s['huber']:.6f}  R2={s['r2']:.4f}  "
        f"MAPE={s['curve_mape']*100:.2f}%  MAE={s['curve_mae']:.4f}  "
        f"final_MAE={s['final_mae']:.4f}  rho={s['final_spearman']:.4f}"
    )


def _source_of(config):
    if config.data.eval_dataset == "c4_en":
        return "marin"
    if config.data.tokenizer == "autoresearch_bpe_8k":
        return "nanochat"
    return "steplaw"


def _per_source(preds, cfgs, gts):
    cfg_by_id = {c.run_id: c for c in cfgs}
    gt_by_id = {g.run_id: g for g in gts}
    by_src = {}
    for p in preds:
        cfg = cfg_by_id[p.run_id]
        gt = gt_by_id[p.run_id]
        src = _source_of(cfg)
        common = sorted(set(p.predictions) & set(gt.losses))
        if not common:
            continue
        final_step = max(common)
        by_src.setdefault(src, {"pred": [], "truth": []})
        by_src[src]["pred"].append(p.predictions[final_step])
        by_src[src]["truth"].append(gt.losses[final_step])
    out = {}
    for src, vals in by_src.items():
        out[src] = {
            "MAE": compute_mae(vals["pred"], vals["truth"]),
            "RMSE": compute_rmse(vals["pred"], vals["truth"]),
            "rho": compute_spearman(vals["pred"], vals["truth"]),
            "n": len(vals["pred"]),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-out", default="baselines/xgboost_feat_v1/")
    parser.add_argument("--preds-out", default="baselines/xgboost_feat_v1_predictions.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    train_cfgs, train_gt = load_split("train")
    val_cfgs, val_gt = load_split("val")
    print(f"train: {len(train_cfgs)} runs, val: {len(val_cfgs)} runs")

    predictor = XGBoostFeatV1Predictor().fit(train_cfgs, train_gt, verbose=args.verbose)
    predictor.save(args.model_out)

    val_preds = predictor.predict_batch(val_cfgs)
    save_predictions(val_preds, args.preds_out)

    result = evaluate(val_preds, val_gt, val_cfgs)
    print()
    print("-" * 120)
    print(_fmt("xgb_feat_v1 val", result))
    print("-" * 120)

    per_source = _per_source(val_preds, val_cfgs, val_gt)
    print()
    print("  Per-source final-loss:")
    for src in ("marin", "steplaw", "nanochat"):
        if src not in per_source:
            continue
        m = per_source[src]
        print(
            f"    {src:8s}  n={m['n']:4d}  "
            f"MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  rho={m['rho']:.4f}"
        )

    report = {
        "aggregate": result.summary(),
        "per_source": per_source,
        "hyperparameters": predictor._feat_v1_meta(),
    }
    Path(args.preds_out).with_suffix(".report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
