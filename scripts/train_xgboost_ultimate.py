#!/usr/bin/env python3
"""Train xgboost_ultimate: per-source CatBoost + per-source PCA-XGBoost blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from losscast_bench.baselines.xgboost_ultimate import XGBoostUltimatePredictor
from losscast_bench.data import load_split
from losscast_bench.metrics.scoring import (
    compute_mae, compute_rmse, compute_spearman, evaluate,
)
from losscast_bench.schema import save_predictions


def persource(preds, val_cfgs, val_gt):
    cfg_by_id = {c.run_id: c for c in val_cfgs}
    gt_by_id = {g.run_id: g for g in val_gt}
    out = {}
    sources = [
        ("c4_en", None, "marin"),
        ("train", "gpt2", "steplaw"),
        ("train", "autoresearch_bpe_8k", "nanochat"),
    ]
    for eval_ds, tokenizer, src in sources:
        ps, gs = [], []
        for p in preds:
            cfg = cfg_by_id.get(p.run_id); gt = gt_by_id.get(p.run_id)
            if cfg is None or gt is None:
                continue
            if cfg.data.eval_dataset != eval_ds:
                continue
            if tokenizer is not None and cfg.data.tokenizer != tokenizer:
                continue
            common = sorted(set(p.predictions) & set(gt.losses))
            if not common: continue
            fs = max(common)
            ps.append(p.predictions[fs]); gs.append(gt.losses[fs])
        if ps:
            out[src] = {
                "MAE": compute_mae(ps, gs), "RMSE": compute_rmse(ps, gs),
                "rho": compute_spearman(ps, gs), "n": len(ps),
            }
    return out


def _fmt(name, r):
    s = r.summary()
    return (
        f"{name:>22s}  huber={s['huber']:.6f}  R²={s['r2']:.4f}  "
        f"MAPE={s['curve_mape']*100:.2f}%  MAE={s['curve_mae']:.4f}  "
        f"final_MAE={s['final_mae']:.4f}  ρ={s['final_spearman']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-out", default="baselines/xgboost_ultimate/")
    parser.add_argument("--preds-out", default="baselines/xgboost_ultimate_predictions.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    train_cfgs, train_gt = load_split("train")
    val_cfgs, val_gt = load_split("val")
    print(f"train: {len(train_cfgs)} runs, val: {len(val_cfgs)} runs")

    predictor = XGBoostUltimatePredictor().fit(train_cfgs, train_gt, verbose=args.verbose)
    predictor.save(args.model_out)

    val_preds = predictor.predict_batch(val_cfgs)
    save_predictions(val_preds, args.preds_out)

    print()
    print("─" * 120)
    print(_fmt("ultimate val", evaluate(val_preds, val_gt, val_cfgs)))
    print("─" * 120)

    ps = persource(val_preds, val_cfgs, val_gt)
    print()
    print("  Per-source final-loss (directly vs NCPL Table 1):")
    for src in ("marin", "steplaw", "nanochat"):
        if src not in ps:
            continue
        m = ps[src]
        print(f"    {src:8s}  n={m['n']:4d}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  ρ={m['rho']:.4f}")

    report = {
        "aggregate": evaluate(val_preds, val_gt, val_cfgs).summary(),
        "per_source": ps,
    }
    Path(args.preds_out).with_suffix(".report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
