#!/usr/bin/env python3
"""Run every trained baseline through the full metric suite on val."""

from __future__ import annotations

import json
from pathlib import Path

from losscast_bench.data import load_split
from losscast_bench.metrics.scoring import (
    evaluate, compute_mae, compute_rmse, compute_spearman,
)
from losscast_bench.baselines.chinchilla import predict_batch as chin_predict
from losscast_bench.baselines.chinchilla_refit import ChinchillaRefitPredictor
from losscast_bench.baselines.xgboost_baseline import XGBoostPredictor
from losscast_bench.baselines.xgboost_v2 import XGBoostPredictorV2
from losscast_bench.baselines.xgboost_ensemble import XGBoostEnsemblePredictor
from losscast_bench.baselines.pca_xgboost import PCAXGBoostPredictor
from losscast_bench.baselines.two_stage import TwoStagePredictor
from losscast_bench.baselines.mlp_baseline import MLPPredictor
from losscast_bench.baselines.xgboost_ultimate import XGBoostUltimatePredictor
from losscast_bench.baselines.xgboost_feat_v1 import XGBoostFeatV1Predictor


_SOURCES = [
    # (eval_dataset_key, tokenizer_filter, display_name)
    ("c4_en", None, "marin"),
    ("train", "gpt2", "steplaw"),
    ("train", "autoresearch_bpe_8k", "nanochat"),
]


def _persrc(preds, cfg_by_id, gt_by_id):
    out = {}
    for eval_ds, tokenizer, src in _SOURCES:
        ps, gs = [], []
        for p in preds:
            cfg = cfg_by_id.get(p.run_id)
            gt = gt_by_id.get(p.run_id)
            if cfg is None or gt is None:
                continue
            if cfg.data.eval_dataset != eval_ds:
                continue
            if tokenizer is not None and cfg.data.tokenizer != tokenizer:
                continue
            common = sorted(set(p.predictions) & set(gt.losses))
            if not common:
                continue
            fs = max(common)
            ps.append(p.predictions[fs])
            gs.append(gt.losses[fs])
        if ps:
            out[src] = {
                "MAE": compute_mae(ps, gs),
                "RMSE": compute_rmse(ps, gs),
                "rho": compute_spearman(ps, gs),
                "n": len(ps),
            }
    return out


def main():
    val_cfgs, val_gt = load_split("val")
    cfg_by_id = {c.run_id: c for c in val_cfgs}
    gt_by_id = {g.run_id: g for g in val_gt}

    bl_specs = [
        ("chinchilla", lambda: chin_predict(val_cfgs)),
        ("chinchilla_refit", lambda: ChinchillaRefitPredictor.load("baselines/chinchilla_refit.json").predict_batch(val_cfgs)),
        ("xgboost", lambda: XGBoostPredictor.load("baselines/xgboost.pkl").predict_batch(val_cfgs)),
        ("xgboost_v2", lambda: XGBoostPredictorV2.load("baselines/xgboost_v2.pkl").predict_batch(val_cfgs)),
        ("xgboost_ensemble", lambda: XGBoostEnsemblePredictor.load("baselines/xgboost_ensemble.pkl").predict_batch(val_cfgs)),
        ("pca_xgboost", lambda: PCAXGBoostPredictor.load("baselines/pca_xgboost.pkl").predict_batch(val_cfgs)),
        ("two_stage", lambda: TwoStagePredictor.load("baselines/two_stage.pkl").predict_batch(val_cfgs)),
        ("mlp", lambda: MLPPredictor.load("baselines/mlp_baseline.pt").predict_batch(val_cfgs)),
        ("xgboost_ultimate", lambda: XGBoostUltimatePredictor.load("baselines/xgboost_ultimate/").predict_batch(val_cfgs)),
        ("xgboost_feat_v1", lambda: XGBoostFeatV1Predictor.load("baselines/xgboost_feat_v1/").predict_batch(val_cfgs)),
    ]

    bl = []
    for name, loader in bl_specs:
        try:
            bl.append((name, loader()))
        except Exception as e:
            print(f"  [skip] {name}: {e}")

    n_val = len(val_cfgs)
    print(f"\n=== Aggregate on val ({n_val} runs) ===")
    hdr = f"{'baseline':<20} {'Huber':>10} {'R2':>7} {'MAPE':>7} {'MAE':>8} {'RMSE':>8} {'fMAE':>8} {'fRMSE':>8} {'rho':>7}"
    print(hdr); print("-" * len(hdr))
    agg = {}
    for name, preds in bl:
        r = evaluate(preds, val_gt, val_cfgs).summary()
        agg[name] = r
        print(f"{name:<20} {r['huber']:>10.6f} {r['r2']:>7.4f} {r['curve_mape']*100:>6.2f}% "
              f"{r['curve_mae']:>8.4f} {r['curve_rmse']:>8.4f} "
              f"{r['final_mae']:>8.4f} {r['final_rmse']:>8.4f} {r['final_spearman']:>7.4f}")

    print("\n=== Per-source final-loss MAE / RMSE / rho ===")
    hdr = (f"{'baseline':<20} {'Marin MAE':>10} {'RMSE':>7} {'rho':>7}"
           f"   {'StepLaw MAE':>11} {'RMSE':>7} {'rho':>7}"
           f"   {'Nanochat MAE':>12} {'RMSE':>7} {'rho':>7}")
    print(hdr); print("-" * len(hdr))
    persrc = {}
    for name, preds in bl:
        persrc[name] = _persrc(preds, cfg_by_id, gt_by_id)
        m = persrc[name].get("marin", {})
        s = persrc[name].get("steplaw", {})
        nc = persrc[name].get("nanochat", {})
        print(
            f"{name:<20}"
            f" {m.get('MAE', float('nan')):>10.4f} {m.get('RMSE', float('nan')):>7.4f} {m.get('rho', float('nan')):>7.4f}"
            f"   {s.get('MAE', float('nan')):>11.4f} {s.get('RMSE', float('nan')):>7.4f} {s.get('rho', float('nan')):>7.4f}"
            f"   {nc.get('MAE', float('nan')):>12.4f} {nc.get('RMSE', float('nan')):>7.4f} {nc.get('rho', float('nan')):>7.4f}"
        )

    with open("baselines/full_comparison.json", "w") as f:
        json.dump({"aggregate": agg, "per_source": persrc}, f, indent=2)
    print("\nWrote baselines/full_comparison.json")


if __name__ == "__main__":
    main()
