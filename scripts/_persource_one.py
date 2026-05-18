#!/usr/bin/env python3
"""Compute per-source final-loss MAE/RMSE/ρ for ONE baseline.

Invoked per-baseline so each run is a clean process (avoids the multi-model
loading deadlock we hit when chaining XGBoost + sklearn PCA + torch in one
interpreter). Writes a single CSV line to stdout.
"""

from __future__ import annotations

import argparse
import sys

from losscast_bench.data import load_split
from losscast_bench.metrics.scoring import compute_mae, compute_rmse, compute_spearman


def predictions_for(name: str, configs):
    if name == "chinchilla":
        from losscast_bench.baselines.chinchilla import predict_batch
        return predict_batch(configs)
    if name == "chinchilla_refit":
        from losscast_bench.baselines.chinchilla_refit import ChinchillaRefitPredictor
        return ChinchillaRefitPredictor.load("baselines/chinchilla_refit.json").predict_batch(configs)
    if name == "xgboost":
        from losscast_bench.baselines.xgboost_baseline import XGBoostPredictor
        return XGBoostPredictor.load("baselines/xgboost.pkl").predict_batch(configs)
    if name == "xgboost_v2":
        from losscast_bench.baselines.xgboost_v2 import XGBoostPredictorV2
        return XGBoostPredictorV2.load("baselines/xgboost_v2.pkl").predict_batch(configs)
    if name == "xgboost_ensemble":
        from losscast_bench.baselines.xgboost_ensemble import XGBoostEnsemblePredictor
        return XGBoostEnsemblePredictor.load("baselines/xgboost_ensemble.pkl").predict_batch(configs)
    if name == "pca_xgboost":
        from losscast_bench.baselines.pca_xgboost import PCAXGBoostPredictor
        return PCAXGBoostPredictor.load("baselines/pca_xgboost.pkl").predict_batch(configs)
    if name == "two_stage":
        from losscast_bench.baselines.two_stage import TwoStagePredictor
        return TwoStagePredictor.load("baselines/two_stage.pkl").predict_batch(configs)
    if name == "mlp":
        from losscast_bench.baselines.mlp_baseline import MLPPredictor
        return MLPPredictor.load("baselines/mlp_baseline.pt").predict_batch(configs)
    raise ValueError(f"unknown baseline: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()

    val_cfgs, val_gt = load_split("val")
    cfg_by_id = {c.run_id: c for c in val_cfgs}
    gt_by_id = {g.run_id: g for g in val_gt}

    preds = predictions_for(args.name, val_cfgs)

    out = [args.name]
    for key, src in [("c4_en", "marin"), ("train", "steplaw")]:
        ps, gs = [], []
        for p in preds:
            cfg = cfg_by_id.get(p.run_id)
            gt = gt_by_id.get(p.run_id)
            if cfg is None or gt is None:
                continue
            if cfg.data.eval_dataset != key:
                continue
            common = sorted(set(p.predictions) & set(gt.losses))
            if not common:
                continue
            fs = max(common)
            ps.append(p.predictions[fs])
            gs.append(gt.losses[fs])
        mae = compute_mae(ps, gs)
        rmse = compute_rmse(ps, gs)
        rho = compute_spearman(ps, gs)
        out += [f"{mae:.4f}", f"{rmse:.4f}", f"{rho:.4f}"]

    print(",".join(out))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
