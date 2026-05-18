#!/usr/bin/env python3
"""
Run a baseline method on configs and save predictions.

Usage:
    python scripts/run_baseline.py --split val --method chinchilla --output predictions.json
    python scripts/run_baseline.py --configs configs.json --method chinchilla --output predictions.json
"""

import argparse
from losscast_bench.schema import load_configs, save_predictions
from losscast_bench.data import load_split
from losscast_bench.baselines.chinchilla import predict_batch as chinchilla_predict


def _xgboost_predict(configs):
    from losscast_bench.baselines.xgboost_baseline import XGBoostPredictor
    from pathlib import Path
    model_path = "baselines/xgboost.pkl"
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No fitted XGBoost model at {model_path}. "
            "Run scripts/train_xgboost.py first."
        )
    return XGBoostPredictor.load(model_path).predict_batch(configs)


def _chinchilla_refit_predict(configs):
    from losscast_bench.baselines.chinchilla_refit import ChinchillaRefitPredictor
    from pathlib import Path
    fit_path = "baselines/chinchilla_refit.json"
    if not Path(fit_path).exists():
        raise FileNotFoundError(
            f"No fitted coefficients at {fit_path}. "
            "Run scripts/train_chinchilla_refit.py first."
        )
    return ChinchillaRefitPredictor.load(fit_path).predict_batch(configs)


def _xgboost_v2_predict(configs):
    from losscast_bench.baselines.xgboost_v2 import XGBoostPredictorV2
    from pathlib import Path
    p = "baselines/xgboost_v2.pkl"
    if not Path(p).exists():
        raise FileNotFoundError(f"No fitted model at {p}. Run scripts/train_xgboost_v2.py.")
    return XGBoostPredictorV2.load(p).predict_batch(configs)


def _xgboost_ensemble_predict(configs):
    from losscast_bench.baselines.xgboost_ensemble import XGBoostEnsemblePredictor
    from pathlib import Path
    p = "baselines/xgboost_ensemble.pkl"
    if not Path(p).exists():
        raise FileNotFoundError(f"No fitted model at {p}. Run scripts/train_xgboost_ensemble.py.")
    return XGBoostEnsemblePredictor.load(p).predict_batch(configs)


def _pca_xgboost_predict(configs):
    from losscast_bench.baselines.pca_xgboost import PCAXGBoostPredictor
    from pathlib import Path
    p = "baselines/pca_xgboost.pkl"
    if not Path(p).exists():
        raise FileNotFoundError(f"No fitted model at {p}. Run scripts/train_pca_xgboost.py.")
    return PCAXGBoostPredictor.load(p).predict_batch(configs)


def _two_stage_predict(configs):
    from losscast_bench.baselines.two_stage import TwoStagePredictor
    from pathlib import Path
    p = "baselines/two_stage.pkl"
    if not Path(p).exists():
        raise FileNotFoundError(f"No fitted model at {p}. Run scripts/train_two_stage.py.")
    return TwoStagePredictor.load(p).predict_batch(configs)


def _mlp_predict(configs):
    from losscast_bench.baselines.mlp_baseline import MLPPredictor
    from pathlib import Path
    p = "baselines/mlp_baseline.pt"
    if not Path(p).exists():
        raise FileNotFoundError(f"No fitted model at {p}. Run scripts/train_mlp.py.")
    return MLPPredictor.load(p).predict_batch(configs)


def _ultimate_predict(configs):
    from losscast_bench.baselines.xgboost_ultimate import XGBoostUltimatePredictor
    from pathlib import Path
    p = "baselines/xgboost_ultimate/"
    if not Path(p).exists():
        raise FileNotFoundError(f"No fitted model at {p}. Run scripts/train_xgboost_ultimate.py.")
    return XGBoostUltimatePredictor.load(p).predict_batch(configs)


def _xgboost_feat_v1_predict(configs):
    from losscast_bench.baselines.xgboost_feat_v1 import XGBoostFeatV1Predictor
    from pathlib import Path
    p = "baselines/xgboost_feat_v1/"
    if not Path(p).exists():
        raise FileNotFoundError(f"No fitted model at {p}. Run scripts/train_xgboost_feat_v1.py.")
    return XGBoostFeatV1Predictor.load(p).predict_batch(configs)


METHODS = {
    "chinchilla": chinchilla_predict,
    "chinchilla_refit": _chinchilla_refit_predict,
    "xgboost": _xgboost_predict,
    "xgboost_v2": _xgboost_v2_predict,
    "xgboost_ensemble": _xgboost_ensemble_predict,
    "pca_xgboost": _pca_xgboost_predict,
    "two_stage": _two_stage_predict,
    "mlp": _mlp_predict,
    "xgboost_ultimate": _ultimate_predict,
    "xgboost_feat_v1": _xgboost_feat_v1_predict,
}


def main():
    parser = argparse.ArgumentParser(description="Run a baseline predictor")
    parser.add_argument("--split", "-s", default=None, help="Data split (train/val/test)")
    parser.add_argument("--configs", "-c", default=None, help="Path to configs JSON (alternative to --split)")
    parser.add_argument("--method", "-m", default="chinchilla", choices=METHODS.keys(), help="Baseline method")
    parser.add_argument("--output", "-o", required=True, help="Output predictions JSON")
    args = parser.parse_args()

    if args.split:
        configs, _ = load_split(args.split)
    elif args.configs:
        configs = load_configs(args.configs)
    else:
        parser.error("Either --split or --configs is required")

    print(f"Loaded {len(configs)} configs, running {args.method} baseline...")

    predict_fn = METHODS[args.method]
    predictions = predict_fn(configs)

    save_predictions(predictions, args.output)
    print(f"Saved {len(predictions)} predictions to {args.output}")

    for p in predictions:
        print(f"  {p.run_id}: final_loss={p.final_loss:.4f}")


if __name__ == "__main__":
    main()
