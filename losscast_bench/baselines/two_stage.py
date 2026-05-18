"""
Two-stage baseline: predict scale (final loss) and trajectory (normalized
curve shape) separately, then combine.

Stage 1 — `final_model` (XGBoost v2): takes run-level features, predicts the
final loss as a scalar.

Stage 2 — `shape_model` (XGBoost v2): takes (run-level features, step), predicts
the shape ratio ``loss[t] / final_loss``. Trained with the *true* final loss
from each training curve.

At inference: ``loss[t] = predicted_final_loss × predicted_shape[t]``.

This decoupling is motivated by the observation that absolute loss magnitude
(dominated by param count + token count + eval_dataset) and curve trajectory
(dominated by schedule + optimizer dynamics) are governed by largely
disjoint subsets of the config.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..schema import RunConfig, RunGroundTruth, RunPrediction
from .xgboost_v2 import XGBoostPredictorV2
from .pca_xgboost import _run_level_row


@dataclass
class TwoStagePredictor:
    """Final-loss scale + normalized curve-shape factorization."""

    n_estimators: int = 500
    max_depth: int = 7
    learning_rate: float = 0.05

    _final_booster: object = field(default=None, init=False, repr=False)
    _shape: Optional[XGBoostPredictorV2] = field(default=None, init=False, repr=False)
    _final_cat_vocab: dict = field(default_factory=dict, init=False, repr=False)

    # ── Stage 1: final-loss regressor (run-level features only) ───────────

    def _final_X(self, configs: list[RunConfig], fit: bool) -> np.ndarray:
        return np.asarray(
            [_run_level_row(c, self._final_cat_vocab, fit=fit) for c in configs],
            dtype=np.float32,
        )

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        configs: list[RunConfig],
        ground_truths: list[RunGroundTruth],
        verbose: bool = False,
    ) -> "TwoStagePredictor":
        import xgboost as xgb

        gt_map = {g.run_id: g for g in ground_truths}
        kept_cfgs: list[RunConfig] = []
        final_losses: list[float] = []
        shape_gts: list[RunGroundTruth] = []

        for cfg in configs:
            gt = gt_map.get(cfg.run_id)
            if gt is None or not gt.losses:
                continue
            final_step = max(gt.losses.keys())
            final_loss = gt.losses[final_step]
            if final_loss is None or not math.isfinite(final_loss) or final_loss <= 0:
                continue
            kept_cfgs.append(cfg)
            final_losses.append(final_loss)
            # Shape GT: loss[t] / final_loss at every step except the final,
            # so the shape model doesn't trivially learn "output 1".
            # Actually keeping all points including final helps, since shape=1
            # is a valid regression target (anchors the curve).
            shape_gts.append(
                RunGroundTruth(
                    run_id=cfg.run_id,
                    losses={s: l / final_loss for s, l in gt.losses.items()
                            if l is not None and math.isfinite(l)},
                )
            )

        if not kept_cfgs:
            raise ValueError("No usable training runs for two_stage baseline.")

        # Stage 1: fit final-loss regressor
        X_final = self._final_X(kept_cfgs, fit=True)
        y_final = np.asarray(final_losses, dtype=np.float32)
        self._final_booster = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            tree_method="hist",
        )
        self._final_booster.fit(X_final, y_final)
        if verbose:
            tr = self._final_booster.predict(X_final)
            print(f"  final-loss stage train MAE: {np.abs(tr - y_final).mean():.4f}")

        # Stage 2: fit curve-shape XGBoost (per-(run, step))
        self._shape = XGBoostPredictorV2(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
        )
        self._shape.fit(kept_cfgs, shape_gts)
        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_batch(self, configs: list[RunConfig]) -> list[RunPrediction]:
        if self._final_booster is None or self._shape is None:
            raise RuntimeError("Predictor not fitted.")

        # Stage 1
        X_final = self._final_X(configs, fit=False)
        final_pred = self._final_booster.predict(X_final)

        # Stage 2
        shape_preds = self._shape.predict_batch(configs)

        out: list[RunPrediction] = []
        for cfg, final_loss, shape in zip(configs, final_pred, shape_preds):
            fl = max(float(final_loss), 1e-6)
            predictions = {s: fl * v for s, v in shape.predictions.items()}
            out.append(RunPrediction(run_id=cfg.run_id, predictions=predictions))
        return out

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "final": {
                "booster": self._final_booster,
                "cat_vocab": self._final_cat_vocab,
            },
            "shape": {
                "booster": self._shape._booster,
                "cat_vocab": self._shape._cat_vocab,
                "config": self._shape._config(),
            },
            "hparams": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f)

    @classmethod
    def load(cls, path: str | Path) -> "TwoStagePredictor":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        inst = cls(**blob["hparams"])
        inst._final_booster = blob["final"]["booster"]
        inst._final_cat_vocab = blob["final"]["cat_vocab"]
        shape = XGBoostPredictorV2(**blob["shape"]["config"])
        shape._booster = blob["shape"]["booster"]
        shape._cat_vocab = blob["shape"]["cat_vocab"]
        inst._shape = shape
        return inst


def predict_batch(
    configs: list[RunConfig],
    train_configs: Optional[list[RunConfig]] = None,
    train_ground_truths: Optional[list[RunGroundTruth]] = None,
    model_path: Optional[str] = None,
) -> list[RunPrediction]:
    if model_path is not None and Path(model_path).exists():
        predictor = TwoStagePredictor.load(model_path)
    elif train_configs is not None and train_ground_truths is not None:
        predictor = TwoStagePredictor().fit(train_configs, train_ground_truths)
        if model_path is not None:
            predictor.save(model_path)
    else:
        raise ValueError("two_stage baseline needs a saved model or training data.")
    return predictor.predict_batch(configs)
