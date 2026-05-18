"""
PCA + XGBoost baseline: decompose training loss curves into a low-rank basis
and predict the basis coefficients from the run config.

Pipeline (fit):
  1. Resample every training curve to ``K`` evenly-spaced fractional steps
     (0, 1/(K-1), ..., 1) via linear interpolation on the run's own grid.
  2. Log-transform and subtract the step-wise Chinchilla prediction → work
     in the residual space where scaling-law structure is already removed.
  3. PCA on the resulting (N_train × K) matrix, keeping ``n_components``
     principal components.
  4. For each principal coefficient, train an XGBoost regressor on the
     run-level feature vector (no step feature).

Pipeline (predict):
  1. Extract run-level features.
  2. Predict ``n_components`` coefficients with the XGBoost heads.
  3. Inverse-PCA → reconstructed K-point residual curve.
  4. Add Chinchilla prediction back → K-point absolute log-loss curve.
  5. Interpolate to the run's eval_steps, exponentiate.

Run-level features: the v2 numeric + categorical set, but computed with
step=0 (initial point) so the step-varying slots have consistent defaults.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..schema import RunConfig, RunGroundTruth, RunPrediction
from .chinchilla import chinchilla_loss
from .xgboost_v2 import (
    FEATURE_NAMES_V2,
    XGBoostPredictorV2,
    _extra_config_features,
    _step_phase,
)
from .xgboost_baseline import (
    CATEGORICAL_FIELDS,
    _numeric_config_features,
    _safe_log,
    _step_features,
)


def _run_level_row(
    config: RunConfig,
    cat_vocab: dict[str, dict[str, int]],
    fit: bool,
) -> list[float]:
    """Build a feature vector that does NOT depend on the step.

    Uses step=total_steps/2 as a fixed reference point so that all step-varying
    slots (log_tokens_seen, chinchilla_pred, etc.) have consistent, non-zero
    values across runs rather than being silently zeroed out.
    """
    ref_step = max(int((config.schedule.total_steps or 1) // 2), 1)
    num = _numeric_config_features(config)
    num.update(_step_features(config, ref_step))
    num.update(_extra_config_features(config))
    num["step_lr_phase"] = _step_phase(config, ref_step)

    vec = [num[name] for name in FEATURE_NAMES_V2 if name in num]

    # Label-encode categoricals, mirroring XGBoostPredictor._encode_categorical.
    from .xgboost_baseline import _categoricals
    cats = _categoricals(config)
    for key in CATEGORICAL_FIELDS:
        val = cats[key]
        vocab = cat_vocab.setdefault(key, {})
        if fit:
            if val not in vocab:
                vocab[val] = len(vocab)
            vec.append(float(vocab[val]))
        else:
            vec.append(float(vocab.get(val, -1)))
    return vec


def _resample_curve(
    losses: dict[int, float],
    total_steps: int,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (grid_fracs, resampled_log_losses) on K evenly-spaced fracs.

    Linear interpolation in log-space; values outside the observed range are
    held at the nearest endpoint.
    """
    if total_steps <= 0:
        total_steps = 1
    steps = sorted(losses.keys())
    xs = np.array([s / total_steps for s in steps], dtype=np.float64)
    ys = np.log(np.array([max(losses[s], 1e-6) for s in steps], dtype=np.float64))
    grid = np.linspace(0.0, 1.0, K)
    resampled = np.interp(grid, xs, ys)
    return grid, resampled


def _chinchilla_on_grid(config: RunConfig, grid: np.ndarray) -> np.ndarray:
    """Chinchilla prediction evaluated on a fractional grid (length K)."""
    n_params = max(config.model.n_params_approx, 1)
    tokens_total = float(config.data.tokens_total or 0.0)
    vocab = max(int(config.model.vocab_size or 2), 2)
    out = np.empty(len(grid))
    for i, frac in enumerate(grid):
        if frac <= 0.0:
            out[i] = math.log(vocab)
        else:
            tokens = max(frac * tokens_total, 1.0)
            out[i] = chinchilla_loss(n_params, tokens)
    return np.log(np.clip(out, 1e-6, None))


@dataclass
class PCAXGBoostPredictor:
    """PCA over log-residual curves + XGBoost regression on PCA coefficients."""

    K: int = 16
    n_components: int = 8
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05

    # Fitted state
    _pca: object = field(default=None, init=False, repr=False)
    _heads: list = field(default_factory=list, init=False, repr=False)
    _cat_vocab: dict[str, dict[str, int]] = field(default_factory=dict, init=False, repr=False)
    _grid: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # ── Feature / target matrices ─────────────────────────────────────────

    def _build_matrix(
        self,
        configs: list[RunConfig],
        ground_truths: Optional[list[RunGroundTruth]],
        fit: bool,
    ) -> tuple[np.ndarray, Optional[np.ndarray], list[RunConfig]]:
        """Build run-level feature matrix X and (if fitting) residual matrix Y.

        Runs with too few loss points (<3) are skipped during training but
        kept during prediction (features only).
        """
        gt_map = {g.run_id: g for g in ground_truths} if ground_truths else None

        feats: list[list[float]] = []
        residuals: list[np.ndarray] = []
        kept: list[RunConfig] = []

        for cfg in configs:
            if gt_map is not None:
                gt = gt_map.get(cfg.run_id)
                if gt is None or len(gt.losses) < 3:
                    continue
                grid, log_curve = _resample_curve(
                    gt.losses, cfg.schedule.total_steps or 1, self.K,
                )
                chin_log = _chinchilla_on_grid(cfg, grid)
                residuals.append(log_curve - chin_log)

            feats.append(_run_level_row(cfg, self._cat_vocab, fit=fit))
            kept.append(cfg)
            if self._grid is None:
                # Shared grid used by every subsequent call.
                self._grid = np.linspace(0.0, 1.0, self.K)

        X = np.asarray(feats, dtype=np.float32)
        Y = np.asarray(residuals, dtype=np.float32) if residuals else None
        return X, Y, kept

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        configs: list[RunConfig],
        ground_truths: list[RunGroundTruth],
        verbose: bool = False,
    ) -> "PCAXGBoostPredictor":
        import xgboost as xgb
        from sklearn.decomposition import PCA

        X, Y, _ = self._build_matrix(configs, ground_truths, fit=True)
        if Y is None or len(Y) == 0:
            raise ValueError("No usable training curves (need ≥3 points per run).")

        n_comp = min(self.n_components, min(Y.shape) - 1)
        self._pca = PCA(n_components=n_comp)
        coeffs = self._pca.fit_transform(Y).astype(np.float32)  # (N, n_comp)
        if verbose:
            print(f"  PCA explained variance ratios: {self._pca.explained_variance_ratio_.round(3)}")

        # We need a feature row per training curve, aligned with Y. Rebuild so
        # we only include rows whose ground truth was actually kept.
        # ``_build_matrix`` already returned ``X`` with one row per training
        # curve *that was also kept as a target*; this holds because we append
        # to ``feats`` only when the GT check passed.
        assert X.shape[0] == Y.shape[0], (X.shape, Y.shape)

        self._heads = []
        for j in range(n_comp):
            head = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=5.0,
                reg_lambda=1.0,
                random_state=42 + j,
                n_jobs=-1,
                objective="reg:squarederror",
                tree_method="hist",
            )
            head.fit(X, coeffs[:, j])
            self._heads.append(head)
        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_batch(self, configs: list[RunConfig]) -> list[RunPrediction]:
        if self._pca is None or not self._heads:
            raise RuntimeError("Predictor not fitted.")

        X, _, kept = self._build_matrix(configs, ground_truths=None, fit=False)
        # Predict PCA coefficients → reconstruct residual curves
        coeffs = np.stack([h.predict(X) for h in self._heads], axis=1)
        residual_grid = self._pca.inverse_transform(coeffs)  # (N, K)

        grid = self._grid if self._grid is not None else np.linspace(0.0, 1.0, self.K)

        out: list[RunPrediction] = []
        for cfg, resid in zip(kept, residual_grid):
            chin_log = _chinchilla_on_grid(cfg, grid)
            pred_log = chin_log + resid
            predictions: dict[int, float] = {}
            total_steps = max(int(cfg.schedule.total_steps or 1), 1)
            vocab = max(int(cfg.model.vocab_size or 2), 2)
            for step in cfg.eval_steps:
                if step == 0:
                    predictions[step] = math.log(vocab)
                    continue
                frac = step / total_steps
                # Linear interp in log-space on our K-point grid
                v = float(np.interp(frac, grid, pred_log))
                predictions[step] = math.exp(v)
            out.append(RunPrediction(run_id=cfg.run_id, predictions=predictions))
        return out

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "pca": self._pca,
                "heads": self._heads,
                "cat_vocab": self._cat_vocab,
                "grid": self._grid,
                "hparams": {
                    "K": self.K,
                    "n_components": self.n_components,
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                },
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "PCAXGBoostPredictor":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        inst = cls(**blob["hparams"])
        inst._pca = blob["pca"]
        inst._heads = blob["heads"]
        inst._cat_vocab = blob["cat_vocab"]
        inst._grid = blob["grid"]
        return inst


def predict_batch(
    configs: list[RunConfig],
    train_configs: Optional[list[RunConfig]] = None,
    train_ground_truths: Optional[list[RunGroundTruth]] = None,
    model_path: Optional[str] = None,
) -> list[RunPrediction]:
    if model_path is not None and Path(model_path).exists():
        predictor = PCAXGBoostPredictor.load(model_path)
    elif train_configs is not None and train_ground_truths is not None:
        predictor = PCAXGBoostPredictor().fit(train_configs, train_ground_truths)
        if model_path is not None:
            predictor.save(model_path)
    else:
        raise ValueError("pca_xgboost baseline needs a saved model or training data.")
    return predictor.predict_batch(configs)
