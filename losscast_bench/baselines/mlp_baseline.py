"""
Small MLP baseline for LossCast-Bench (PyTorch, CPU-only).

Same (run, step) row format as the XGBoost v2 baseline, but:
  - categorical fields are **one-hot** encoded instead of label-encoded (MLPs
    handle ordinals very poorly compared to trees)
  - numeric features are standardized (mean=0, std=1) using statistics from
    the training split
  - the model is a 3-layer MLP (256 → 256 → 128) with GELU activations and
    dropout, trained with AdamW and early stopping on a 10 %% holdout of the
    training split (so the benchmark val is never touched during fitting)

This is the neural equivalent of the XGBoost baseline — no code-layer input,
just the flattened RunConfig + step, trained to regress the loss directly.
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
from .xgboost_baseline import (
    CATEGORICAL_FIELDS,
    _categoricals,
    _numeric_config_features,
    _step_features,
)
from .xgboost_v2 import (
    FEATURE_NAMES_V2,
    _extra_config_features,
    _step_phase,
)

# Numeric features = everything in FEATURE_NAMES_V2 minus the categorical codes
_NUMERIC_NAMES = tuple(n for n in FEATURE_NAMES_V2 if not n.endswith("_code"))


def _numeric_row(config: RunConfig, step: int) -> list[float]:
    num = _numeric_config_features(config)
    num.update(_step_features(config, step))
    num.update(_extra_config_features(config))
    num["step_lr_phase"] = _step_phase(config, step)
    return [num[name] for name in _NUMERIC_NAMES]


@dataclass
class _Standardizer:
    mean: np.ndarray = field(default=None)
    std: np.ndarray = field(default=None)

    def fit(self, X: np.ndarray) -> "_Standardizer":
        self.mean = X.mean(axis=0).astype(np.float32)
        std = X.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        self.std = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)


@dataclass
class MLPPredictor:
    """Small PyTorch MLP regressing loss from (run, step) features."""

    hidden_dims: tuple[int, ...] = (256, 256, 128)
    dropout: float = 0.1
    lr: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 200
    patience: int = 12
    val_frac: float = 0.1
    seed: int = 42

    # Fitted state
    _model: object = field(default=None, init=False, repr=False)
    _standardizer: _Standardizer = field(default_factory=_Standardizer, init=False, repr=False)
    _cat_vocab: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)

    # ── Feature extraction ────────────────────────────────────────────────

    def _onehot(self, config: RunConfig) -> list[float]:
        out: list[float] = []
        cats = _categoricals(config)
        for key in CATEGORICAL_FIELDS:
            vocab = self._cat_vocab.get(key, [])
            vec = [0.0] * len(vocab)
            val = cats[key]
            if val in vocab:
                vec[vocab.index(val)] = 1.0
            out.extend(vec)
        return out

    def _rows(
        self,
        configs: list[RunConfig],
        gt_map: Optional[dict[str, RunGroundTruth]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], list[tuple[str, int]]]:
        feats: list[list[float]] = []
        targets: list[float] = []
        index: list[tuple[str, int]] = []

        for cfg in configs:
            if gt_map is not None:
                gt = gt_map.get(cfg.run_id)
                if gt is None:
                    continue
                eval_steps = sorted(gt.losses.keys())
            else:
                eval_steps = cfg.eval_steps

            cat_vec = self._onehot(cfg)
            for step in eval_steps:
                row = _numeric_row(cfg, step) + cat_vec
                feats.append(row)
                index.append((cfg.run_id, step))
                if gt_map is not None:
                    targets.append(gt_map[cfg.run_id].losses[step])

        X = np.asarray(feats, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32) if gt_map is not None else None
        return X, y, index

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        configs: list[RunConfig],
        ground_truths: list[RunGroundTruth],
        verbose: bool = False,
    ) -> "MLPPredictor":
        import torch
        import torch.nn as nn

        # Build categorical vocabulary from training data
        self._cat_vocab = {k: [] for k in CATEGORICAL_FIELDS}
        for cfg in configs:
            for k, v in _categoricals(cfg).items():
                if v not in self._cat_vocab[k]:
                    self._cat_vocab[k].append(v)

        gt_map = {g.run_id: g for g in ground_truths}
        X, y, _ = self._rows(configs, gt_map)
        if len(X) == 0:
            raise ValueError("No training rows.")

        # Standardize numeric columns only (the one-hot columns stay 0/1).
        n_numeric = len(_NUMERIC_NAMES)
        self._standardizer.fit(X[:, :n_numeric])
        X[:, :n_numeric] = self._standardizer.transform(X[:, :n_numeric])

        # Hold out val_frac of the rows for early stopping.
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(X))
        n_val = int(len(X) * self.val_frac)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        torch.manual_seed(self.seed)
        in_dim = X.shape[1]
        layers: list = []
        prev = in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(self.dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        model = nn.Sequential(*layers)

        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.SmoothL1Loss(beta=0.05)

        X_tr_t = torch.from_numpy(X_tr)
        y_tr_t = torch.from_numpy(y_tr).unsqueeze(1)
        X_va_t = torch.from_numpy(X_va)
        y_va_t = torch.from_numpy(y_va).unsqueeze(1)

        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        n_tr = len(X_tr_t)

        for epoch in range(self.max_epochs):
            model.train()
            idx = torch.randperm(n_tr)
            total = 0.0
            for start in range(0, n_tr, self.batch_size):
                bi = idx[start:start + self.batch_size]
                opt.zero_grad()
                out = model(X_tr_t[bi])
                loss = loss_fn(out, y_tr_t[bi])
                loss.backward()
                opt.step()
                total += float(loss) * len(bi)
            tr_loss = total / n_tr

            model.eval()
            with torch.no_grad():
                va_pred = model(X_va_t)
                va_mse = float(torch.nn.functional.mse_loss(va_pred, y_va_t))

            if va_mse < best_val - 1e-6:
                best_val = va_mse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            if verbose and (epoch < 5 or (epoch + 1) % 10 == 0):
                print(f"  epoch {epoch+1:3d}  train_loss={tr_loss:.5f}  val_mse={va_mse:.5f}  best={best_val:.5f}")

            if bad_epochs >= self.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch+1} (best val_mse={best_val:.5f})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model
        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_batch(self, configs: list[RunConfig]) -> list[RunPrediction]:
        if self._model is None:
            raise RuntimeError("Predictor not fitted.")
        import torch

        X, _, index = self._rows(configs, gt_map=None)
        n_numeric = len(_NUMERIC_NAMES)
        X[:, :n_numeric] = self._standardizer.transform(X[:, :n_numeric])

        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.from_numpy(X)).squeeze(-1).numpy()

        by_run: dict[str, dict[int, float]] = {}
        for (run_id, step), p in zip(index, preds):
            by_run.setdefault(run_id, {})[int(step)] = float(p)

        return [
            RunPrediction(run_id=c.run_id, predictions=by_run.get(c.run_id, {}))
            for c in configs
        ]

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._model.state_dict(),
            "architecture": {
                "hidden_dims": list(self.hidden_dims),
                "dropout": self.dropout,
            },
            "standardizer": {
                "mean": self._standardizer.mean,
                "std": self._standardizer.std,
            },
            "cat_vocab": self._cat_vocab,
            "input_dim": next(iter(self._model.parameters())).shape[1],
        }, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "MLPPredictor":
        import torch
        import torch.nn as nn
        blob = torch.load(str(path), weights_only=False)
        inst = cls(
            hidden_dims=tuple(blob["architecture"]["hidden_dims"]),
            dropout=blob["architecture"]["dropout"],
        )
        # Rebuild the model from the saved architecture + input_dim
        layers = []
        prev = blob["input_dim"]
        for h in inst.hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(inst.dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        model = nn.Sequential(*layers)
        model.load_state_dict(blob["state_dict"])
        inst._model = model
        inst._standardizer.mean = blob["standardizer"]["mean"]
        inst._standardizer.std = blob["standardizer"]["std"]
        inst._cat_vocab = blob["cat_vocab"]
        return inst


def predict_batch(
    configs: list[RunConfig],
    train_configs: Optional[list[RunConfig]] = None,
    train_ground_truths: Optional[list[RunGroundTruth]] = None,
    model_path: Optional[str] = None,
) -> list[RunPrediction]:
    if model_path is not None and Path(model_path).exists():
        predictor = MLPPredictor.load(model_path)
    elif train_configs is not None and train_ground_truths is not None:
        predictor = MLPPredictor().fit(train_configs, train_ground_truths)
        if model_path is not None:
            predictor.save(model_path)
    else:
        raise ValueError("mlp baseline needs a saved model or training data.")
    return predictor.predict_batch(configs)
