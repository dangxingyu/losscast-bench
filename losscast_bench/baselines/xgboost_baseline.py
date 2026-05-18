"""
XGBoost baseline for LossCast-Bench.

Flattens a RunConfig + step into a numeric feature vector and trains a gradient
boosted tree to predict the loss at that step.

Key design choices:
- One training row per (run, eval_step). Step is an input feature, so a single
  model predicts the whole curve by varying step.
- The Chinchilla prediction at each step is included as a feature, so the model
  effectively learns the residual over scaling laws (same trick NCPL uses).
- Categorical fields (optimizer name, activation, schedule, tokenizer,
  eval_dataset) are label-encoded; XGBoost handles them as numeric ordinals
  when enable_categorical is not used — for tree models this is fine because
  the classes are few and splits can approximate one-hot behavior.

Usage:
    from losscast_bench.baselines.xgboost_baseline import XGBoostPredictor
    model = XGBoostPredictor().fit(train_configs, train_ground_truths)
    predictions = model.predict_batch(val_configs)
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


# Categorical fields and the vocabularies we learn during fit().
CATEGORICAL_FIELDS = (
    "optimizer_name",
    "activation",
    "norm_type",
    "lr_schedule",
    "eval_dataset",
    "tokenizer",
    "arch",
)


def _safe_log(x, floor: float = 1e-12) -> float:
    if x is None:
        return math.log(floor)
    return math.log(max(float(x), floor))


def _num(x, default: float = 0.0) -> float:
    """Coerce to float, falling back to a default when x is None/NaN."""
    if x is None:
        return float(default)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(v) or math.isinf(v):
        return float(default)
    return v


def _bool(x) -> float:
    return 1.0 if x else 0.0


def _categoricals(config: RunConfig) -> dict[str, str]:
    return {
        "optimizer_name": config.optimizer.name,
        "activation": config.model.activation,
        "norm_type": config.model.norm_type,
        "lr_schedule": config.schedule.lr_schedule,
        "eval_dataset": config.data.eval_dataset,
        "tokenizer": config.data.tokenizer,
        "arch": config.model.arch,
    }


def _numeric_config_features(config: RunConfig) -> dict[str, float]:
    """Fields that don't depend on the step — computed once per run."""
    m = config.model
    o = config.optimizer
    s = config.schedule
    d = config.data

    n_heads = m.n_heads or 1
    d_model = max(int(m.d_model), 1)
    head_dim = m.head_dim or (d_model // n_heads)
    n_kv = m.n_kv_heads or n_heads
    d_ff = m.d_ff or 4 * d_model
    n_params = max(m.n_params_approx, 1)

    total_steps = max(int(s.total_steps or 1), 1)
    warmup_steps = max(int(s.warmup_steps or 0), 0)
    cooldown_steps = max(int(s.cooldown_steps or 0), 0)
    warmup_ratio = warmup_steps / total_steps
    cooldown_ratio = cooldown_steps / total_steps

    tokens_total = _num(d.tokens_total, 1.0)
    tokens_per_param = tokens_total / n_params

    return {
        # Model
        "n_layers": _num(m.n_layers),
        "d_model": float(d_model),
        "n_heads": float(n_heads),
        "head_dim": float(head_dim),
        "d_ff": float(d_ff),
        "d_ff_ratio": float(d_ff) / d_model,
        "vocab_size": _num(m.vocab_size),
        "n_kv_heads": float(n_kv),
        "gqa_ratio": float(n_heads) / max(n_kv, 1),
        "log_n_params": _safe_log(n_params),
        "tied_embeddings": _bool(m.tied_embeddings),
        "rope": _bool(m.rope),

        # Optimizer
        "log_lr": _safe_log(o.lr),
        "weight_decay": _num(o.weight_decay),
        "beta1": _num(o.beta1, 0.9),
        "beta2": _num(o.beta2, 0.95),
        "log_eps": _safe_log(_num(o.eps, 1e-8)),
        "grad_clip": _num(o.grad_clip, 0.0),
        "has_grad_clip": _bool(o.grad_clip is not None),
        "mup": _bool(o.mup),

        # Schedule
        "log_warmup_steps": _safe_log(max(warmup_steps, 1)),
        "log_total_steps": _safe_log(total_steps),
        "warmup_ratio": warmup_ratio,
        "cooldown_ratio": cooldown_ratio,
        "final_lr_ratio": _num(s.final_lr_ratio, 0.1),

        # Data
        "seq_len": _num(d.seq_len),
        "log_batch_tokens": _safe_log(_num(d.batch_tokens, 1.0)),
        "log_tokens_total": _safe_log(tokens_total),
        "log_tokens_per_param": _safe_log(tokens_per_param),

        # Hardware / parallelism
        "dp_size": _num(config.dp_size, 1),
        "tp_size": _num(config.tp_size, 1),
        "log_eval_interval": _safe_log(max(int(config.eval_interval or 1), 1)),
    }


def _step_features(config: RunConfig, step: int) -> dict[str, float]:
    """Features that vary with the eval step."""
    total_steps = max(int(config.schedule.total_steps or 1), 1)
    tokens_total = _num(config.data.tokens_total, 1.0)
    n_params = max(config.model.n_params_approx, 1)
    vocab_size = max(int(config.model.vocab_size or 2), 2)

    step_frac = step / total_steps
    tokens_seen = max(step_frac * tokens_total, 1.0)

    if step == 0:
        chinchilla_pred = math.log(vocab_size)
    else:
        chinchilla_pred = chinchilla_loss(n_params, tokens_seen)

    return {
        "step": float(step),
        "step_frac": float(step_frac),
        "log_step": _safe_log(max(step, 1)),
        "log_tokens_seen": _safe_log(tokens_seen),
        "chinchilla_pred": float(chinchilla_pred),
        "is_init": 1.0 if step == 0 else 0.0,
    }


# Deterministic feature ordering so the fitted booster matches at predict time.
NUMERIC_FEATURE_NAMES: tuple[str, ...] = (
    # Model
    "n_layers", "d_model", "n_heads", "head_dim", "d_ff", "d_ff_ratio",
    "vocab_size", "n_kv_heads", "gqa_ratio", "log_n_params",
    "tied_embeddings", "rope",
    # Optimizer
    "log_lr", "weight_decay", "beta1", "beta2", "log_eps",
    "grad_clip", "has_grad_clip", "mup",
    # Schedule
    "log_warmup_steps", "log_total_steps", "warmup_ratio",
    "cooldown_ratio", "final_lr_ratio",
    # Data
    "seq_len", "log_batch_tokens", "log_tokens_total", "log_tokens_per_param",
    # Hardware / parallelism
    "dp_size", "tp_size", "log_eval_interval",
    # Step-varying
    "step", "step_frac", "log_step", "log_tokens_seen",
    "chinchilla_pred", "is_init",
)

FEATURE_NAMES: tuple[str, ...] = NUMERIC_FEATURE_NAMES + tuple(
    f"{c}_code" for c in CATEGORICAL_FIELDS
)


@dataclass
class XGBoostPredictor:
    """Gradient boosted trees over flattened RunConfig + step features."""

    n_estimators: int = 600
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: float = 5.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1

    # Fitted state
    _booster: object = field(default=None, init=False, repr=False)
    _cat_vocab: dict[str, dict[str, int]] = field(default_factory=dict, init=False, repr=False)

    # ── Feature building ──────────────────────────────────────────────────

    def _encode_categorical(self, config: RunConfig, fit: bool) -> list[float]:
        """Label-encode categorical fields. Unknown values → -1."""
        cats = _categoricals(config)
        out = []
        for key in CATEGORICAL_FIELDS:
            val = cats[key]
            vocab = self._cat_vocab.setdefault(key, {})
            if fit:
                if val not in vocab:
                    vocab[val] = len(vocab)
                out.append(float(vocab[val]))
            else:
                out.append(float(vocab.get(val, -1)))
        return out

    def _row(self, config: RunConfig, step: int, fit: bool) -> list[float]:
        num = _numeric_config_features(config)
        num.update(_step_features(config, step))
        vec = [num[name] for name in NUMERIC_FEATURE_NAMES]
        vec.extend(self._encode_categorical(config, fit=fit))
        return vec

    def _build_matrix(
        self,
        configs: list[RunConfig],
        ground_truths: Optional[list[RunGroundTruth]] = None,
        fit: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray], list[tuple[str, int]]]:
        """Build (X, y, index). index[i] = (run_id, step) for row i.

        If ground_truths is provided, also returns y. Rows are emitted only for
        eval steps that exist in the ground truth (drops steps the benchmark
        didn't record).
        """
        gt_map = {gt.run_id: gt for gt in ground_truths} if ground_truths else None

        rows: list[list[float]] = []
        targets: list[float] = []
        index: list[tuple[str, int]] = []

        for config in configs:
            if gt_map is not None:
                gt = gt_map.get(config.run_id)
                if gt is None:
                    continue
                eval_steps = sorted(gt.losses.keys())
            else:
                eval_steps = config.eval_steps

            for step in eval_steps:
                rows.append(self._row(config, step, fit=fit))
                index.append((config.run_id, step))
                if gt_map is not None:
                    targets.append(gt_map[config.run_id].losses[step])

        X = np.asarray(rows, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32) if gt_map is not None else None
        return X, y, index

    # ── Training / inference ──────────────────────────────────────────────

    def fit(
        self,
        configs: list[RunConfig],
        ground_truths: list[RunGroundTruth],
        val_configs: Optional[list[RunConfig]] = None,
        val_ground_truths: Optional[list[RunGroundTruth]] = None,
        verbose: bool = False,
    ) -> "XGBoostPredictor":
        import xgboost as xgb

        X, y, _ = self._build_matrix(configs, ground_truths, fit=True)

        eval_set = None
        if val_configs is not None and val_ground_truths is not None:
            X_val, y_val, _ = self._build_matrix(val_configs, val_ground_truths, fit=False)
            eval_set = [(X, y), (X_val, y_val)]

        self._booster = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            objective="reg:squarederror",
            tree_method="hist",
        )

        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = verbose

        self._booster.fit(X, y, **fit_kwargs)
        return self

    def predict_batch(self, configs: list[RunConfig]) -> list[RunPrediction]:
        if self._booster is None:
            raise RuntimeError("Predictor not fitted — call .fit() first.")

        X, _, index = self._build_matrix(configs, ground_truths=None, fit=False)
        y_pred = self._booster.predict(X)

        by_run: dict[str, dict[int, float]] = {}
        for (run_id, step), pred in zip(index, y_pred):
            by_run.setdefault(run_id, {})[int(step)] = float(pred)

        return [
            RunPrediction(run_id=c.run_id, predictions=by_run.get(c.run_id, {}))
            for c in configs
        ]

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"booster": self._booster, "cat_vocab": self._cat_vocab, "config": self._config()},
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostPredictor":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        inst = cls(**blob["config"])
        inst._booster = blob["booster"]
        inst._cat_vocab = blob["cat_vocab"]
        return inst

    def _config(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


def predict_batch(
    configs: list[RunConfig],
    train_configs: Optional[list[RunConfig]] = None,
    train_ground_truths: Optional[list[RunGroundTruth]] = None,
    model_path: Optional[str | Path] = None,
) -> list[RunPrediction]:
    """Convenience wrapper matching the Chinchilla baseline's interface.

    Either provide a saved model via ``model_path`` or training data via
    ``train_configs`` + ``train_ground_truths`` (fits a fresh model).
    """
    if model_path is not None and Path(model_path).exists():
        predictor = XGBoostPredictor.load(model_path)
    elif train_configs is not None and train_ground_truths is not None:
        predictor = XGBoostPredictor().fit(train_configs, train_ground_truths)
        if model_path is not None:
            predictor.save(model_path)
    else:
        raise ValueError(
            "xgboost baseline needs either a fitted model (model_path) "
            "or training data (train_configs, train_ground_truths)."
        )
    return predictor.predict_batch(configs)
