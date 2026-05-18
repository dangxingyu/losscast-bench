"""
XGBoost v2 — same architecture as ``xgboost_baseline`` but with an extended
feature set: interaction terms, ratio features, power-law combinations, and
an approximate LR-integral schedule feature.

The v1 model in ``xgboost_baseline.py`` remains the reference — v2 is a
feature-engineering ablation showing how much headroom hand-crafted features
buy over the baseline feature set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..schema import RunConfig, RunGroundTruth, RunPrediction
from .chinchilla import chinchilla_loss
from .xgboost_baseline import (
    CATEGORICAL_FIELDS,
    XGBoostPredictor,
    _bool,
    _num,
    _numeric_config_features,
    _safe_log,
    _step_features,
)


# ── Extra, hand-crafted features layered on top of the v1 feature set ───────

def _approx_lr_integral(config: RunConfig) -> float:
    """Approximate ∫ lr(t) dt over training, as a fraction of peak_lr × total_steps.

    Crude piecewise-linear approximation regardless of the declared schedule:
      - warmup phase (frac = warmup_ratio): avg lr = 0.5 * peak_lr (linear ramp up)
      - stable phase: avg lr = peak_lr
      - cooldown phase (frac = cooldown_ratio): avg lr = 0.5 * peak_lr * (1 + final_lr_ratio)
      - any remainder gets min_lr_ratio * peak_lr as the floor.
    """
    s = config.schedule
    total_steps = max(int(s.total_steps or 1), 1)
    warmup = max(int(s.warmup_steps or 0), 0) / total_steps
    cooldown = max(int(s.cooldown_steps or 0), 0) / total_steps
    stable = max(1.0 - warmup - cooldown, 0.0)
    final_frac = _num(s.final_lr_ratio, 0.1)

    # Fraction of peak LR integrated over training.
    frac_of_peak = (
        0.5 * warmup
        + stable
        + 0.5 * (1.0 + final_frac) * cooldown
    )
    return frac_of_peak


EXTRA_NUMERIC_FEATURE_NAMES: tuple[str, ...] = (
    # Interactions
    "d_model_x_n_layers",
    "log_lr_x_log_batch",
    "log_compute_budget",        # log(N * D)
    # Ratios / shape
    "log_n_over_log_d",
    "gqa_efficiency",            # n_heads * head_dim / d_model
    "d_ff_x_n_layers",
    # Schedule integrals
    "lr_integral_frac",          # fraction of peak LR integrated over training
    "log_lr_integral",           # log of approximate total LR budget
    "cooldown_x_final_lr",       # measures how much annealing actually happens
    "effective_peak_frac",       # 1 - warmup_ratio - cooldown_ratio (stable fraction)
    # Step-varying extras
    "step_lr_phase",             # 0 (warmup), 1 (stable), 2 (cooldown) — coarse
)

FEATURE_NAMES_V2: tuple[str, ...] = (
    # Inherit v1 numeric features (defined in xgboost_baseline)
    # plus the extras below
    "n_layers", "d_model", "n_heads", "head_dim", "d_ff", "d_ff_ratio",
    "vocab_size", "n_kv_heads", "gqa_ratio", "log_n_params",
    "tied_embeddings", "rope",
    "log_lr", "weight_decay", "beta1", "beta2", "log_eps",
    "grad_clip", "has_grad_clip", "mup",
    "log_warmup_steps", "log_total_steps", "warmup_ratio",
    "cooldown_ratio", "final_lr_ratio",
    "seq_len", "log_batch_tokens", "log_tokens_total", "log_tokens_per_param",
    "dp_size", "tp_size", "log_eval_interval",
    "step", "step_frac", "log_step", "log_tokens_seen",
    "chinchilla_pred", "is_init",
) + EXTRA_NUMERIC_FEATURE_NAMES + tuple(f"{c}_code" for c in CATEGORICAL_FIELDS)


def _extra_config_features(config: RunConfig) -> dict[str, float]:
    m = config.model
    o = config.optimizer
    s = config.schedule
    d = config.data

    n_heads = m.n_heads or 1
    d_model = max(int(m.d_model), 1)
    head_dim = m.head_dim or (d_model // n_heads)
    d_ff = m.d_ff or 4 * d_model
    n_params = max(m.n_params_approx, 1)
    tokens_total = _num(d.tokens_total, 1.0)
    batch_tokens = _num(d.batch_tokens, 1.0)

    log_n = _safe_log(n_params)
    log_d = _safe_log(tokens_total)
    log_lr = _safe_log(o.lr)
    log_batch = _safe_log(batch_tokens)

    total_steps = max(int(s.total_steps or 1), 1)
    warmup_ratio = max(int(s.warmup_steps or 0), 0) / total_steps
    cooldown_ratio = max(int(s.cooldown_steps or 0), 0) / total_steps
    final_lr_ratio = _num(s.final_lr_ratio, 0.1)

    lr_frac = _approx_lr_integral(config)
    lr_integral_abs = float(o.lr or 0.0) * lr_frac * total_steps

    return {
        "d_model_x_n_layers": float(d_model) * float(m.n_layers),
        "log_lr_x_log_batch": log_lr * log_batch,
        "log_compute_budget": log_n + log_d,
        "log_n_over_log_d": log_n / log_d if log_d != 0 else 0.0,
        "gqa_efficiency": float(n_heads) * float(head_dim) / float(d_model),
        "d_ff_x_n_layers": float(d_ff) * float(m.n_layers),
        "lr_integral_frac": lr_frac,
        "log_lr_integral": _safe_log(lr_integral_abs),
        "cooldown_x_final_lr": cooldown_ratio * final_lr_ratio,
        "effective_peak_frac": max(1.0 - warmup_ratio - cooldown_ratio, 0.0),
    }


def _step_phase(config: RunConfig, step: int) -> float:
    """Coarse phase indicator: 0=warmup, 1=stable, 2=cooldown."""
    s = config.schedule
    total_steps = max(int(s.total_steps or 1), 1)
    warmup_end = max(int(s.warmup_steps or 0), 0)
    cooldown_start = total_steps - max(int(s.cooldown_steps or 0), 0)
    if step < warmup_end:
        return 0.0
    if step >= cooldown_start:
        return 2.0
    return 1.0


# ── Subclass with extended feature row ──────────────────────────────────────


@dataclass
class XGBoostPredictorV2(XGBoostPredictor):
    """XGBoost with the v1 feature set plus extra interactions and schedule features."""

    def _row(self, config: RunConfig, step: int, fit: bool) -> list[float]:
        num = _numeric_config_features(config)
        num.update(_step_features(config, step))
        num.update(_extra_config_features(config))
        num["step_lr_phase"] = _step_phase(config, step)

        vec = [num[name] for name in FEATURE_NAMES_V2 if name in num]
        # (Every feature except categorical codes is present in ``num``; the
        # categorical ones are handled by the parent class.)
        vec.extend(self._encode_categorical(config, fit=fit))
        return vec


def predict_batch(
    configs: list[RunConfig],
    train_configs: Optional[list[RunConfig]] = None,
    train_ground_truths: Optional[list[RunGroundTruth]] = None,
    model_path: Optional[str] = None,
) -> list[RunPrediction]:
    """Convenience wrapper matching the other baselines' interface."""
    from pathlib import Path
    if model_path is not None and Path(model_path).exists():
        predictor = XGBoostPredictorV2.load(model_path)
    elif train_configs is not None and train_ground_truths is not None:
        predictor = XGBoostPredictorV2().fit(train_configs, train_ground_truths)
        if model_path is not None:
            predictor.save(model_path)
    else:
        raise ValueError(
            "xgboost_v2 baseline needs either a fitted model (model_path) "
            "or training data (train_configs, train_ground_truths)."
        )
    return predictor.predict_batch(configs)
