"""
Chinchilla scaling law with coefficients refit on the benchmark's own train
split. Same functional form as Hoffmann et al. (2022):

    L(N, D) = E + A / N^alpha + B / D^beta

where N = non-embedding params, D = tokens seen so far.

The coefficients (E, A, B, alpha, beta) are fit to the training data's full
loss curves (one point per (run, eval_step)) by minimizing MSE in log-space,
which is more numerically stable than direct MSE on losses.

Rationale: Hoffmann's published coefficients were fit to DeepMind's data with
their tokenizer and eval protocol. Marin (C4 English eval) and StepLaw
(smoothed pretraining loss) sit on different loss scales, so the canonical
coefficients bake in a ~0.5-nat offset on StepLaw alone. Refitting closes
that gap while keeping the baseline strictly configuration-agnostic — only
(N, D) are used, so the model remains blind to optimizer, schedule, and
architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..schema import RunConfig, RunGroundTruth, RunPrediction


@dataclass
class ChinchillaFit:
    """Parametric fit for L(N, D) = E + A/N^alpha + B/D^beta."""

    E: float = 1.69
    A: float = 406.4
    B: float = 410.7
    alpha: float = 0.34
    beta: float = 0.28

    def predict(self, n_params: float, n_tokens: float) -> float:
        n_params = max(n_params, 1.0)
        n_tokens = max(n_tokens, 1.0)
        return self.E + self.A / (n_params ** self.alpha) + self.B / (n_tokens ** self.beta)

    def to_dict(self) -> dict:
        return {"E": self.E, "A": self.A, "B": self.B, "alpha": self.alpha, "beta": self.beta}


def _collect_fit_points(
    configs: list[RunConfig],
    ground_truths: list[RunGroundTruth],
) -> tuple[list[float], list[float], list[float]]:
    """Flatten (run, step) rows into parallel (N, D, loss) lists.

    Step 0 is excluded — Chinchilla's functional form is undefined at D=0 and
    the initial cross-entropy (≈log vocab_size) isn't meaningful to fit.
    """
    gt_map = {g.run_id: g for g in ground_truths}
    Ns: list[float] = []
    Ds: list[float] = []
    Ls: list[float] = []

    for cfg in configs:
        gt = gt_map.get(cfg.run_id)
        if gt is None:
            continue

        n_params = max(cfg.model.n_params_approx, 1)
        total_steps = max(cfg.schedule.total_steps or 1, 1)
        tokens_total = float(cfg.data.tokens_total or 0.0)
        if tokens_total <= 0:
            continue

        for step, loss in gt.losses.items():
            if step == 0 or loss is None:
                continue
            if not math.isfinite(loss):
                continue
            frac = step / total_steps
            tokens = max(frac * tokens_total, 1.0)
            Ns.append(n_params)
            Ds.append(tokens)
            Ls.append(float(loss))

    return Ns, Ds, Ls


def fit_chinchilla(
    configs: list[RunConfig],
    ground_truths: list[RunGroundTruth],
    init: Optional[ChinchillaFit] = None,
    alpha_bounds: tuple[float, float] = (0.05, 1.5),
    beta_bounds: tuple[float, float] = (0.05, 1.5),
    e_margin: float = 0.05,
) -> ChinchillaFit:
    """Refit (E, A, B, alpha, beta) on the given runs.

    Loss function: MSE in loss-space on all (run, step) points with step > 0.
    We constrain α, β to a meaningful range and E to ``[0, min(L) - e_margin]``
    so the fit cannot degenerate into two cancelling large terms (which can
    happen when mixing sources with different irreducible loss floors). The
    bounds still leave plenty of room around typical LLM scaling-law values
    (α≈0.2–0.5, β≈0.2–0.4).
    """
    try:
        import numpy as np
        from scipy.optimize import least_squares
    except ImportError as e:
        raise RuntimeError(
            "chinchilla_refit requires numpy + scipy. "
            "Install with: pip install numpy scipy"
        ) from e

    Ns, Ds, Ls = _collect_fit_points(configs, ground_truths)
    if not Ns:
        raise ValueError("No (run, step) points available to fit Chinchilla.")

    N = np.asarray(Ns, dtype=np.float64)
    D = np.asarray(Ds, dtype=np.float64)
    L = np.asarray(Ls, dtype=np.float64)

    # E must stay below observed losses, above zero (cross-entropy is nonneg).
    e_upper = max(float(L.min()) - e_margin, 0.0)
    init = init or ChinchillaFit()
    E0 = min(max(init.E, 0.0), e_upper) if e_upper > 0 else 0.0

    # Parameters: (E, A, B, alpha, beta). Bounds keep the fit interpretable.
    x0 = np.array([
        E0,
        max(init.A, 1.0),
        max(init.B, 1.0),
        min(max(init.alpha, alpha_bounds[0]), alpha_bounds[1]),
        min(max(init.beta, beta_bounds[0]), beta_bounds[1]),
    ], dtype=np.float64)

    lower = np.array([0.0, 1e-3, 1e-3, alpha_bounds[0], beta_bounds[0]])
    upper = np.array([max(e_upper, 1e-6), 1e6, 1e6, alpha_bounds[1], beta_bounds[1]])

    def _residuals(x):
        E, A, B, alpha, beta = x
        pred = E + A * np.power(N, -alpha) + B * np.power(D, -beta)
        return pred - L

    res = least_squares(
        _residuals, x0, bounds=(lower, upper),
        method="trf", xtol=1e-10, ftol=1e-10, max_nfev=20000,
    )

    E, A, B, alpha, beta = res.x
    return ChinchillaFit(
        E=float(E), A=float(A), B=float(B),
        alpha=float(alpha), beta=float(beta),
    )


@dataclass
class ChinchillaRefitPredictor:
    """Chinchilla scaling law with coefficients fit to the benchmark's train."""

    fit: Optional[ChinchillaFit] = None

    def fit_from(
        self,
        configs: list[RunConfig],
        ground_truths: list[RunGroundTruth],
    ) -> "ChinchillaRefitPredictor":
        self.fit = fit_chinchilla(configs, ground_truths)
        return self

    def predict_run(self, config: RunConfig) -> RunPrediction:
        if self.fit is None:
            raise RuntimeError("Predictor has no fitted coefficients.")
        total_steps = max(config.schedule.total_steps or 1, 1)
        tokens_total = float(config.data.tokens_total or 0.0)
        n_params = max(config.model.n_params_approx, 1)
        vocab = max(int(config.model.vocab_size or 2), 2)

        predictions: dict[int, float] = {}
        for step in config.eval_steps:
            if step == 0:
                # Same convention as baseline Chinchilla: initial CE ≈ log(vocab).
                predictions[step] = math.log(vocab)
                continue
            frac = step / total_steps
            tokens_seen = max(frac * tokens_total, 1.0)
            predictions[step] = self.fit.predict(n_params, tokens_seen)
        return RunPrediction(run_id=config.run_id, predictions=predictions)

    def predict_batch(self, configs: list[RunConfig]) -> list[RunPrediction]:
        return [self.predict_run(c) for c in configs]

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.fit.to_dict() if self.fit else {}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ChinchillaRefitPredictor":
        import json
        with open(path) as f:
            blob = json.load(f)
        return cls(fit=ChinchillaFit(**blob))


def predict_batch(
    configs: list[RunConfig],
    train_configs: Optional[list[RunConfig]] = None,
    train_ground_truths: Optional[list[RunGroundTruth]] = None,
    model_path: Optional[str | Path] = None,
) -> list[RunPrediction]:
    """Convenience wrapper mirroring the interface of the other baselines.

    Either pass a saved fit via ``model_path`` or training data to refit.
    """
    if model_path is not None and Path(model_path).exists():
        predictor = ChinchillaRefitPredictor.load(model_path)
    elif train_configs is not None and train_ground_truths is not None:
        predictor = ChinchillaRefitPredictor().fit_from(train_configs, train_ground_truths)
        if model_path is not None:
            predictor.save(model_path)
    else:
        raise ValueError(
            "chinchilla_refit baseline needs either a saved fit (model_path) "
            "or training data (train_configs, train_ground_truths)."
        )
    return predictor.predict_batch(configs)
