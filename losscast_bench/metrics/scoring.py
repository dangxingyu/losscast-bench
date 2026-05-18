"""
Evaluation metrics for LossCast-Bench.

Metrics:
  1. Huber Loss (δ=0.01) — primary ranking metric, following NCPL
  2. R² — coefficient of determination across all (pred, truth) pairs
  3. Extrapolation Error — Huber on runs beyond a compute threshold
  4. Final Loss Huber — Huber on final-step predictions only
  5. Curve MAPE — mean absolute percentage error across all curve points
  6. MAE / RMSE — absolute and squared errors (reported both curve-wide and
     final-only, so results can be compared to NCPL paper Table 1)
  7. Spearman ρ — rank correlation on final-loss predictions
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..schema import RunPrediction, RunGroundTruth, RunConfig


# ── Core metric functions ────────────────────────────────────────────────────

def huber_loss(pred: float, truth: float, delta: float = 0.01) -> float:
    """Huber loss for a single (pred, truth) pair."""
    a = abs(pred - truth)
    if a <= delta:
        return 0.5 * a * a
    return delta * (a - 0.5 * delta)


def compute_r2(preds: list[float], truths: list[float]) -> float:
    """R² (coefficient of determination)."""
    n = len(truths)
    if n == 0:
        return 0.0
    mean_truth = sum(truths) / n
    ss_res = sum((p - t) ** 2 for p, t in zip(preds, truths))
    ss_tot = sum((t - mean_truth) ** 2 for t in truths)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def compute_mape(preds: list[float], truths: list[float]) -> float:
    """Mean absolute percentage error."""
    n = len(truths)
    if n == 0:
        return 0.0
    total = 0.0
    count = 0
    for p, t in zip(preds, truths):
        if abs(t) > 1e-8:
            total += abs(p - t) / abs(t)
            count += 1
    return total / count if count > 0 else 0.0


def compute_mae(preds: list[float], truths: list[float]) -> float:
    """Mean absolute error."""
    n = len(truths)
    if n == 0:
        return 0.0
    return sum(abs(p - t) for p, t in zip(preds, truths)) / n


def compute_rmse(preds: list[float], truths: list[float]) -> float:
    """Root mean squared error."""
    n = len(truths)
    if n == 0:
        return 0.0
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(preds, truths)) / n)


def compute_spearman(preds: list[float], truths: list[float]) -> float:
    """Spearman rank correlation. Returns 0 on empty or degenerate input."""
    n = len(preds)
    if n < 2:
        return 0.0

    def _ranks(xs: list[float]) -> list[float]:
        """Average-rank for ties (standard Spearman)."""
        indexed = sorted(range(n), key=lambda i: xs[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[indexed[j + 1]] == xs[indexed[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1  # 1-indexed average rank
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg
            i = j + 1
        return ranks

    rp = _ranks(list(preds))
    rt = _ranks(list(truths))
    mp = sum(rp) / n
    mt = sum(rt) / n
    num = sum((a - mp) * (b - mt) for a, b in zip(rp, rt))
    dp = math.sqrt(sum((a - mp) ** 2 for a in rp))
    dt = math.sqrt(sum((b - mt) ** 2 for b in rt))
    if dp == 0 or dt == 0:
        return 0.0
    return num / (dp * dt)


# ── Per-run evaluation ───────────────────────────────────────────────────────

@dataclass
class RunResult:
    """Evaluation result for a single run."""
    run_id: str
    huber: float
    final_huber: float
    r2: float
    mape: float
    n_steps: int


@dataclass
class EvalResult:
    """Aggregate evaluation result across all runs."""

    # Primary
    huber: float  # mean Huber across all (run, step) pairs
    r2: float  # R² across all (run, step) pairs

    # Final loss track
    final_huber: float  # Huber on final step only
    final_r2: float

    # Curve track
    curve_mape: float

    # Absolute / squared error (curve-wide and final-only).
    # Reported explicitly so results compare directly to NCPL paper Table 1.
    curve_mae: float
    curve_rmse: float
    final_mae: float
    final_rmse: float
    final_spearman: float  # rank correlation on final-loss predictions

    # Extrapolation (if compute threshold provided)
    extrap_huber: Optional[float]
    extrap_r2: Optional[float]
    extrap_n_runs: int

    # Metadata
    n_runs: int
    n_points: int  # total (run, step) pairs evaluated

    per_run: list[RunResult]

    def summary(self) -> dict:
        """Return a compact summary dict."""
        d = {
            "huber": round(self.huber, 6),
            "r2": round(self.r2, 4),
            "final_huber": round(self.final_huber, 6),
            "final_r2": round(self.final_r2, 4),
            "curve_mape": round(self.curve_mape, 4),
            "curve_mae": round(self.curve_mae, 6),
            "curve_rmse": round(self.curve_rmse, 6),
            "final_mae": round(self.final_mae, 6),
            "final_rmse": round(self.final_rmse, 6),
            "final_spearman": round(self.final_spearman, 4),
            "n_runs": self.n_runs,
            "n_points": self.n_points,
        }
        if self.extrap_huber is not None:
            d["extrap_huber"] = round(self.extrap_huber, 6)
            d["extrap_r2"] = round(self.extrap_r2, 4)
            d["extrap_n_runs"] = self.extrap_n_runs
        return d


def evaluate_run(
    pred: RunPrediction,
    truth: RunGroundTruth,
    delta: float = 0.01,
    expected_steps: Optional[list[int]] = None,
) -> RunResult:
    """Evaluate a single run's predictions against ground truth."""
    if expected_steps is None:
        scoring_steps = sorted(set(pred.predictions.keys()) & set(truth.losses.keys()))
    else:
        scoring_steps = sorted(
            set(expected_steps) & set(pred.predictions.keys()) & set(truth.losses.keys())
        )
    if not scoring_steps:
        return RunResult(
            run_id=pred.run_id, huber=float("inf"),
            final_huber=float("inf"), r2=0.0, mape=1.0, n_steps=0,
        )

    preds_list = [pred.predictions[s] for s in scoring_steps]
    truths_list = [truth.losses[s] for s in scoring_steps]

    huber_vals = [huber_loss(p, t, delta) for p, t in zip(preds_list, truths_list)]
    mean_huber = sum(huber_vals) / len(huber_vals)

    # Final step
    final_step = max(scoring_steps)
    final_h = huber_loss(pred.predictions[final_step], truth.losses[final_step], delta)

    return RunResult(
        run_id=pred.run_id,
        huber=mean_huber,
        final_huber=final_h,
        r2=compute_r2(preds_list, truths_list),
        mape=compute_mape(preds_list, truths_list),
        n_steps=len(scoring_steps),
    )


# ── Full evaluation ──────────────────────────────────────────────────────────

def evaluate(
    predictions: list[RunPrediction],
    ground_truths: list[RunGroundTruth],
    configs: Optional[list[RunConfig]] = None,
    delta: float = 0.01,
    extrap_compute_threshold: Optional[float] = None,
) -> EvalResult:
    """
    Evaluate all predictions against ground truths.

    Args:
        predictions: List of participant predictions.
        ground_truths: List of ground truth loss curves.
        configs: Optional configs (needed for extrapolation split).
        delta: Huber loss delta parameter.
        extrap_compute_threshold: If provided, runs with tokens_total above
            this are scored separately as "extrapolation".
    """
    truth_map = {gt.run_id: gt for gt in ground_truths}
    config_map = {c.run_id: c for c in configs} if configs else {}

    per_run = []
    all_preds = []
    all_truths = []
    final_preds = []
    final_truths = []
    extrap_preds = []
    extrap_truths = []

    for pred in predictions:
        if pred.run_id not in truth_map:
            continue
        truth = truth_map[pred.run_id]
        expected_steps = (
            config_map[pred.run_id].eval_steps
            if pred.run_id in config_map else None
        )
        result = evaluate_run(pred, truth, delta, expected_steps=expected_steps)
        per_run.append(result)

        # Collect all points
        if expected_steps is None:
            scoring_steps = sorted(set(pred.predictions.keys()) & set(truth.losses.keys()))
        else:
            scoring_steps = sorted(
                set(expected_steps) & set(pred.predictions.keys()) & set(truth.losses.keys())
            )
        for s in scoring_steps:
            all_preds.append(pred.predictions[s])
            all_truths.append(truth.losses[s])

        # Final
        if scoring_steps:
            fs = max(scoring_steps)
            final_preds.append(pred.predictions[fs])
            final_truths.append(truth.losses[fs])

        # Extrapolation
        if extrap_compute_threshold and pred.run_id in config_map:
            cfg = config_map[pred.run_id]
            if cfg.data.tokens_total > extrap_compute_threshold:
                for s in scoring_steps:
                    extrap_preds.append(pred.predictions[s])
                    extrap_truths.append(truth.losses[s])

    # Aggregate
    n_points = len(all_preds)
    mean_huber = (
        sum(huber_loss(p, t, delta) for p, t in zip(all_preds, all_truths)) / n_points
        if n_points > 0 else float("inf")
    )

    final_huber = (
        sum(huber_loss(p, t, delta) for p, t in zip(final_preds, final_truths)) / len(final_preds)
        if final_preds else float("inf")
    )

    # Extrapolation
    extrap_huber = None
    extrap_r2 = None
    extrap_n = 0
    if extrap_preds:
        extrap_huber = sum(huber_loss(p, t, delta) for p, t in zip(extrap_preds, extrap_truths)) / len(extrap_preds)
        extrap_r2 = compute_r2(extrap_preds, extrap_truths)
        # Count unique runs in extrapolation set
        extrap_run_ids = set()
        for pred in predictions:
            if pred.run_id in config_map:
                cfg = config_map[pred.run_id]
                if cfg.data.tokens_total > extrap_compute_threshold:
                    extrap_run_ids.add(pred.run_id)
        extrap_n = len(extrap_run_ids)

    return EvalResult(
        huber=mean_huber,
        r2=compute_r2(all_preds, all_truths),
        final_huber=final_huber,
        final_r2=compute_r2(final_preds, final_truths),
        curve_mape=compute_mape(all_preds, all_truths),
        curve_mae=compute_mae(all_preds, all_truths),
        curve_rmse=compute_rmse(all_preds, all_truths),
        final_mae=compute_mae(final_preds, final_truths),
        final_rmse=compute_rmse(final_preds, final_truths),
        final_spearman=compute_spearman(final_preds, final_truths),
        extrap_huber=extrap_huber,
        extrap_r2=extrap_r2,
        extrap_n_runs=extrap_n,
        n_runs=len(per_run),
        n_points=n_points,
        per_run=per_run,
    )
