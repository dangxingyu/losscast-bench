"""Unit tests for evaluation metrics."""

import math
import pytest

from losscast_bench.metrics.scoring import (
    huber_loss,
    compute_r2,
    compute_mape,
    compute_mae,
    compute_rmse,
    compute_spearman,
    evaluate_run,
    evaluate,
    RunResult,
    EvalResult,
)
from losscast_bench.schema import (
    RunPrediction,
    RunGroundTruth,
    RunConfig,
    ModelConfig,
    OptimizerConfig,
    DataConfig,
    ScheduleConfig,
)


# ── Huber loss tests ────────────────────────────────────────────────────────

class TestHuberLoss:
    def test_zero_error(self):
        assert huber_loss(3.0, 3.0) == 0.0

    def test_small_error_quadratic(self):
        # |pred - truth| = 0.005 <= delta=0.01 → quadratic regime
        result = huber_loss(3.0, 3.005, delta=0.01)
        expected = 0.5 * 0.005**2
        assert abs(result - expected) < 1e-12

    def test_large_error_linear(self):
        # |pred - truth| = 1.0 >> delta=0.01 → linear regime
        result = huber_loss(3.0, 4.0, delta=0.01)
        expected = 0.01 * (1.0 - 0.5 * 0.01)
        assert abs(result - expected) < 1e-12

    def test_at_delta_boundary(self):
        # Exactly at delta, both formulas should give same result
        delta = 0.01
        result = huber_loss(3.0, 3.0 + delta, delta=delta)
        expected = 0.5 * delta**2
        assert abs(result - expected) < 1e-12

    def test_symmetric(self):
        assert huber_loss(3.0, 4.0) == huber_loss(4.0, 3.0)

    def test_custom_delta(self):
        result = huber_loss(0.0, 0.5, delta=1.0)
        # |0.5| <= 1.0, so quadratic
        expected = 0.5 * 0.5**2
        assert abs(result - expected) < 1e-12


# ── R² tests ───────────────────────────────────────────────────────────────

class TestR2:
    def test_perfect_predictions(self):
        truths = [1.0, 2.0, 3.0, 4.0]
        preds = [1.0, 2.0, 3.0, 4.0]
        assert compute_r2(preds, truths) == 1.0

    def test_mean_predictor(self):
        truths = [1.0, 2.0, 3.0, 4.0]
        mean = 2.5
        preds = [mean, mean, mean, mean]
        assert abs(compute_r2(preds, truths)) < 1e-10

    def test_worse_than_mean(self):
        truths = [1.0, 2.0, 3.0]
        preds = [10.0, 10.0, 10.0]  # terrible predictions
        r2 = compute_r2(preds, truths)
        assert r2 < 0

    def test_empty(self):
        assert compute_r2([], []) == 0.0

    def test_constant_truth(self):
        # All truths are the same → ss_tot = 0
        truths = [3.0, 3.0, 3.0]
        preds = [3.0, 3.0, 3.0]
        assert compute_r2(preds, truths) == 1.0

    def test_constant_truth_nonzero_error(self):
        truths = [3.0, 3.0, 3.0]
        preds = [3.1, 3.0, 2.9]
        assert compute_r2(preds, truths) == 0.0


# ── MAPE tests ──────────────────────────────────────────────────────────────

class TestMAPE:
    def test_perfect(self):
        assert compute_mape([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_known_value(self):
        preds = [1.1, 2.2]
        truths = [1.0, 2.0]
        # |0.1/1| + |0.2/2| = 0.1 + 0.1 = 0.2, mean = 0.1
        expected = 0.1
        assert abs(compute_mape(preds, truths) - expected) < 1e-10

    def test_skip_near_zero_truth(self):
        preds = [1.0, 5.0]
        truths = [0.0, 5.0]  # first truth is zero → skipped
        # Only second pair: |0/5| = 0
        assert compute_mape(preds, truths) == 0.0

    def test_empty(self):
        assert compute_mape([], []) == 0.0


# ── MAE / RMSE / Spearman tests ────────────────────────────────────────────

class TestMAE:
    def test_perfect(self):
        assert compute_mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_value(self):
        assert compute_mae([1.0, 2.0], [2.0, 4.0]) == 1.5

    def test_empty(self):
        assert compute_mae([], []) == 0.0


class TestRMSE:
    def test_perfect(self):
        assert compute_rmse([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_known_value(self):
        # errors = (1, 1) → RMSE = 1.0
        assert abs(compute_rmse([1.0, 2.0], [2.0, 3.0]) - 1.0) < 1e-12
        # errors = (3, 4) → RMSE = sqrt(25/2) = 5/sqrt(2)
        assert abs(compute_rmse([0.0, 0.0], [3.0, 4.0]) - math.sqrt(12.5)) < 1e-12

    def test_empty(self):
        assert compute_rmse([], []) == 0.0


class TestSpearman:
    def test_perfect_monotone(self):
        # identical ordering → ρ=1
        assert abs(compute_spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) - 1.0) < 1e-12

    def test_reversed(self):
        assert abs(compute_spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) + 1.0) < 1e-12

    def test_ties_use_avg_rank(self):
        # [1, 1, 2] vs [10, 10, 20] — both have ties in first two, ρ should be 1
        rho = compute_spearman([1.0, 1.0, 2.0], [10.0, 10.0, 20.0])
        assert abs(rho - 1.0) < 1e-12

    def test_constant_returns_zero(self):
        assert compute_spearman([1.0, 1.0, 1.0], [10.0, 20.0, 30.0]) == 0.0

    def test_too_small(self):
        assert compute_spearman([1.0], [2.0]) == 0.0


# ── Per-run evaluation tests ───────────────────────────────────────────────

class TestEvaluateRun:
    def test_perfect_prediction(self):
        pred = RunPrediction(run_id="r1", predictions={100: 8.0, 500: 5.0, 1000: 3.5})
        truth = RunGroundTruth(run_id="r1", losses={100: 8.0, 500: 5.0, 1000: 3.5})
        result = evaluate_run(pred, truth)
        assert result.huber == 0.0
        assert result.final_huber == 0.0
        assert result.r2 == 1.0
        assert result.mape == 0.0
        assert result.n_steps == 3

    def test_no_common_steps(self):
        pred = RunPrediction(run_id="r1", predictions={100: 8.0})
        truth = RunGroundTruth(run_id="r1", losses={200: 7.0})
        result = evaluate_run(pred, truth)
        assert result.huber == float("inf")
        assert result.n_steps == 0

    def test_partial_overlap(self):
        pred = RunPrediction(run_id="r1", predictions={100: 8.0, 500: 5.0, 1000: 3.5})
        truth = RunGroundTruth(run_id="r1", losses={100: 8.1, 500: 5.1})
        result = evaluate_run(pred, truth)
        assert result.n_steps == 2
        # Final step should be 500 (max of common)
        assert result.final_huber == huber_loss(5.0, 5.1)

    def test_final_huber_uses_last_common_step(self):
        pred = RunPrediction(run_id="r1", predictions={100: 8.0, 1000: 3.5})
        truth = RunGroundTruth(run_id="r1", losses={100: 8.0, 1000: 3.6})
        result = evaluate_run(pred, truth)
        assert result.final_huber == huber_loss(3.5, 3.6)

    def test_expected_steps_limit_scoring_grid(self):
        pred = RunPrediction(run_id="r1", predictions={0: 9.0, 500: 5.0, 750: 100.0, 1000: 3.5})
        truth = RunGroundTruth(run_id="r1", losses={0: 9.0, 500: 5.0, 750: 0.0, 1000: 3.5})
        result = evaluate_run(pred, truth, expected_steps=[0, 500, 1000])
        assert result.huber == 0.0
        assert result.n_steps == 3


# ── Full evaluation tests ──────────────────────────────────────────────────

def _make_config(run_id, tokens_total=10e9):
    return RunConfig(
        run_id=run_id,
        model=ModelConfig(arch="transformer", n_layers=12, d_model=768, n_heads=12),
        optimizer=OptimizerConfig(),
        data=DataConfig(dataset="c4", tokens_total=tokens_total),
        schedule=ScheduleConfig(total_steps=1000),
        eval_interval=100,
    )


class TestEvaluate:
    def test_basic_evaluation(self):
        preds = [
            RunPrediction(run_id="r1", predictions={100: 8.0, 500: 5.0, 1000: 3.5}),
            RunPrediction(run_id="r2", predictions={100: 7.0, 500: 4.5, 1000: 3.0}),
        ]
        truths = [
            RunGroundTruth(run_id="r1", losses={100: 8.1, 500: 5.1, 1000: 3.6}),
            RunGroundTruth(run_id="r2", losses={100: 7.1, 500: 4.6, 1000: 3.1}),
        ]
        result = evaluate(preds, truths)
        assert result.n_runs == 2
        assert result.n_points == 6
        assert result.huber >= 0
        assert len(result.per_run) == 2

    def test_configs_define_scoring_grid(self):
        configs = [_make_config("r1")]
        expected = configs[0].eval_steps
        grid_values = {step: float(10 - step / 1000) for step in expected}
        preds = [
            RunPrediction(run_id="r1", predictions={**grid_values, 150: 100.0}),
        ]
        truths = [
            RunGroundTruth(run_id="r1", losses={**grid_values, 150: 0.0}),
        ]
        result = evaluate(preds, truths, configs)
        assert result.huber == 0.0
        assert result.n_points == len(expected)

    def test_perfect_evaluation(self):
        preds = [RunPrediction(run_id="r1", predictions={100: 8.0, 500: 5.0})]
        truths = [RunGroundTruth(run_id="r1", losses={100: 8.0, 500: 5.0})]
        result = evaluate(preds, truths)
        assert result.huber == 0.0
        assert result.r2 == 1.0
        assert result.final_huber == 0.0

    def test_unmatched_run_ids_skipped(self):
        preds = [RunPrediction(run_id="r1", predictions={100: 8.0})]
        truths = [RunGroundTruth(run_id="r2", losses={100: 8.0})]
        result = evaluate(preds, truths)
        assert result.n_runs == 0
        assert result.n_points == 0

    def test_extrapolation_split(self):
        configs = [_make_config("r1", 10e9), _make_config("r2", 100e9)]
        preds = [
            RunPrediction(run_id="r1", predictions={100: 8.0, 500: 5.0, 1000: 3.5}),
            RunPrediction(run_id="r2", predictions={100: 7.0, 500: 4.5, 1000: 3.0}),
        ]
        truths = [
            RunGroundTruth(run_id="r1", losses={100: 8.1, 500: 5.1, 1000: 3.6}),
            RunGroundTruth(run_id="r2", losses={100: 7.1, 500: 4.6, 1000: 3.1}),
        ]
        result = evaluate(preds, truths, configs, extrap_compute_threshold=50e9)
        assert result.extrap_huber is not None
        assert result.extrap_n_runs == 1  # only r2 is above threshold

    def test_no_extrapolation_when_no_threshold(self):
        preds = [RunPrediction(run_id="r1", predictions={100: 8.0})]
        truths = [RunGroundTruth(run_id="r1", losses={100: 8.1})]
        result = evaluate(preds, truths)
        assert result.extrap_huber is None

    def test_summary_keys(self):
        preds = [RunPrediction(run_id="r1", predictions={100: 8.0, 500: 5.0})]
        truths = [RunGroundTruth(run_id="r1", losses={100: 8.1, 500: 5.1})]
        result = evaluate(preds, truths)
        s = result.summary()
        required_keys = {
            "huber", "r2", "final_huber", "final_r2", "curve_mape",
            "curve_mae", "curve_rmse", "final_mae", "final_rmse",
            "final_spearman", "n_runs", "n_points",
        }
        assert required_keys.issubset(set(s.keys()))

    def test_mae_rmse_spearman_propagate(self):
        # Two runs so Spearman has >=2 data points
        preds = [
            RunPrediction(run_id="r1", predictions={100: 8.0, 1000: 3.5}),
            RunPrediction(run_id="r2", predictions={100: 7.0, 1000: 3.0}),
        ]
        truths = [
            RunGroundTruth(run_id="r1", losses={100: 8.2, 1000: 3.7}),
            RunGroundTruth(run_id="r2", losses={100: 7.1, 1000: 3.1}),
        ]
        result = evaluate(preds, truths)
        # Curve-wide errors: (0.2, 0.2, 0.1, 0.1) → MAE = 0.15
        assert abs(result.curve_mae - 0.15) < 1e-10
        # Final-step errors: r1 final=1000 err=0.2, r2 final=1000 err=0.1 → MAE=0.15
        assert abs(result.final_mae - 0.15) < 1e-10
        # Final-step predictions agree on ordering with truths → ρ=1
        assert abs(result.final_spearman - 1.0) < 1e-10
