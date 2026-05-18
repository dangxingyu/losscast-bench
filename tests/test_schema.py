"""Unit tests for schema validation and data classes."""

import json
import pytest
import tempfile
from pathlib import Path

from losscast_bench.schema import (
    ModelConfig,
    OptimizerConfig,
    DataConfig,
    ScheduleConfig,
    RunConfig,
    RunGroundTruth,
    RunPrediction,
    load_configs,
    load_ground_truths,
    load_predictions,
    save_predictions,
    validate_submission,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_config(run_id="run_1", eval_interval=100, total_steps=1000):
    """Create a minimal valid RunConfig."""
    return RunConfig(
        run_id=run_id,
        model=ModelConfig(arch="transformer", n_layers=12, d_model=768, n_heads=12),
        optimizer=OptimizerConfig(),
        data=DataConfig(dataset="c4"),
        schedule=ScheduleConfig(total_steps=total_steps),
        eval_interval=eval_interval,
    )


def _make_prediction(run_id="run_1", steps=None, values=None):
    """Create a minimal valid RunPrediction."""
    steps = steps or [100, 200, 300]
    values = values or [8.0, 6.0, 4.5]
    return RunPrediction(
        run_id=run_id,
        predictions=dict(zip(steps, values)),
    )


def _make_ground_truth(run_id="run_1", steps=None, values=None):
    """Create a minimal valid RunGroundTruth."""
    steps = steps or [100, 200, 300]
    values = values or [8.1, 6.1, 4.6]
    return RunGroundTruth(
        run_id=run_id,
        losses=dict(zip(steps, values)),
    )


# ── ModelConfig tests ───────────────────────────────────────────────────────

class TestModelConfig:
    def test_n_params_approx_transformer(self):
        m = ModelConfig(arch="transformer", n_layers=12, d_model=768, n_heads=12,
                        activation="swiglu")
        params = m.n_params_approx
        assert params > 0
        assert params == 12 * (4 * 768**2 + 3 * 768 * (4 * 768))

    def test_n_params_approx_gelu(self):
        m = ModelConfig(arch="transformer", n_layers=12, d_model=768, n_heads=12,
                        activation="gelu")
        params = m.n_params_approx
        assert params == 12 * (4 * 768**2 + 2 * 768 * (4 * 768))

    def test_n_params_custom_d_ff(self):
        m = ModelConfig(arch="transformer", n_layers=12, d_model=768, n_heads=12,
                        d_ff=2048, activation="gelu")
        params = m.n_params_approx
        assert params == 12 * (4 * 768**2 + 2 * 768 * 2048)

    def test_defaults(self):
        m = ModelConfig(arch="transformer", n_layers=1, d_model=64)
        assert m.vocab_size == 32000
        assert m.activation == "swiglu"
        assert m.norm_type == "rmsnorm"
        assert m.rope is True
        assert m.n_heads is None
        assert m.n_kv_heads is None
        assert m.tied_embeddings is False


# ── RunConfig serialization tests ───────────────────────────────────────────

class TestRunConfigSerialization:
    def test_roundtrip_dict(self):
        cfg = _make_config()
        d = cfg.to_dict()
        cfg2 = RunConfig.from_dict(d)
        assert cfg2.run_id == cfg.run_id
        assert cfg2.model.arch == cfg.model.arch
        assert cfg2.eval_interval == cfg.eval_interval
        assert cfg2.model.n_params_approx == cfg.model.n_params_approx

    def test_roundtrip_json(self):
        cfg = _make_config()
        j = cfg.to_json()
        cfg2 = RunConfig.from_json(j)
        assert cfg2.run_id == cfg.run_id

    def test_from_dict_defaults(self):
        """Minimal dict with only required fields."""
        d = {
            "run_id": "test",
            "model": {"arch": "transformer", "n_layers": 6, "d_model": 512},
            "optimizer": {},
            "data": {"dataset": "c4"},
            "schedule": {},
        }
        cfg = RunConfig.from_dict(d)
        assert cfg.run_id == "test"
        assert cfg.eval_interval == 500
        assert cfg.precision == "bf16"
        assert cfg.dp_size == 1


# ── eval_steps property tests ──────────────────────────────────────────────

class TestEvalStepsProperty:
    def test_eval_steps_computed(self):
        cfg = _make_config(eval_interval=500, total_steps=2000)
        assert cfg.eval_steps == [0, 500, 1000, 1500, 2000]

    def test_eval_steps_exact_division(self):
        cfg = _make_config(eval_interval=100, total_steps=300)
        assert cfg.eval_steps == [0, 100, 200, 300]

    def test_eval_steps_non_exact_division(self):
        cfg = _make_config(eval_interval=300, total_steps=1000)
        # 300, 600, 900 — 1000 is not a multiple of 300, so final step appended
        assert cfg.eval_steps == [0, 300, 600, 900, 1000]

    def test_eval_steps_interval_equals_total(self):
        cfg = _make_config(eval_interval=1000, total_steps=1000)
        assert cfg.eval_steps == [0, 1000]


# ── Renamed fields tests ───────────────────────────────────────────────────

class TestRenamedFields:
    def test_mup_field(self):
        opt = OptimizerConfig(mup=True)
        assert opt.mup is True
        opt2 = OptimizerConfig()
        assert opt2.mup is False

    def test_batch_tokens_field(self):
        data = DataConfig(dataset="c4", batch_tokens=1048576)
        assert data.batch_tokens == 1048576

    def test_final_lr_ratio_field(self):
        sched = ScheduleConfig(final_lr_ratio=0.05)
        assert sched.final_lr_ratio == 0.05

    def test_eval_interval_field(self):
        cfg = _make_config(eval_interval=250)
        assert cfg.eval_interval == 250

    def test_code_commit_roundtrip(self):
        cfg = RunConfig(
            run_id="test",
            model=ModelConfig(arch="transformer", n_layers=6, d_model=512),
            optimizer=OptimizerConfig(),
            data=DataConfig(dataset="c4"),
            schedule=ScheduleConfig(),
            code_repo="karpathy/autoresearch",
            code_commit="a3f7b2c",
        )
        d = cfg.to_dict()
        assert d["code_repo"] == "karpathy/autoresearch"
        assert d["code_commit"] == "a3f7b2c"
        cfg2 = RunConfig.from_dict(d)
        assert cfg2.code_repo == "karpathy/autoresearch"
        assert cfg2.code_commit == "a3f7b2c"

    def test_code_commit_optional_defaults_none(self):
        cfg = _make_config()
        assert cfg.code_repo is None
        assert cfg.code_commit is None
        d = cfg.to_dict()
        cfg2 = RunConfig.from_dict(d)
        assert cfg2.code_repo is None
        assert cfg2.code_commit is None

    def test_from_dict_without_code_commit(self):
        """Existing data without code_commit fields still parses cleanly."""
        d = {
            "run_id": "legacy_run",
            "model": {"arch": "transformer", "n_layers": 6, "d_model": 512},
            "optimizer": {},
            "data": {"dataset": "c4"},
            "schedule": {},
        }
        cfg = RunConfig.from_dict(d)
        assert cfg.code_repo is None
        assert cfg.code_commit is None

    def test_roundtrip_renamed_fields(self):
        cfg = RunConfig(
            run_id="test",
            model=ModelConfig(arch="transformer", n_layers=6, d_model=512),
            optimizer=OptimizerConfig(mup=True),
            data=DataConfig(dataset="c4", batch_tokens=262144),
            schedule=ScheduleConfig(final_lr_ratio=0.05),
            eval_interval=250,
        )
        d = cfg.to_dict()
        assert d["optimizer"]["mup"] is True
        assert d["data"]["batch_tokens"] == 262144
        assert d["schedule"]["final_lr_ratio"] == 0.05
        assert d["eval_interval"] == 250

        cfg2 = RunConfig.from_dict(d)
        assert cfg2.optimizer.mup is True
        assert cfg2.data.batch_tokens == 262144
        assert cfg2.schedule.final_lr_ratio == 0.05
        assert cfg2.eval_interval == 250


# ── RunPrediction / RunGroundTruth tests ────────────────────────────────────

class TestPredictionGroundTruth:
    def test_prediction_final_loss(self):
        p = _make_prediction(steps=[100, 500, 1000], values=[8.0, 5.0, 3.5])
        assert p.final_loss == 3.5

    def test_ground_truth_final_loss(self):
        gt = _make_ground_truth(steps=[100, 500, 1000], values=[8.1, 5.1, 3.6])
        assert gt.final_loss == 3.6

    def test_prediction_to_dict_string_keys(self):
        p = _make_prediction()
        d = p.to_dict()
        assert all(isinstance(k, str) for k in d["predictions"].keys())

    def test_prediction_from_dict_int_keys(self):
        d = {"run_id": "r1", "predictions": {"100": 8.0, "500": 5.0}}
        p = RunPrediction.from_dict(d)
        assert all(isinstance(k, int) for k in p.predictions.keys())
        assert p.predictions[100] == 8.0

    def test_ground_truth_to_dict_string_keys(self):
        gt = _make_ground_truth()
        d = gt.to_dict()
        assert all(isinstance(k, str) for k in d["losses"].keys())


# ── Batch I/O tests ─────────────────────────────────────────────────────────

class TestBatchIO:
    def test_save_load_predictions_roundtrip(self, tmp_path):
        preds = [_make_prediction("r1"), _make_prediction("r2")]
        path = str(tmp_path / "preds.json")
        save_predictions(preds, path)
        loaded = load_predictions(path)
        assert len(loaded) == 2
        assert loaded[0].run_id == "r1"
        assert loaded[1].run_id == "r2"

    def test_load_configs_from_runs_wrapper(self, tmp_path):
        data = {"runs": [
            {
                "run_id": "t1",
                "model": {"arch": "transformer", "n_layers": 6, "d_model": 512},
                "optimizer": {},
                "data": {"dataset": "c4"},
                "schedule": {},
                "eval_interval": 100,
            }
        ]}
        path = tmp_path / "configs.json"
        path.write_text(json.dumps(data))
        configs = load_configs(str(path))
        assert len(configs) == 1
        assert configs[0].run_id == "t1"

    def test_load_configs_from_bare_list(self, tmp_path):
        data = [
            {
                "run_id": "t1",
                "model": {"arch": "transformer", "n_layers": 6, "d_model": 512},
                "optimizer": {},
                "data": {"dataset": "c4"},
                "schedule": {},
                "eval_interval": 100,
            }
        ]
        path = tmp_path / "configs.json"
        path.write_text(json.dumps(data))
        configs = load_configs(str(path))
        assert len(configs) == 1


# ── Validation tests ────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_submission(self):
        configs = [_make_config("r1", eval_interval=100, total_steps=200)]
        steps = configs[0].eval_steps
        preds = [_make_prediction("r1", steps, [8.0] * len(steps))]
        errors = validate_submission(preds, configs)
        assert errors == []

    def test_missing_run(self):
        configs = [_make_config("r1"), _make_config("r2")]
        preds = [_make_prediction("r1", configs[0].eval_steps,
                                  [5.0] * len(configs[0].eval_steps))]
        errors = validate_submission(preds, configs)
        assert any("Missing predictions" in e for e in errors)

    def test_extra_run(self):
        configs = [_make_config("r1")]
        steps = configs[0].eval_steps
        preds = [
            _make_prediction("r1", steps, [5.0] * len(steps)),
            _make_prediction("r_unknown", [100], [5.0]),
        ]
        errors = validate_submission(preds, configs)
        assert any("Extra predictions" in e for e in errors)

    def test_missing_required_steps(self):
        configs = [_make_config("r1", eval_interval=100, total_steps=300)]
        # Expected steps are [0, 100, 200, 300], but only one is provided.
        preds = [_make_prediction("r1", [100], [8.0])]
        errors = validate_submission(preds, configs)
        assert any("missing predictions" in e for e in errors)

    def test_unexpected_steps(self):
        configs = [_make_config("r1", eval_interval=100, total_steps=200)]
        preds = [_make_prediction("r1", [0, 100, 150, 200], [8.0, 7.0, 6.0, 5.0])]
        errors = validate_submission(preds, configs)
        assert any("unexpected steps" in e for e in errors)

    def test_ground_truth_limits_required_steps(self):
        configs = [_make_config("r1", eval_interval=100, total_steps=300)]
        gts = [_make_ground_truth("r1", [0, 100, 200], [8.1, 6.1, 4.6])]
        preds = [_make_prediction("r1", [0, 100, 200], [8.0, 6.0, 4.5])]
        errors = validate_submission(preds, configs, gts)
        assert errors == []

    def test_negative_loss(self):
        configs = [_make_config("r1", eval_interval=100, total_steps=100)]
        preds = [_make_prediction("r1", [100], [-1.0])]
        errors = validate_submission(preds, configs)
        assert any("negative loss" in e for e in errors)

    def test_suspiciously_high_loss(self):
        configs = [_make_config("r1", eval_interval=100, total_steps=100)]
        preds = [_make_prediction("r1", [100], [25.0])]
        errors = validate_submission(preds, configs)
        assert any("suspiciously high" in e for e in errors)

    def test_multiple_errors(self):
        configs = [
            _make_config("r1", eval_interval=100, total_steps=200),
            _make_config("r2", eval_interval=100, total_steps=100),
        ]
        preds = [
            _make_prediction("r1", [100], [8.0]),  # missing step 200
            # missing r2 entirely
            _make_prediction("r3", [100], [5.0]),  # extra run
        ]
        errors = validate_submission(preds, configs)
        assert len(errors) >= 3

    def test_empty_submission(self):
        configs = [_make_config("r1")]
        preds = []
        errors = validate_submission(preds, configs)
        assert any("Missing predictions" in e for e in errors)
