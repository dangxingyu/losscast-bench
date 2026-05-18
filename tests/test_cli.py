"""Integration tests for eval_cli."""

import json
import subprocess
import sys
import pytest
from pathlib import Path

from losscast_bench.data import load_split

PROJECT_ROOT = Path(__file__).parent.parent
CLI_MODULE = "scripts.eval_cli"


def run_cli(*args, check=True) -> subprocess.CompletedProcess:
    """Run eval_cli as a subprocess and return the result."""
    cmd = [sys.executable, "-m", CLI_MODULE, *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=check,
    )


@pytest.fixture
def preds_path(tmp_path):
    """Generate Chinchilla baseline predictions and return the path."""
    from losscast_bench.schema import save_predictions
    from losscast_bench.baselines.chinchilla import predict_batch

    configs, _ = load_split("val")
    preds = predict_batch(configs)
    path = tmp_path / "preds.json"
    save_predictions(preds, str(path))
    return str(path)


@pytest.fixture
def gt_path(tmp_path):
    """Write ground truths to a monolithic file for CLI testing."""
    _, gts = load_split("val")
    data = {"runs": [gt.to_dict() for gt in gts]}
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def configs_path(tmp_path):
    """Write configs to a monolithic file for CLI testing."""
    configs, _ = load_split("val")
    data = {"runs": [c.to_dict() for c in configs]}
    path = tmp_path / "configs.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def bad_preds_path(tmp_path):
    """Create a predictions file with various issues."""
    # Get an actual run_id from val split
    configs, _ = load_split("val")
    real_run_id = configs[0].run_id
    # Missing a run, has an extra run, too few steps, negative value
    data = {
        "runs": [
            {
                "run_id": real_run_id,
                "predictions": {"500": 8.0},  # too few steps
            },
            {
                "run_id": "nonexistent_run",
                "predictions": {"500": -1.0},  # extra run + negative value
            },
        ]
    }
    path = tmp_path / "bad_preds.json"
    path.write_text(json.dumps(data))
    return str(path)


# ── Basic evaluation tests (using --split) ─────────────────────────────────

class TestBasicEvalSplit:
    def test_table_format_with_split(self, preds_path):
        result = run_cli(
            "--split", "val",
            "-p", preds_path,
        )
        assert result.returncode == 0
        assert "Huber Loss" in result.stdout
        assert "R²" in result.stdout

    def test_json_format_with_split(self, preds_path):
        result = run_cli(
            "--split", "val",
            "-p", preds_path,
            "--format", "json",
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "huber" in data
        assert "r2" in data
        assert "final_huber" in data
        assert "n_runs" in data

    def test_json_with_verbose(self, preds_path):
        result = run_cli(
            "--split", "val",
            "-p", preds_path,
            "--format", "json",
            "--verbose",
        )
        data = json.loads(result.stdout)
        assert "per_run" in data
        assert len(data["per_run"]) > 0
        assert "run_id" in data["per_run"][0]


# ── Basic evaluation tests (using explicit files) ─────────────────────────

class TestBasicEvalFiles:
    def test_table_format_default(self, preds_path, gt_path, configs_path):
        result = run_cli(
            "-p", preds_path,
            "-g", gt_path,
            "-c", configs_path,
        )
        assert result.returncode == 0
        assert "Huber Loss" in result.stdout
        assert "R²" in result.stdout

    def test_table_format_explicit(self, preds_path, gt_path, configs_path):
        result = run_cli(
            "-p", preds_path,
            "-g", gt_path,
            "-c", configs_path,
            "--format", "table",
        )
        assert result.returncode == 0
        assert "LossCast-Bench Evaluation Results" in result.stdout

    def test_json_format(self, preds_path, gt_path, configs_path):
        result = run_cli(
            "-p", preds_path,
            "-g", gt_path,
            "-c", configs_path,
            "--format", "json",
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "huber" in data
        assert "r2" in data


# ── Verbose / per-run breakdown ─────────────────────────────────────────────

class TestVerbose:
    def test_verbose_shows_per_run(self, preds_path):
        result = run_cli(
            "--split", "val",
            "-p", preds_path,
            "--format", "table",
            "--verbose",
        )
        assert result.returncode == 0
        assert "Per-run breakdown" in result.stdout
        # Check that at least one run ID appears in the output
        assert "ncpl_" in result.stdout

    def test_no_verbose_no_per_run(self, preds_path):
        result = run_cli(
            "--split", "val",
            "-p", preds_path,
            "--format", "table",
        )
        assert "Per-run breakdown" not in result.stdout


# ── Validation tests ────────────────────────────────────────────────────────

class TestValidation:
    def test_validation_with_split(self, preds_path):
        result = run_cli(
            "--split", "val",
            "-p", preds_path,
        )
        assert result.returncode == 0
        assert "Validation Report" in result.stdout
        assert "PASSED" in result.stdout

    def test_validation_with_configs(self, preds_path, gt_path, configs_path):
        result = run_cli(
            "-p", preds_path,
            "-g", gt_path,
            "-c", configs_path,
        )
        assert result.returncode == 0
        assert "Validation Report" in result.stdout
        assert "PASSED" in result.stdout

    def test_validation_bad_submission(self, bad_preds_path, gt_path, configs_path):
        result = run_cli(
            "-p", bad_preds_path,
            "-g", gt_path,
            "-c", configs_path,
            check=False,
        )
        assert result.returncode == 1
        assert "FAILED" in result.stdout or "error" in result.stdout.lower()
        assert "LossCast-Bench Evaluation Results" not in result.stdout

    def test_allow_invalid_runs_debug_eval(self, bad_preds_path, gt_path, configs_path):
        result = run_cli(
            "-p", bad_preds_path,
            "-g", gt_path,
            "-c", configs_path,
            "--allow-invalid",
            check=False,
        )
        assert result.returncode == 0
        assert "FAILED" in result.stdout
        assert "LossCast-Bench Evaluation Results" in result.stdout

    def test_validate_only_good(self, preds_path):
        result = run_cli(
            "--split", "val",
            "-p", preds_path,
            "--validate-only",
        )
        assert result.returncode == 0

    def test_validate_only_bad(self, bad_preds_path, gt_path, configs_path):
        result = run_cli(
            "-p", bad_preds_path,
            "-g", gt_path,
            "-c", configs_path,
            "--validate-only",
            check=False,
        )
        assert result.returncode == 1

    def test_validate_only_json(self, bad_preds_path, gt_path, configs_path):
        result = run_cli(
            "-p", bad_preds_path,
            "-g", gt_path,
            "-c", configs_path,
            "--validate-only",
            "--format", "json",
            check=False,
        )
        data = json.loads(result.stdout)
        assert data["status"] == "FAILED"
        assert len(data["errors"]) > 0

    def test_validate_only_without_configs_or_split(self, preds_path, gt_path):
        result = run_cli(
            "-p", preds_path,
            "--validate-only",
            check=False,
        )
        assert result.returncode == 2


# ── Output file tests ──────────────────────────────────────────────────────

class TestOutputFile:
    def test_save_results(self, preds_path, tmp_path):
        out_file = str(tmp_path / "results.json")
        result = run_cli(
            "--split", "val",
            "-p", preds_path,
            "-o", out_file,
        )
        assert result.returncode == 0
        with open(out_file) as f:
            data = json.load(f)
        assert "huber" in data
        # Output file always includes per-run
        assert "per_run" in data

    def test_save_with_validation(self, preds_path, tmp_path):
        out_file = str(tmp_path / "results.json")
        result = run_cli(
            "--split", "val",
            "-p", preds_path,
            "-o", out_file,
        )
        assert result.returncode == 0
        with open(out_file) as f:
            data = json.load(f)
        assert "validation" in data
        assert data["validation"]["status"] == "PASSED"


# ── Extrapolation tests ────────────────────────────────────────────────────

class TestExtrapolation:
    def test_extrap_threshold(self, preds_path):
        result = run_cli(
            "--split", "val",
            "-p", preds_path,
            "--extrap-threshold", "10e9",
            "--format", "json",
        )
        data = json.loads(result.stdout)
        assert "extrap_huber" in data


# ── Custom delta tests ──────────────────────────────────────────────────────

class TestCustomDelta:
    def test_custom_delta(self, preds_path):
        result = run_cli(
            "--split", "val",
            "-p", preds_path,
            "--delta", "0.1",
            "--format", "json",
        )
        data = json.loads(result.stdout)
        assert data["huber"] >= 0
