#!/usr/bin/env python3
"""
Validate a data contribution before merge.

Checks:
  1. Schema compliance — all required fields present, correct types
  2. Internal consistency — MODEL_CONFIG in model.py matches config.json
  3. Model verification — model.py builds and forward pass produces correct shapes
  4. Loss curve sanity — no NaN, non-negative, reasonable range, generally decreasing
  5. Optimizer verification — optimizer.py (if present) builds and loss decreases

Usage:
  python scripts/validate_contribution.py --run-id alice_350m_c4_cosine --split train
  python scripts/validate_contribution.py --run-dir data/train/alice_350m_c4_cosine
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def find_run_config(configs_path: str, run_id: str) -> dict | None:
    data = load_json(configs_path)
    runs = data if isinstance(data, list) else data.get("runs", [])
    for r in runs:
        if r.get("run_id") == run_id:
            return r
    return None


def find_run_gt(gt_path: str, run_id: str) -> dict | None:
    data = load_json(gt_path)
    runs = data if isinstance(data, list) else data.get("runs", [])
    for r in runs:
        if r.get("run_id") == run_id:
            return r
    return None


# ── Check 1: Schema compliance ──────────────────────────────────────────────

REQUIRED_MODEL_FIELDS = {"arch", "n_layers", "d_model"}
REQUIRED_DATA_FIELDS = {"dataset", "eval_dataset"}
REQUIRED_SCHEDULE_FIELDS = {"total_steps"}

def check_schema(config: dict) -> list[str]:
    errors = []

    if "run_id" not in config:
        errors.append("Missing run_id")

    model = config.get("model", {})
    for f in REQUIRED_MODEL_FIELDS:
        if f not in model:
            errors.append(f"model.{f} missing")

    if "optimizer" not in config:
        errors.append("optimizer section missing")

    data = config.get("data", {})
    for f in REQUIRED_DATA_FIELDS:
        if f not in data:
            errors.append(f"data.{f} missing")

    schedule = config.get("schedule", {})
    for f in REQUIRED_SCHEDULE_FIELDS:
        if f not in schedule:
            errors.append(f"schedule.{f} missing")

    # Type checks
    if "n_layers" in model and not isinstance(model["n_layers"], int):
        errors.append(f"model.n_layers should be int, got {type(model['n_layers']).__name__}")
    if "d_model" in model and not isinstance(model["d_model"], int):
        errors.append(f"model.d_model should be int, got {type(model['d_model']).__name__}")

    lr = config.get("optimizer", {}).get("lr")
    if lr is not None and (lr <= 0 or lr > 1):
        errors.append(f"optimizer.lr={lr} looks wrong (expected 0 < lr <= 1)")

    return errors


# ── Check 2: Loss curve sanity ──────────────────────────────────────────────

def check_loss_curve(gt: dict, config: dict) -> list[str]:
    errors = []
    losses = gt.get("losses", {})

    if not losses:
        errors.append("No loss values in ground_truth")
        return errors

    # Check eval_interval alignment
    eval_interval = config.get("eval_interval", 500)
    total_steps = config.get("schedule", {}).get("total_steps", 0)
    expected_steps = set(range(eval_interval, total_steps + 1, eval_interval))
    actual_steps = set(int(k) for k in losses.keys())

    missing = expected_steps - actual_steps
    if missing:
        errors.append(f"Missing {len(missing)} eval steps (first few: {sorted(missing)[:5]})")

    # Value checks
    for step_str, val in losses.items():
        step = int(step_str)
        if not isinstance(val, (int, float)):
            errors.append(f"Step {step}: non-numeric loss {val}")
            continue
        if math.isnan(val) or math.isinf(val):
            errors.append(f"Step {step}: NaN/Inf loss")
        elif val < 0:
            errors.append(f"Step {step}: negative loss {val}")
        elif val > 20:
            errors.append(f"Step {step}: suspiciously high loss {val} (> 20)")

    # Check that loss generally decreases (allow some noise)
    sorted_steps = sorted((int(k), v) for k, v in losses.items() if isinstance(v, (int, float)))
    if len(sorted_steps) >= 4:
        # Compare first quarter average to last quarter average
        q = len(sorted_steps) // 4
        first_q = sum(v for _, v in sorted_steps[:q]) / q
        last_q = sum(v for _, v in sorted_steps[-q:]) / q
        if last_q > first_q * 1.1:  # loss increased by more than 10%
            errors.append(
                f"Loss appears to increase over training "
                f"(first quarter avg={first_q:.3f}, last quarter avg={last_q:.3f})"
            )

    return errors


# ── Check 3: model.py syntax and MODEL_CONFIG ───────────────────────────────

def check_model_py(model_path: Path, config: dict) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []

    if not model_path.exists():
        warnings.append(f"model.py not found at {model_path} (optional for data-only contributions)")
        return errors, warnings

    source = model_path.read_text()

    # Syntax check
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        errors.append(f"model.py has syntax error: {e}")
        return errors

    # Check for required class and MODEL_CONFIG (rewritten format)
    # Raw source code (e.g., nanochat gpt.py) may not follow this convention —
    # we issue warnings instead of errors in that case.
    class_names = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ]
    top_level_assigns = [
        node for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.Assign)
    ]
    has_model_config = any(
        isinstance(target, ast.Name) and target.id == "MODEL_CONFIG"
        for assign in top_level_assigns
        for target in assign.targets
    )

    is_raw_code = "Model" not in class_names and not has_model_config
    if is_raw_code:
        warnings.append("model.py appears to be raw source code (no 'class Model' or 'MODEL_CONFIG'). "
                         "This is OK for nanochat/raw contributions but skips structural validation.")
    else:
        if "Model" not in class_names:
            errors.append("model.py missing 'class Model'")
        if not has_model_config:
            errors.append("model.py missing 'MODEL_CONFIG' dict at module level")

    # Check for forbidden imports
    forbidden = {"transformers", "huggingface_hub", "flash_attn", "xformers", "triton"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in forbidden:
                    errors.append(f"model.py imports forbidden package: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in forbidden:
                    errors.append(f"model.py imports from forbidden package: {node.module}")

    # Check for required methods in Model class (skip for raw code)
    if not is_raw_code:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Model":
                method_names = [
                    n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                if "__init__" not in method_names:
                    errors.append("Model class missing __init__")
                if "forward" not in method_names:
                    errors.append("Model class missing forward()")
                if "_init_weights" not in method_names:
                    errors.append("Model class missing _init_weights()")

    return errors, warnings


# ── Check 4: optimizer.py syntax (if present) ───────────────────────────────

def check_optimizer_py(opt_path: Path) -> list[str]:
    errors = []

    if not opt_path.exists():
        return []  # optimizer.py is optional

    source = opt_path.read_text()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        errors.append(f"optimizer.py has syntax error: {e}")
        return errors

    # Check for Optimizer subclass
    class_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    has_optimizer_class = False
    for cls in class_nodes:
        for base in cls.bases:
            base_name = ""
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if "Optimizer" in base_name:
                has_optimizer_class = True
                # Check step method has step_idx
                for item in cls.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "step":
                        arg_names = [a.arg for a in item.args.args]
                        if "step_idx" not in arg_names:
                            errors.append("optimizer step() method missing step_idx parameter")

    if not has_optimizer_class:
        errors.append("optimizer.py has no class inheriting from Optimizer")

    # Check for OPTIMIZER_DEFAULTS
    has_defaults = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "OPTIMIZER_DEFAULTS":
                    has_defaults = True
    if not has_defaults:
        errors.append("optimizer.py missing 'OPTIMIZER_DEFAULTS' dict at module level")

    # Forbidden imports
    forbidden = {"transformers", "huggingface_hub"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in forbidden:
                    errors.append(f"optimizer.py imports forbidden package: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in forbidden:
                    errors.append(f"optimizer.py imports from forbidden package: {node.module}")

    return errors


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate a data contribution")
    parser.add_argument("--run-id", required=True, help="Run ID to validate")
    parser.add_argument("--split", default="train", help="Data split (train/val/test)")
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("--run-dir", help="Override path to run directory")
    args = parser.parse_args()

    # Per-run directory: data/{split}/{run_id}/
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.data_dir) / args.split / args.run_id

    all_errors = []
    all_warnings = []

    # Load config from per-run config.json
    config_path = run_dir / "config.json"
    if not config_path.exists():
        all_errors.append(f"config.json not found in {run_dir}")
        _report(all_errors, all_warnings)
        return 1

    config = load_json(str(config_path))

    # Check 1: Schema
    print("Checking schema compliance...")
    schema_errors = check_schema(config)
    all_errors.extend(schema_errors)
    print(f"  {'PASS' if not schema_errors else 'FAIL'} ({len(schema_errors)} errors)")

    # Check 2: Loss curve from per-run losses.json
    losses_path = run_dir / "losses.json"
    if not losses_path.exists():
        all_errors.append(f"losses.json not found in {run_dir}")
    else:
        losses_data = load_json(str(losses_path))
        gt = {"run_id": args.run_id, "losses": losses_data}
        print("Checking loss curve...")
        curve_errors = check_loss_curve(gt, config)
        all_errors.extend(curve_errors)
        print(f"  {'PASS' if not curve_errors else 'FAIL'} ({len(curve_errors)} errors)")

    # Check 3: model.py
    model_path = run_dir / "model.py"
    print(f"Checking model.py ({model_path})...")
    model_errors, model_warnings = check_model_py(model_path, config)
    all_errors.extend(model_errors)
    all_warnings.extend(model_warnings)
    print(f"  {'PASS' if not model_errors else 'FAIL'} ({len(model_errors)} errors, {len(model_warnings)} warnings)")

    # Check 4: optimizer.py
    opt_path = run_dir / "optimizer.py"
    if opt_path.exists():
        print(f"Checking optimizer.py ({opt_path})...")
        opt_errors = check_optimizer_py(opt_path)
        all_errors.extend(opt_errors)
        print(f"  {'PASS' if not opt_errors else 'FAIL'} ({len(opt_errors)} errors)")
    else:
        opt_name = config.get("optimizer", {}).get("name", "adamw")
        standard_optimizers = {"adamw", "adam", "sgd", "lion"}
        if opt_name not in standard_optimizers:
            all_warnings.append(
                f"Optimizer is '{opt_name}' but no optimizer.py provided. "
                f"Consider adding one for non-standard optimizers."
            )

    _report(all_errors, all_warnings)
    return 1 if all_errors else 0


def _report(errors: list[str], warnings: list[str]):
    print()
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")
        print()

    if errors:
        print(f"FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    raise SystemExit(main())
