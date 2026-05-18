#!/usr/bin/env python3
"""
LossCast-Bench — Numerical Equivalence Verification

Verifies that a converted model.py (and optionally optimizer.py) produces
identical numerical results to the original training code.

Three levels of verification:
  1. Forward pass:   Same input → same logits (or close enough)
  2. Backward pass:  Same loss → same gradients
  3. Optimizer step: Same update rule → same parameter delta after one step

Usage:
  python verify_equivalence.py \
    --original-model original_model.py \
    --converted-model converted/model.py \
    --config converted/config.json \
    [--original-optimizer original_optim.py] \
    [--converted-optimizer converted/optimizer.py] \
    [--seed 42] \
    [--seq-len 128] \
    [--batch-size 2] \
    [--atol 1e-5] \
    [--rtol 1e-4]

The original model/optimizer files are loaded via importlib, so they must be
self-contained Python files with the expected class names.

Expected conventions:
  - Original model file: must define a model class and a config/dataclass
    that can build the model. The script will try common patterns:
      * GPT(GPTConfig(...))
      * Model(MODEL_CONFIG)
      * Transformer(config)
  - Converted model file: must define Model(config: dict) and MODEL_CONFIG
  - Original optimizer file: any Optimizer subclass
  - Converted optimizer file: CustomOptimizer + OPTIMIZER_DEFAULTS
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ── Utility: load a Python file as a module ──────────────────────────────────

def load_module(path: str, module_name: str = "loaded_module"):
    """Import a .py file as a module."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Utility: build model from module ─────────────────────────────────────────

def build_original_model(mod, config_overrides: Optional[dict] = None):
    """
    Try to build the original model from a loaded module.
    Supports multiple conventions (GPT, Model, Transformer, etc.).
    """
    # Try common patterns
    # Pattern 1: GPT + GPTConfig (nanochat style)
    if hasattr(mod, "GPT") and hasattr(mod, "GPTConfig"):
        cfg_cls = mod.GPTConfig
        kwargs = config_overrides or {}
        config = cfg_cls(**kwargs)
        model = mod.GPT(config)
        if hasattr(model, "init_weights"):
            model.init_weights()
        return model, config

    # Pattern 2: Model + MODEL_CONFIG (losscast-bench style)
    if hasattr(mod, "Model") and hasattr(mod, "MODEL_CONFIG"):
        config = dict(mod.MODEL_CONFIG)
        if config_overrides:
            config.update(config_overrides)
        model = mod.Model(config)
        return model, config

    # Pattern 3: any nn.Module subclass + config
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
            # Try to find a matching config
            config_name = name + "Config"
            if hasattr(mod, config_name):
                cfg_cls = getattr(mod, config_name)
                config = cfg_cls(**(config_overrides or {}))
                model = obj(config)
                if hasattr(model, "init_weights"):
                    model.init_weights()
                return model, config

    raise RuntimeError(
        f"Could not find a model class in {mod.__file__}. "
        "Expected one of: GPT+GPTConfig, Model+MODEL_CONFIG, or <Name>+<Name>Config"
    )


def build_converted_model(mod):
    """Build the converted model (always follows losscast-bench convention)."""
    assert hasattr(mod, "Model"), "Converted model must define class Model"
    assert hasattr(mod, "MODEL_CONFIG"), "Converted model must define MODEL_CONFIG"
    model = mod.Model(mod.MODEL_CONFIG)
    return model, mod.MODEL_CONFIG


def build_original_optimizer(mod, model_params):
    """Build optimizer from original optimizer module."""
    # Try common patterns
    # Pattern 1: MuonAdamW (nanochat style) — needs param_groups from model
    if hasattr(mod, "MuonAdamW"):
        # This requires the model's setup_optimizer method
        raise RuntimeError(
            "MuonAdamW requires model.setup_optimizer(). "
            "Please use --original-setup-fn to specify how to create the optimizer."
        )

    # Pattern 2: any Optimizer subclass + defaults
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, torch.optim.Optimizer) and obj is not torch.optim.Optimizer:
            defaults = getattr(mod, "OPTIMIZER_DEFAULTS", {})
            lr = defaults.pop("lr", 1e-3) if "lr" in defaults else 1e-3
            return obj(model_params, lr=lr, **{k: v for k, v in defaults.items() if k != "name"})

    raise RuntimeError(f"Could not find an Optimizer subclass in {mod.__file__}")


def build_converted_optimizer(mod, model_params):
    """Build optimizer from converted optimizer module."""
    assert hasattr(mod, "CustomOptimizer"), "Converted optimizer must define CustomOptimizer"
    defaults = dict(mod.OPTIMIZER_DEFAULTS)
    name = defaults.pop("name", "custom")
    # Remove non-constructor args
    for key in list(defaults.keys()):
        if key in ("name",):
            defaults.pop(key)
    return mod.CustomOptimizer(model_params, **defaults)


# ── Core verification functions ──────────────────────────────────────────────

def compare_tensors(a: torch.Tensor, b: torch.Tensor, name: str, atol: float, rtol: float) -> dict:
    """Compare two tensors and return a result dict."""
    if a.shape != b.shape:
        return {
            "name": name,
            "passed": False,
            "reason": f"Shape mismatch: {a.shape} vs {b.shape}",
            "max_abs_diff": float("inf"),
            "max_rel_diff": float("inf"),
        }

    abs_diff = (a.float() - b.float()).abs()
    max_abs = abs_diff.max().item()

    # Relative diff (avoid division by zero)
    denom = b.float().abs().clamp(min=1e-8)
    rel_diff = abs_diff / denom
    max_rel = rel_diff.max().item()
    mean_abs = abs_diff.mean().item()

    passed = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)

    return {
        "name": name,
        "passed": passed,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "max_rel_diff": max_rel,
        "atol": atol,
        "rtol": rtol,
    }


def verify_forward(
    original_model: nn.Module,
    converted_model: nn.Module,
    input_ids: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict:
    """
    Verify forward pass equivalence.
    Both models receive the same input_ids and should produce the same logits.
    """
    original_model.eval()
    converted_model.eval()

    with torch.no_grad():
        # Handle different forward signatures
        try:
            logits_orig = original_model(input_ids)
        except TypeError:
            # Some models return loss when targets=None, or have different signatures
            logits_orig = original_model.forward(input_ids)

        logits_conv = converted_model(input_ids)

    # If original returns a loss (scalar), skip this check
    if logits_orig.dim() == 0:
        return {
            "stage": "forward",
            "passed": False,
            "reason": "Original model returned scalar (loss?), not logits. Check forward() signature.",
            "details": [],
        }

    # Crop to same vocab size if needed (nanochat pads vocab)
    min_vocab = min(logits_orig.shape[-1], logits_conv.shape[-1])
    logits_orig = logits_orig[..., :min_vocab]
    logits_conv = logits_conv[..., :min_vocab]

    result = compare_tensors(logits_orig, logits_conv, "logits", atol, rtol)

    return {
        "stage": "forward",
        "passed": result["passed"],
        "details": [result],
        "logits_shape_orig": list(logits_orig.shape),
        "logits_shape_conv": list(logits_conv.shape),
    }


def verify_backward(
    original_model: nn.Module,
    converted_model: nn.Module,
    input_ids: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict:
    """
    Verify backward pass equivalence.
    Both models compute gradients from the same dummy loss; gradients should match.
    """
    original_model.train()
    converted_model.train()
    original_model.zero_grad()
    converted_model.zero_grad()

    # Forward
    logits_orig = original_model(input_ids)
    logits_conv = converted_model(input_ids)

    # Crop vocab if needed
    min_vocab = min(logits_orig.shape[-1], logits_conv.shape[-1])
    logits_orig = logits_orig[..., :min_vocab]
    logits_conv = logits_conv[..., :min_vocab]

    # Dummy loss: mean of all logits (simple, deterministic)
    loss_orig = logits_orig.sum()
    loss_conv = logits_conv.sum()

    loss_orig.backward()
    loss_conv.backward()

    # Compare gradients for all named parameters
    orig_grads = {n: p.grad for n, p in original_model.named_parameters() if p.grad is not None}
    conv_grads = {n: p.grad for n, p in converted_model.named_parameters() if p.grad is not None}

    details = []
    all_passed = True

    # Try to match parameters by name (may differ between original and converted)
    # Strategy: match by shape + position if names don't match
    if set(orig_grads.keys()) == set(conv_grads.keys()):
        # Names match — compare directly
        for name in sorted(orig_grads.keys()):
            result = compare_tensors(orig_grads[name], conv_grads[name], f"grad/{name}", atol, rtol)
            details.append(result)
            if not result["passed"]:
                all_passed = False
    else:
        # Names don't match — compare by shape groups
        orig_by_shape = {}
        for n, g in orig_grads.items():
            orig_by_shape.setdefault(g.shape, []).append((n, g))
        conv_by_shape = {}
        for n, g in conv_grads.items():
            conv_by_shape.setdefault(g.shape, []).append((n, g))

        matched = 0
        for shape in orig_by_shape:
            if shape not in conv_by_shape:
                details.append({"name": f"shape {shape}", "passed": False, "reason": "No matching shape in converted"})
                all_passed = False
                continue
            orig_list = orig_by_shape[shape]
            conv_list = conv_by_shape[shape]
            for i, ((on, og), (cn, cg)) in enumerate(zip(orig_list, conv_list)):
                result = compare_tensors(og, cg, f"grad/{on} <-> {cn}", atol, rtol)
                details.append(result)
                if not result["passed"]:
                    all_passed = False
                matched += 1

        details.append({
            "name": "_summary",
            "matched_params": matched,
            "orig_params": len(orig_grads),
            "conv_params": len(conv_grads),
            "passed": True,
        })

    return {
        "stage": "backward",
        "passed": all_passed,
        "n_orig_grads": len(orig_grads),
        "n_conv_grads": len(conv_grads),
        "details": details,
    }


def verify_optimizer_step(
    original_model: nn.Module,
    converted_model: nn.Module,
    original_optimizer,
    converted_optimizer,
    input_ids: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> dict:
    """
    Verify that one optimizer step produces the same parameter update.
    """
    original_model.train()
    converted_model.train()

    # Save initial parameters
    orig_params_before = {n: p.data.clone() for n, p in original_model.named_parameters()}
    conv_params_before = {n: p.data.clone() for n, p in converted_model.named_parameters()}

    # Forward + backward
    original_model.zero_grad()
    converted_model.zero_grad()

    logits_orig = original_model(input_ids)
    logits_conv = converted_model(input_ids)

    min_vocab = min(logits_orig.shape[-1], logits_conv.shape[-1])
    loss_orig = logits_orig[..., :min_vocab].sum()
    loss_conv = logits_conv[..., :min_vocab].sum()

    loss_orig.backward()
    loss_conv.backward()

    # Optimizer step
    # Handle different step() signatures
    try:
        original_optimizer.step()
    except TypeError:
        original_optimizer.step(step_idx=0)

    try:
        converted_optimizer.step(step_idx=0)
    except TypeError:
        converted_optimizer.step()

    # Compare parameter deltas
    details = []
    all_passed = True

    orig_deltas = {n: p.data - orig_params_before[n] for n, p in original_model.named_parameters()}
    conv_deltas = {n: p.data - conv_params_before[n] for n, p in converted_model.named_parameters()}

    if set(orig_deltas.keys()) == set(conv_deltas.keys()):
        for name in sorted(orig_deltas.keys()):
            result = compare_tensors(orig_deltas[name], conv_deltas[name], f"delta/{name}", atol, rtol)
            details.append(result)
            if not result["passed"]:
                all_passed = False
    else:
        # Match by shape
        orig_by_shape = {}
        for n, d in orig_deltas.items():
            orig_by_shape.setdefault(d.shape, []).append((n, d))
        conv_by_shape = {}
        for n, d in conv_deltas.items():
            conv_by_shape.setdefault(d.shape, []).append((n, d))

        for shape in orig_by_shape:
            if shape not in conv_by_shape:
                details.append({"name": f"delta shape {shape}", "passed": False, "reason": "Missing"})
                all_passed = False
                continue
            for (on, od), (cn, cd) in zip(orig_by_shape[shape], conv_by_shape[shape]):
                result = compare_tensors(od, cd, f"delta/{on} <-> {cn}", atol, rtol)
                details.append(result)
                if not result["passed"]:
                    all_passed = False

    return {
        "stage": "optimizer_step",
        "passed": all_passed,
        "details": details,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify numerical equivalence of converted code")
    parser.add_argument("--original-model", required=True, help="Path to original model .py file")
    parser.add_argument("--converted-model", required=True, help="Path to converted model.py")
    parser.add_argument("--config", default=None, help="Path to converted config.json (for reference)")
    parser.add_argument("--original-optimizer", default=None, help="Path to original optimizer .py")
    parser.add_argument("--converted-optimizer", default=None, help="Path to converted optimizer.py")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")  # CPU for reproducibility

    print("=" * 70)
    print("  LossCast-Bench — Numerical Equivalence Verification")
    print("=" * 70)

    # Load modules
    print(f"\n  Loading original model:   {args.original_model}")
    orig_mod = load_module(args.original_model, "original_model")
    print(f"  Loading converted model:  {args.converted_model}")
    conv_mod = load_module(args.converted_model, "converted_model")

    # Build models
    print("\n  Building models...")
    orig_model, orig_config = build_original_model(orig_mod)
    conv_model, conv_config = build_converted_model(conv_mod)
    orig_model.to(device).float()
    conv_model.to(device).float()

    # Parameter count comparison
    orig_params = sum(p.numel() for p in orig_model.parameters())
    conv_params = sum(p.numel() for p in conv_model.parameters())
    print(f"  Original params:  {orig_params:,}")
    print(f"  Converted params: {conv_params:,}")
    if orig_params != conv_params:
        print(f"  WARNING: Parameter count mismatch ({abs(orig_params - conv_params):,} difference)")

    # Generate dummy input
    vocab_size = conv_config.get("vocab_size", 32000) if isinstance(conv_config, dict) else getattr(conv_config, "vocab_size", 32000)
    torch.manual_seed(args.seed)
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)

    results = []

    # ── Stage 1: Forward pass ──
    print("\n" + "-" * 70)
    print("  Stage 1: Forward Pass Equivalence")
    print("-" * 70)
    fwd_result = verify_forward(orig_model, conv_model, input_ids, args.atol, args.rtol)
    results.append(fwd_result)
    if fwd_result["passed"]:
        detail = fwd_result["details"][0]
        print(f"  PASSED  (max_abs_diff={detail['max_abs_diff']:.2e}, max_rel_diff={detail['max_rel_diff']:.2e})")
    else:
        print(f"  FAILED")
        for d in fwd_result["details"]:
            reason = d.get("reason", f"max_abs_diff={d.get('max_abs_diff', '?'):.2e}")
            print(f"    {d['name']}: {reason}")

    # ── Stage 2: Backward pass ──
    print("\n" + "-" * 70)
    print("  Stage 2: Backward Pass (Gradient) Equivalence")
    print("-" * 70)
    bwd_result = verify_backward(orig_model, conv_model, input_ids, args.atol, args.rtol)
    results.append(bwd_result)
    n_passed = sum(1 for d in bwd_result["details"] if d.get("passed", False))
    n_total = len([d for d in bwd_result["details"] if d.get("name", "").startswith("grad/")])
    if bwd_result["passed"]:
        print(f"  PASSED  ({n_passed}/{n_total} gradient tensors match)")
    else:
        print(f"  FAILED  ({n_passed}/{n_total} gradient tensors match)")
        for d in bwd_result["details"]:
            if not d.get("passed", True) and d["name"] != "_summary":
                reason = d.get("reason", f"max_abs_diff={d.get('max_abs_diff', '?'):.2e}")
                print(f"    {d['name']}: {reason}")

    # ── Stage 3: Optimizer step (if both optimizer files provided) ──
    if args.original_optimizer and args.converted_optimizer:
        print("\n" + "-" * 70)
        print("  Stage 3: Optimizer Step Equivalence")
        print("-" * 70)

        orig_opt_mod = load_module(args.original_optimizer, "original_optimizer")
        conv_opt_mod = load_module(args.converted_optimizer, "converted_optimizer")

        # Re-build models fresh (gradients got consumed)
        torch.manual_seed(args.seed)
        orig_model2, _ = build_original_model(orig_mod)
        conv_model2, _ = build_converted_model(conv_mod)
        orig_model2.to(device).float()
        conv_model2.to(device).float()

        orig_opt = build_original_optimizer(orig_opt_mod, orig_model2.parameters())
        conv_opt = build_converted_optimizer(conv_opt_mod, conv_model2.parameters())

        opt_result = verify_optimizer_step(
            orig_model2, conv_model2, orig_opt, conv_opt,
            input_ids, atol=args.atol * 10, rtol=args.rtol * 10,  # looser tolerance for optimizer
        )
        results.append(opt_result)
        n_passed = sum(1 for d in opt_result["details"] if d.get("passed", False))
        n_total = len(opt_result["details"])
        if opt_result["passed"]:
            print(f"  PASSED  ({n_passed}/{n_total} parameter deltas match)")
        else:
            print(f"  FAILED  ({n_passed}/{n_total} parameter deltas match)")
            for d in opt_result["details"]:
                if not d.get("passed", True):
                    reason = d.get("reason", f"max_abs_diff={d.get('max_abs_diff', '?'):.2e}")
                    print(f"    {d['name']}: {reason}")
    elif args.converted_optimizer:
        print("\n  Skipping optimizer step verification (no --original-optimizer provided)")
    else:
        print("\n  Skipping optimizer step verification (no optimizer files provided)")

    # ── Summary ──
    print("\n" + "=" * 70)
    all_passed = all(r["passed"] for r in results)
    if all_passed:
        print("  ALL STAGES PASSED")
    else:
        failed = [r["stage"] for r in results if not r["passed"]]
        print(f"  FAILED STAGES: {', '.join(failed)}")
    print("=" * 70)

    if args.json:
        print("\n" + json.dumps(results, indent=2, default=str))

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
