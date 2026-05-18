#!/usr/bin/env python3
"""
LossCast-Bench — Nanochat-specific equivalence verification.

Nanochat has a non-standard model interface:
  - Model is built on meta device, then moved to real device, then init_weights()
  - Optimizer is created via model.setup_optimizer() (hybrid Muon+AdamW)
  - Forward pass signature: forward(idx, targets=None, ...)
  - Uses custom Linear class that casts weights to match input dtype

This script handles all of that and compares against a converted losscast-bench
model.py and optimizer.py.

Usage:
  python verify_nanochat.py \
    --nanochat-dir /path/to/nanochat-source \
    --converted-dir /path/to/converted/outputs \
    [--seq-len 128] \
    [--batch-size 2] \
    [--seed 42]

All nanochat model parameters (n_layers, d_model, n_heads, vocab_size, etc.)
are read from the converted config.json — no need to specify depth manually.
"""

import argparse
import importlib.util
import json
import math
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_nanochat_model(nanochat_dir: str, converted_config: dict):
    """Build nanochat GPT model using parameters from converted config.json.

    The converted config.json already has the fully resolved n_layers, d_model,
    n_heads, vocab_size, etc. We use these directly to build the GPTConfig,
    so there's no need to re-derive them from depth/aspect_ratio.
    """
    # We need to set up the nanochat package so imports work
    sys.path.insert(0, nanochat_dir)

    # Patch: nanochat's common.py tries to init distributed, we just need the basics
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    from nanochat.gpt import GPT, GPTConfig

    model_cfg = converted_config["model"]
    schedule_cfg = converted_config.get("schedule", {})
    seq_len = converted_config.get("data", {}).get("seq_len", 2048)

    config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=model_cfg["vocab_size"],
        n_layer=model_cfg["n_layers"],
        n_head=model_cfg["n_heads"],
        n_kv_head=model_cfg.get("n_kv_heads") or model_cfg["n_heads"],
        n_embd=model_cfg["d_model"],
        window_pattern="SSSL",  # nanochat default
    )

    # Build on meta device, move to CPU, init weights (matching base_train.py)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=torch.device("cpu"))

    # Override COMPUTE_DTYPE to float32 for comparison
    import nanochat.common as common
    original_dtype = common.COMPUTE_DTYPE
    common.COMPUTE_DTYPE = torch.float32
    model.init_weights()
    common.COMPUTE_DTYPE = original_dtype

    return model, config


def load_converted_config(converted_dir: str) -> dict:
    """Load config.json from the converted output directory."""
    config_path = os.path.join(converted_dir, "config.json")
    with open(config_path) as f:
        return json.load(f)


def build_converted_model(converted_dir: str):
    """Build the converted losscast-bench model."""
    model_path = os.path.join(converted_dir, "model.py")
    mod = load_module(model_path, "converted_model")
    model = mod.Model(mod.MODEL_CONFIG)
    return model.float(), mod.MODEL_CONFIG


def build_nanochat_optimizer(model):
    """Build nanochat's optimizer, but single-GPU version."""
    optimizer = model.setup_optimizer(
        unembedding_lr=0.008,
        embedding_lr=0.3,
        matrix_lr=0.02,
        weight_decay=0.28,
        scalar_lr=0.5,
    )
    return optimizer


def build_converted_optimizer(converted_dir: str, model_params):
    """Build the converted optimizer."""
    optim_path = os.path.join(converted_dir, "optimizer.py")
    if not os.path.exists(optim_path):
        return None
    mod = load_module(optim_path, "converted_optimizer")
    defaults = dict(mod.OPTIMIZER_DEFAULTS)
    defaults.pop("name", None)
    return mod.CustomOptimizer(model_params, **defaults)


def compare_tensors(a, b, name, atol=1e-5, rtol=1e-4):
    if a.shape != b.shape:
        return {"name": name, "passed": False, "reason": f"shape {a.shape} vs {b.shape}"}
    abs_diff = (a.float() - b.float()).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    passed = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    return {
        "name": name,
        "passed": passed,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify nanochat conversion equivalence")
    parser.add_argument("--nanochat-dir", required=True, help="Path to nanochat source directory")
    parser.add_argument("--converted-dir", required=True, help="Path to converted outputs (must contain config.json + model.py)")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for dummy input")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    # Read model params from the converted config.json
    converted_config = load_converted_config(args.converted_dir)
    model_cfg = converted_config["model"]
    n_layers = model_cfg["n_layers"]
    d_model = model_cfg["d_model"]
    n_heads = model_cfg["n_heads"]
    vocab_size = model_cfg["vocab_size"]

    print("=" * 70)
    print("  LossCast-Bench — Nanochat Equivalence Verification")
    print(f"  n_layers={n_layers}, d_model={d_model}, n_heads={n_heads}, vocab={vocab_size}")
    print(f"  seq_len={args.seq_len}, batch_size={args.batch_size}")
    print("=" * 70)

    # ── Build both models with the same seed ──
    torch.manual_seed(args.seed)
    print("\n  Building nanochat model...")
    try:
        nc_model, nc_config = build_nanochat_model(args.nanochat_dir, converted_config)
        nc_model.float()
    except Exception as e:
        print(f"  ERROR building nanochat model: {e}")
        print("  This is expected if nanochat dependencies (flash_attn, etc.) are not installed.")
        print("  In that case, use verify_equivalence.py with a standalone original model file instead.")
        return 1

    torch.manual_seed(args.seed)
    print("  Building converted model...")
    conv_model, conv_config = build_converted_model(args.converted_dir)

    nc_params = sum(p.numel() for p in nc_model.parameters())
    conv_params = sum(p.numel() for p in conv_model.parameters())
    print(f"\n  Nanochat params:  {nc_params:,}")
    print(f"  Converted params: {conv_params:,}")
    if nc_params != conv_params:
        print(f"  NOTE: Param count differs by {abs(nc_params - conv_params):,}")
        print(f"        (Nanochat has extra components: value_embeds, smear, backout, per-layer scalars)")

    # ── Generate dummy input ──
    vocab_size = nc_config.vocab_size if hasattr(nc_config, "vocab_size") else 32768
    torch.manual_seed(args.seed)
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len))

    all_results = []

    # ── Stage 1: Forward pass ──
    print("\n" + "-" * 70)
    print("  Stage 1: Forward Pass")
    print("-" * 70)

    nc_model.eval()
    conv_model.eval()

    with torch.no_grad():
        # nanochat forward: returns logits when targets=None
        logits_nc = nc_model(input_ids)
        logits_conv = conv_model(input_ids)

    # Crop to same vocab size (nanochat pads vocab to multiple of 64)
    min_vocab = min(logits_nc.shape[-1], logits_conv.shape[-1])
    logits_nc = logits_nc[..., :min_vocab]
    logits_conv = logits_conv[..., :min_vocab]

    fwd = compare_tensors(logits_nc, logits_conv, "logits", args.atol, args.rtol)
    all_results.append({"stage": "forward", **fwd})

    if fwd["passed"]:
        print(f"  PASSED  (max_abs_diff={fwd['max_abs_diff']:.2e})")
    else:
        print(f"  FAILED  (max_abs_diff={fwd['max_abs_diff']:.2e})")
        print(f"          This might be expected if the converted model simplified")
        print(f"          some nanochat-specific features (smear, backout, etc.)")

    # ── Stage 2: Backward pass ──
    print("\n" + "-" * 70)
    print("  Stage 2: Backward Pass (Gradients)")
    print("-" * 70)

    nc_model.train()
    conv_model.train()
    nc_model.zero_grad()
    conv_model.zero_grad()

    logits_nc = nc_model(input_ids)
    logits_conv = conv_model(input_ids)
    min_vocab = min(logits_nc.shape[-1], logits_conv.shape[-1])

    loss_nc = logits_nc[..., :min_vocab].sum()
    loss_conv = logits_conv[..., :min_vocab].sum()
    loss_nc.backward()
    loss_conv.backward()

    nc_grads = [(n, p.grad) for n, p in nc_model.named_parameters() if p.grad is not None]
    conv_grads = [(n, p.grad) for n, p in conv_model.named_parameters() if p.grad is not None]

    print(f"  Nanochat has {len(nc_grads)} gradient tensors")
    print(f"  Converted has {len(conv_grads)} gradient tensors")

    # Group by shape for matching
    nc_by_shape = {}
    for n, g in nc_grads:
        nc_by_shape.setdefault(g.shape, []).append((n, g))
    conv_by_shape = {}
    for n, g in conv_grads:
        conv_by_shape.setdefault(g.shape, []).append((n, g))

    grad_results = []
    matched = 0
    for shape in sorted(nc_by_shape.keys(), key=lambda s: (-len(s), s)):
        if shape not in conv_by_shape:
            print(f"  SKIP shape {shape}: no match in converted model")
            continue
        for (nn_, ng), (cn, cg) in zip(nc_by_shape[shape], conv_by_shape[shape]):
            r = compare_tensors(ng, cg, f"{nn_} <-> {cn}", args.atol, args.rtol)
            grad_results.append(r)
            matched += 1

    passed_grads = sum(1 for r in grad_results if r["passed"])
    print(f"  Matched {matched} tensor pairs: {passed_grads}/{matched} passed")
    if not all(r["passed"] for r in grad_results):
        for r in grad_results:
            if not r["passed"]:
                print(f"    FAIL: {r['name']} (max_abs_diff={r['max_abs_diff']:.2e})")

    all_results.append({
        "stage": "backward",
        "passed": all(r["passed"] for r in grad_results) if grad_results else False,
        "matched": matched,
        "details": grad_results,
    })

    # ── Stage 3: Optimizer step ──
    optim_path = os.path.join(args.converted_dir, "optimizer.py")
    if os.path.exists(optim_path):
        print("\n" + "-" * 70)
        print("  Stage 3: Optimizer Step")
        print("-" * 70)

        # Re-build both models fresh
        torch.manual_seed(args.seed)
        nc_model2, _ = build_nanochat_model(args.nanochat_dir, converted_config)
        nc_model2.float()
        torch.manual_seed(args.seed)
        conv_model2, _ = build_converted_model(args.converted_dir)

        # Save initial params
        nc_params_before = {n: p.data.clone() for n, p in nc_model2.named_parameters()}
        conv_params_before = {n: p.data.clone() for n, p in conv_model2.named_parameters()}

        # Build optimizers
        nc_optimizer = build_nanochat_optimizer(nc_model2)
        conv_optimizer = build_converted_optimizer(args.converted_dir, conv_model2.parameters())

        if conv_optimizer is None:
            print("  SKIP: No converted optimizer.py found")
        else:
            # Forward + backward
            nc_model2.train()
            conv_model2.train()
            nc_model2.zero_grad()
            conv_model2.zero_grad()

            torch.manual_seed(args.seed)
            input_ids2 = torch.randint(0, vocab_size, (args.batch_size, args.seq_len))

            logits_nc2 = nc_model2(input_ids2)
            logits_conv2 = conv_model2(input_ids2)
            min_v = min(logits_nc2.shape[-1], logits_conv2.shape[-1])
            logits_nc2[..., :min_v].sum().backward()
            logits_conv2[..., :min_v].sum().backward()

            # Optimizer step
            nc_optimizer.step()
            try:
                conv_optimizer.step(step_idx=0)
            except TypeError:
                conv_optimizer.step()

            # Compare deltas
            nc_deltas = {n: p.data - nc_params_before[n] for n, p in nc_model2.named_parameters()}
            conv_deltas = {n: p.data - conv_params_before[n] for n, p in conv_model2.named_parameters()}

            # Group by shape
            nc_d_by_shape = {}
            for n, d in nc_deltas.items():
                if d.abs().max() > 1e-12:  # only non-zero deltas
                    nc_d_by_shape.setdefault(d.shape, []).append((n, d))
            conv_d_by_shape = {}
            for n, d in conv_deltas.items():
                if d.abs().max() > 1e-12:
                    conv_d_by_shape.setdefault(d.shape, []).append((n, d))

            opt_results = []
            for shape in sorted(nc_d_by_shape.keys(), key=lambda s: (-len(s), s)):
                if shape not in conv_d_by_shape:
                    continue
                for (nn_, nd), (cn, cd) in zip(nc_d_by_shape[shape], conv_d_by_shape[shape]):
                    r = compare_tensors(nd, cd, f"{nn_} <-> {cn}", args.atol * 10, args.rtol * 10)
                    opt_results.append(r)

            passed_opt = sum(1 for r in opt_results if r["passed"])
            print(f"  Matched {len(opt_results)} delta pairs: {passed_opt}/{len(opt_results)} passed")
            if not all(r["passed"] for r in opt_results):
                for r in opt_results:
                    if not r["passed"]:
                        print(f"    FAIL: {r['name']} (max_abs_diff={r['max_abs_diff']:.2e})")

            all_results.append({
                "stage": "optimizer_step",
                "passed": all(r["passed"] for r in opt_results) if opt_results else False,
                "details": opt_results,
            })
    else:
        print("\n  Skipping optimizer step verification (no optimizer.py in converted dir)")

    # ── Summary ──
    print("\n" + "=" * 70)
    stages_passed = [r for r in all_results if r.get("passed", False)]
    stages_failed = [r for r in all_results if not r.get("passed", True)]
    if not stages_failed:
        print("  ALL STAGES PASSED")
    else:
        print(f"  PASSED: {', '.join(r['stage'] for r in stages_passed) or 'none'}")
        print(f"  FAILED: {', '.join(r['stage'] for r in stages_failed)}")
    print("=" * 70)

    return 0 if not stages_failed else 1


if __name__ == "__main__":
    sys.exit(main())
