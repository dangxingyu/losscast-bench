#!/usr/bin/env python3
"""
Convert nanochat training runs to losscast-bench format.

Nanochat code is already clean PyTorch — no code-rewriter needed. This script:
  1. Parses training config from WandB / JSON / CLI args → config.json
  2. Copies raw gpt.py → model.py, optim.py → optimizer.py (as-is)
  3. Extracts loss curves from WandB or local logs → losses.json

Usage:
    # From a WandB run (config + losses + code):
    python losscast-dev/convert/nanochat_to_bench.py \\
        --wandb-run nanochat/nanochat/abc123 \\
        --nanochat-dir ~/nanochat \\
        --output data/train/nanochat_d12_v1

    # From a JSON config file:
    python losscast-dev/convert/nanochat_to_bench.py \\
        --config-json wandb_config.json \\
        --losses training_log.json \\
        --nanochat-dir ~/nanochat \\
        --output data/train/nanochat_d12_v1

    # Dry-run (preview config.json without writing):
    python losscast-dev/convert/nanochat_to_bench.py \\
        --config-json '{"depth": 12, "aspect_ratio": 64}' \\
        --output /tmp/test --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ── Nanochat architecture derivation ────────────────────────────────────────
# Mirrors the logic in nanochat/scripts/base_train.py: build_model_meta()

def derive_model_dim(depth: int, aspect_ratio: int, head_dim: int) -> tuple[int, int]:
    """Compute d_model and n_heads from nanochat's depth/aspect_ratio/head_dim.

    nanochat nudges d_model up to the nearest multiple of head_dim so that
    n_heads = d_model / head_dim is an integer. This matches build_model_meta().
    """
    base_dim = depth * aspect_ratio
    d_model = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_heads = d_model // head_dim
    return d_model, n_heads


# ── Config arg key normalization ────────────────────────────────────────────
# WandB configs use hyphens (depth, max-seq-len), Python uses underscores.
# Nanochat CLI uses hyphens, WandB stores them as-is or underscored depending
# on version. We normalize to underscores.

def normalize_keys(d: dict) -> dict:
    """Normalize config keys: replace hyphens with underscores."""
    return {k.replace("-", "_"): v for k, v in d.items()}


# ── Main conversion logic ──────────────────────────────────────────────────

def nanochat_args_to_config(nc: dict, run_id: str) -> dict:
    """Convert nanochat training args → losscast-bench config.json.

    Args:
        nc: Dict of nanochat training args (from WandB config, JSON, or CLI).
            Keys should use underscores (e.g., 'device_batch_size').
        run_id: Unique run identifier for the benchmark.

    Returns:
        A config dict matching the losscast-bench schema.
    """
    nc = normalize_keys(nc)

    # ── Model architecture ──────────────────────────────────────────────
    depth = int(nc.get("depth", 12))
    aspect_ratio = int(nc.get("aspect_ratio", 64))
    head_dim = int(nc.get("head_dim", 128))
    seq_len = int(nc.get("max_seq_len", 2048))
    vocab_size = int(nc.get("vocab_size", 32768))

    d_model, n_heads = derive_model_dim(depth, aspect_ratio, head_dim)

    # ── Batch / data ────────────────────────────────────────────────────
    device_batch_size = int(nc.get("device_batch_size", 32))
    world_size = int(nc.get("world_size", 8))
    batch_tokens = device_batch_size * seq_len * world_size

    # ── Training duration ───────────────────────────────────────────────
    num_iterations = int(nc.get("num_iterations", -1))
    total_steps = num_iterations if num_iterations > 0 else None

    # ── LR schedule ─────────────────────────────────────────────────────
    # nanochat: linear warmup → constant → linear warmdown
    warmup_steps = int(nc.get("warmup_steps", 40))
    warmdown_ratio = float(nc.get("warmdown_ratio", 0.65))
    final_lr_frac = float(nc.get("final_lr_frac", 0.05))
    cooldown_steps = int(total_steps * warmdown_ratio) if total_steps else None

    # ── Optimizer ───────────────────────────────────────────────────────
    # nanochat uses hybrid MuonAdamW:
    #   - Muon for transformer matrix params (attention, MLP weights)
    #   - AdamW for embeddings, lm_head, per-layer scalars
    # Each group has its own LR, betas, and weight decay.
    matrix_lr = float(nc.get("matrix_lr", 0.02))
    embedding_lr = float(nc.get("embedding_lr", 0.3))
    unembedding_lr = float(nc.get("unembedding_lr", 0.008))
    scalar_lr = float(nc.get("scalar_lr", 0.5))
    weight_decay = float(nc.get("weight_decay", 0.28))

    # AdamW LRs are scaled by (d_model / 768)^-0.5 in nanochat
    dmodel_lr_scale = (d_model / 768) ** -0.5

    return {
        "run_id": run_id,
        "model": {
            "arch": "transformer",
            "n_layers": depth,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_kv_heads": n_heads,  # nanochat defaults to MHA
            "head_dim": head_dim,
            "d_ff": 4 * d_model,  # nanochat MLP: c_fc is 4 * n_embd
            "vocab_size": vocab_size,
            "activation": "relu2",
            "norm_type": "rmsnorm",
            "rope": True,
            "tied_embeddings": False,
        },
        "optimizer": {
            "name": "muon_adamw",
            "lr": matrix_lr,
            "weight_decay": weight_decay,
            "beta1": 0.95,   # Muon momentum
            "beta2": 0.9,    # Muon beta2
            "eps": 1e-10,
            "grad_clip": None,
            "extra": {
                "matrix_lr": matrix_lr,
                "embedding_lr": embedding_lr * dmodel_lr_scale,
                "unembedding_lr": unembedding_lr * dmodel_lr_scale,
                "scalar_lr": scalar_lr,
                "muon_momentum": 0.95,
                "muon_ns_steps": 5,
                "dmodel_lr_scale": round(dmodel_lr_scale, 6),
            },
        },
        "data": {
            "dataset": "fineweb_edu",
            "tokenizer": "nanochat_bpe_32k",
            "eval_dataset": "fineweb_edu",
            "seq_len": seq_len,
            "batch_tokens": batch_tokens,
            "tokens_total": batch_tokens * total_steps if total_steps else None,
        },
        "schedule": {
            "lr_schedule": "warmup_stable_warmdown",
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "cooldown_steps": cooldown_steps,
            "final_lr_ratio": final_lr_frac,
        },
        "eval_interval": int(nc.get("eval_every", 250)),
        "precision": "fp8" if nc.get("fp8") else "bf16",
        "hardware": f"{world_size}xH100" if world_size > 1 else "H100",
        "dp_size": world_size,
    }


# ── WandB loss extraction ──────────────────────────────────────────────────

def extract_wandb_losses(run, eval_key: str = "val/bpb") -> dict[str, float]:
    """Extract eval loss curve from a WandB run.

    Args:
        run: A wandb.apis.public.Run object.
        eval_key: The metric key to extract (default: val/bpb for nanochat).

    Returns:
        Dict of {step_str: loss_value}.
    """
    history = run.history(keys=["_step", eval_key], samples=50000, pandas=False)

    losses = {}
    for row in history:
        step = row.get("_step")
        val = row.get(eval_key)
        if step is not None and val is not None:
            try:
                val = float(val)
                if math.isfinite(val):
                    losses[str(int(step))] = round(val, 6)
            except (ValueError, TypeError):
                continue

    if not losses:
        # Try alternative keys
        alt_keys = ["val/loss", "eval/loss", "val_loss", "train/loss"]
        for key in alt_keys:
            history = run.history(keys=["_step", key], samples=50000, pandas=False)
            for row in history:
                step = row.get("_step")
                val = row.get(key)
                if step is not None and val is not None:
                    try:
                        val = float(val)
                        if math.isfinite(val):
                            losses[str(int(step))] = round(val, 6)
                    except (ValueError, TypeError):
                        continue
            if losses:
                log.info(f"Using fallback loss key: {key}")
                break

    return losses


def load_wandb_run(run_path: str) -> tuple[dict, dict[str, float]]:
    """Load config and losses from a WandB run.

    Args:
        run_path: WandB run path like 'entity/project/run_id' or a full URL.

    Returns:
        (config_dict, losses_dict)
    """
    try:
        import wandb
    except ImportError:
        log.error("wandb not installed. Run: uv pip install wandb")
        sys.exit(1)

    # Handle full URLs
    if run_path.startswith("http"):
        # https://wandb.ai/entity/project/runs/run_id → entity/project/run_id
        parts = run_path.rstrip("/").split("/")
        run_path = f"{parts[-4]}/{parts[-3]}/{parts[-1]}"

    api = wandb.Api()
    run = api.run(run_path)
    config = normalize_keys(dict(run.config))
    losses = extract_wandb_losses(run)

    log.info(f"Loaded from WandB: {run_path}")
    log.info(f"  Config keys: {sorted(config.keys())}")
    log.info(f"  Loss points: {len(losses)}")

    return config, losses


# ── Local loss file loading ─────────────────────────────────────────────────

def load_local_losses(path: Path) -> dict[str, float]:
    """Load losses from a local file.

    Supports:
      - JSON dict: {"step": loss, ...}
      - JSON list: [{"step": N, "loss": L}, ...] or [{"_step": N, "val/bpb": L}, ...]
      - JSONL: one JSON object per line
    """
    text = path.read_text().strip()
    data = json.loads(text) if not text.startswith("{") or "\n" not in text else None

    if data is None:
        # Try JSONL
        lines = text.split("\n")
        data = [json.loads(line) for line in lines if line.strip()]

    if isinstance(data, dict):
        # Already in {step: loss} format
        return {str(k): round(float(v), 6) for k, v in data.items()
                if v is not None and math.isfinite(float(v))}

    if isinstance(data, list):
        losses = {}
        for row in data:
            # Try common key patterns
            step = row.get("step") or row.get("_step") or row.get("iteration")
            loss = (row.get("val/bpb") or row.get("val/loss") or row.get("eval/loss")
                    or row.get("loss") or row.get("val_loss"))
            if step is not None and loss is not None:
                try:
                    loss = float(loss)
                    if math.isfinite(loss):
                        losses[str(int(step))] = round(loss, 6)
                except (ValueError, TypeError):
                    continue
        return losses

    log.warning(f"Could not parse losses from {path}")
    return {}


# ── Code copying ───────────────────────────────────────────────────────────

def copy_nanochat_code(
    nanochat_dir: Path,
    output_dir: Path,
    git_ref: Optional[str] = None,
) -> list[str]:
    """Copy raw nanochat source files as model.py and optimizer.py.

    Args:
        nanochat_dir: Path to nanochat repo root.
        output_dir: Destination directory.
        git_ref: Optional git ref (commit, tag) to checkout before copying.

    Returns:
        List of files copied.
    """
    if git_ref:
        result = subprocess.run(
            ["git", "-C", str(nanochat_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        original_ref = result.stdout.strip()

        subprocess.run(
            ["git", "-C", str(nanochat_dir), "checkout", git_ref],
            capture_output=True, text=True, check=True,
        )
        log.info(f"Checked out {git_ref} (was {original_ref[:8]})")

    copied = []
    file_map = {
        "nanochat/gpt.py": "model.py",
        "nanochat/optim.py": "optimizer.py",
    }

    for src_rel, dst_name in file_map.items():
        src = nanochat_dir / src_rel
        if src.exists():
            shutil.copy2(src, output_dir / dst_name)
            copied.append(dst_name)
            log.info(f"  {src_rel} → {dst_name}")
        else:
            log.warning(f"  {src_rel} not found, skipping")

    # Restore original ref if we changed it
    if git_ref and original_ref:
        subprocess.run(
            ["git", "-C", str(nanochat_dir), "checkout", original_ref],
            capture_output=True, text=True,
        )

    return copied


# ── CLI arg parsing ─────────────────────────────────────────────────────────

def parse_kv_args(args_str: str) -> dict:
    """Parse 'key=value key2=value2' string into a dict with typed values."""
    result = {}
    for token in args_str.split():
        if "=" not in token:
            log.warning(f"Skipping malformed arg (no '='): {token}")
            continue
        k, v = token.split("=", 1)
        k = k.replace("-", "_")
        # Type inference
        if v.lower() in ("true", "false"):
            result[k] = v.lower() == "true"
        else:
            for cast in (int, float):
                try:
                    result[k] = cast(v)
                    break
                except ValueError:
                    continue
            else:
                result[k] = v
    return result


def load_config_source(args) -> tuple[dict, Optional[dict]]:
    """Load nanochat config from the specified source.

    Returns:
        (nc_args, losses_or_none)
    """
    losses = None

    if args.wandb_run:
        nc_args, losses = load_wandb_run(args.wandb_run)
        return nc_args, losses

    if args.config_json:
        path = Path(args.config_json)
        if path.exists():
            nc_args = json.loads(path.read_text())
        else:
            nc_args = json.loads(args.config_json)
        return normalize_keys(nc_args), None

    if args.args:
        return parse_kv_args(args.args), None

    log.error("No config source provided. Use --wandb-run, --config-json, or --args.")
    sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert nanochat training runs to losscast-bench format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From WandB:
  %(prog)s --wandb-run nanochat/nanochat/abc123 --nanochat-dir ~/nanochat -o data/train/nc_d12

  # From local config + losses:
  %(prog)s --config-json config.json --losses log.json --nanochat-dir ~/nanochat -o data/train/nc_d12

  # Quick preview:
  %(prog)s --args "depth=12 aspect_ratio=64 num_iterations=5000" -o /tmp/test --dry-run
        """,
    )

    parser.add_argument("-o", "--output", required=True,
                        help="Output directory (e.g., data/train/nanochat_d12_v1)")
    parser.add_argument("--run-id",
                        help="Run ID (default: derived from output directory name)")

    # Config sources (pick one)
    src = parser.add_argument_group("config source (pick one)")
    src.add_argument("--wandb-run",
                     help="WandB run path or URL (e.g., nanochat/nanochat/abc123)")
    src.add_argument("--config-json",
                     help="JSON file or inline JSON string with nanochat args")
    src.add_argument("--args",
                     help="Space-separated key=value pairs (e.g., 'depth=12 matrix_lr=0.02')")

    # Code source
    code = parser.add_argument_group("code source")
    code.add_argument("--nanochat-dir",
                      help="Path to nanochat repo (copies gpt.py, optim.py)")
    code.add_argument("--git-ref",
                      help="Git ref to checkout before copying (commit hash, tag, branch)")

    # Loss curve
    parser.add_argument("--losses",
                        help="Path to local losses file (JSON dict, JSON list, or JSONL)")

    # Options
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview config.json without writing files")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    output = Path(args.output)
    run_id = args.run_id or output.name

    # Load config
    nc_args, wandb_losses = load_config_source(args)
    config = nanochat_args_to_config(nc_args, run_id)

    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    # Write output
    output.mkdir(parents=True, exist_ok=True)

    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"Wrote {output / 'config.json'}")

    # Losses: prefer local file, fall back to WandB-extracted
    losses = None
    if args.losses:
        losses = load_local_losses(Path(args.losses))
    elif wandb_losses:
        losses = wandb_losses

    if losses:
        with open(output / "losses.json", "w") as f:
            json.dump(losses, f, indent=2)
        log.info(f"Wrote {output / 'losses.json'} ({len(losses)} points)")
    else:
        log.warning("No losses available — provide --losses or --wandb-run")

    # Code
    if args.nanochat_dir:
        copied = copy_nanochat_code(
            Path(args.nanochat_dir), output, git_ref=args.git_ref,
        )
        if not copied:
            log.warning("No source files copied — check --nanochat-dir path")

    # Summary
    print(f"\n{'─' * 40}")
    print(f"Output: {output}/")
    print(f"  config.json  ✓")
    print(f"  losses.json  {'✓ ' + str(len(losses)) + ' points' if losses else '✗ missing'}")
    print(f"  model.py     {'✓' if (output / 'model.py').exists() else '✗ missing'}")
    print(f"  optimizer.py {'✓' if (output / 'optimizer.py').exists() else '✗ missing'}")

    m = config["model"]
    print(f"\n  Model: {m['n_layers']}L d={m['d_model']} h={m['n_heads']} "
          f"({m['d_ff']}ff) vocab={m['vocab_size']}")
    print(f"  Optimizer: {config['optimizer']['name']} lr={config['optimizer']['lr']}")
    if config["schedule"]["total_steps"]:
        print(f"  Schedule: {config['schedule']['total_steps']} steps, "
              f"{config['data']['tokens_total']/1e9:.1f}B tokens")


if __name__ == "__main__":
    main()
