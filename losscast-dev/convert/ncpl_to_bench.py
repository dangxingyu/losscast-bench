#!/usr/bin/env python3
"""
Convert NCPL HuggingFace dataset → losscast-bench per-run directories.

Usage:
    python losscast-dev/convert/ncpl_to_bench.py --source marin --dry-run
    python losscast-dev/convert/ncpl_to_bench.py --source steplaw --dry-run
    python losscast-dev/convert/ncpl_to_bench.py --source all --output data/staging/

Output: data/staging/{run_id}/config.json + losses.json per run.
Splits are NOT assigned here — see splits/build_splits.py for that.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("uv pip install datasets")
    exit(1)


# ── NCPL field names (have spaces, not underscores) ─────────────────────────
def conf_get(conf: dict, key: str, default=None):
    """Get from NCPL conf dict, trying both space and underscore variants."""
    return conf.get(key, conf.get(key.replace(" ", "_"), default))


# ── Marin architecture configs (from Fantastic Optimizers paper) ─────────────
# NCPL reports model_size in MB (e.g., 1208.8 MB ≈ 1.2B params).
# Keys are model_size in MB (rounded).
MARIN_ARCHS = {
    # ~130M params
    135: {"n_layers": 32, "d_model": 512, "n_heads": 8, "d_ff": 1365, "vocab_size": 32000},
    # ~300M params
    302: {"n_layers": 32, "d_model": 768, "n_heads": 12, "d_ff": 2048, "vocab_size": 32000},
    # ~520M params
    537: {"n_layers": 32, "d_model": 1024, "n_heads": 16, "d_ff": 2730, "vocab_size": 32000},
    # ~1.2B params
    1209: {"n_layers": 32, "d_model": 1536, "n_heads": 24, "d_ff": 4096, "vocab_size": 32000},
}

# Approx param count for each arch (for split logic)
MARIN_PARAMS = {135: 130e6, 302: 300e6, 537: 520e6, 1209: 1200e6}


def match_marin_arch(model_size_mb: float) -> tuple[dict, float]:
    """Match arch by model_size in MB. Returns (arch_dict, approx_n_params)."""
    size_mb = round(model_size_mb)
    best_key = min(MARIN_ARCHS.keys(), key=lambda k: abs(k - size_mb))
    if abs(best_key - size_mb) > 50:
        print(f"  WARNING: model_size={size_mb}MB, closest={best_key}MB")
    return MARIN_ARCHS[best_key].copy(), MARIN_PARAMS[best_key]


# ── StepLaw architecture configs ────────────────────────────────────────────
# StepLaw run names encode the actual arch: h{d_model}_ffnh{d_ff}_numh{n_heads}_numl{n_layers}
# So we parse from run name rather than using a lookup table.

def parse_steplaw_arch(name: str, conf: dict) -> dict:
    """Parse StepLaw architecture from run name + conf.

    Run name format: step2v2_0618_h1024_ffnh9552_numh16_numl8_lr..._bs..._ti..._mlr...
    NCPL conf has: num layers, num heads, hidden dim (which is actually d_ff!)
    """
    arch = {"vocab_size": 50257}  # GPT-2 tokenizer

    # Try parsing from run name first (most reliable)
    m = re.search(r'_h(\d+)_ffnh(\d+)_numh(\d+)_numl(\d+)', name)
    if m:
        arch["d_model"] = int(m.group(1))
        arch["d_ff"] = int(m.group(2))
        arch["n_heads"] = int(m.group(3))
        arch["n_layers"] = int(m.group(4))
    else:
        # Fallback to conf — but "hidden dim" in NCPL StepLaw is d_ff, not d_model!
        # We need to infer d_model from the run name or model_size.
        n_layers = conf_get(conf, "num layers", 12)
        n_heads = conf_get(conf, "num heads", 12)
        hidden_dim = conf_get(conf, "hidden dim", 1024)

        # For StepLaw, "hidden dim" appears to be d_ff based on the naming pattern
        # h1024_ffnh9552 → d_model=1024, d_ff=9552, but conf shows hidden_dim=9552
        # So hidden_dim in conf = d_ff. We need d_model from elsewhere.
        # Best guess: parse from run name h{d_model} part
        h_match = re.search(r'_h(\d+)', name)
        if h_match:
            arch["d_model"] = int(h_match.group(1))
            arch["d_ff"] = hidden_dim
        else:
            # Can't distinguish d_model from d_ff, use hidden_dim as d_model
            arch["d_model"] = hidden_dim
            arch["d_ff"] = 4 * hidden_dim

        arch["n_layers"] = n_layers
        arch["n_heads"] = n_heads

    return arch


def estimate_params_from_arch(arch: dict) -> float:
    """Rough parameter estimate from arch dict."""
    d = arch.get("d_model", 512)
    d_ff = arch.get("d_ff", 4 * d)
    n_layers = arch.get("n_layers", 12)
    n_heads = arch.get("n_heads", 8)
    vocab = arch.get("vocab_size", 32000)
    head_dim = d // n_heads
    attn = 4 * d * (n_heads * head_dim) * n_layers
    ffn = 2 * d * d_ff * n_layers  # StepLaw uses GELU (2 matrices)
    emb = vocab * d
    return attn + ffn + emb


def sanitize_run_id(name: str, source: str) -> str:
    """Create a clean run_id from NCPL run name."""
    clean = re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)
    clean = re.sub(r'_+', '_', clean).strip('_')
    return f"ncpl_{source}_{clean}"[:120]


def interpret_epsilon(eps_val) -> float:
    """NCPL epsilon field is weird (15, 8, etc.) — likely -log10(eps) or similar.

    Adam eps is typically 1e-8. NCPL stores 8 → 1e-8, 15 → 1e-15.
    """
    if eps_val is None:
        return 1e-8
    eps_val = float(eps_val)
    if eps_val > 1:
        # Likely stored as -log10(eps): 8 → 1e-8, 15 → 1e-15
        return 10 ** (-eps_val)
    return eps_val


def convert_marin_row(row: dict, idx: int) -> tuple[str, dict, dict] | None:
    """Convert one NCPL Marin row → (run_id, config_dict, losses_dict)."""
    conf = row["conf"]
    name = row["name"]
    run_id = sanitize_run_id(name, "marin")

    # Architecture — model_size is in MB
    model_size_mb = conf_get(conf, "model size", 0)
    arch, n_params = match_marin_arch(model_size_mb)

    # Losses: c4en_steps + c4en_losses
    steps = row.get("c4en_steps", [])
    losses = row.get("c4en_losses", [])
    if not steps or not losses or len(steps) != len(losses):
        return None

    # Filter NaN losses
    valid = [(s, l) for s, l in zip(steps, losses) if not math.isnan(l)]
    if len(valid) < 3:
        return None

    losses_dict = {str(s): round(l, 6) for s, l in valid}

    # Infer eval_interval from steps
    if len(valid) >= 2:
        intervals = [valid[i+1][0] - valid[i][0] for i in range(min(10, len(valid)-1))]
        eval_interval = max(set(intervals), key=intervals.count) if intervals else 1000
    else:
        eval_interval = 1000

    # LR schedule
    lr_schedule = conf_get(conf, "lr schedule", "cosine")
    if lr_schedule not in ("cosine", "linear", "wsd", "constant"):
        lr_schedule = "cosine"

    # Data size: NCPL stores in GB
    data_size_gb = conf_get(conf, "data size", 0)
    # Convert GB to tokens: ~4 bytes per token (rough), so GB * 1e9 / 4
    # But more accurately: Marin uses ~1048 tokens/KB for NeoX tokenizer
    # data_size_gb * 1e9 bytes ÷ ~1 byte/token ≈ tokens
    # Actually, NCPL "data size" might already be in tokens-equivalent GB
    # Let's compute from max_step * batch_size instead
    batch_size = conf_get(conf, "batch size", 256)
    seq_len = conf_get(conf, "block_size", 2048) or 2048
    max_step = row.get("max_step", 0)
    tokens_total = batch_size * seq_len * max_step

    config = {
        "run_id": run_id,
        "model": {
            "arch": "transformer",
            **arch,
            "activation": "swiglu",
            "norm_type": "rmsnorm",
            "rope": True,
        },
        "optimizer": {
            "name": conf_get(conf, "optimizer", "adamw"),
            "lr": conf_get(conf, "learning rate"),
            "weight_decay": conf_get(conf, "weight decay"),
            "beta1": conf_get(conf, "beta1", 0.9),
            "beta2": conf_get(conf, "beta2", 0.95),
            "eps": interpret_epsilon(conf_get(conf, "epsilon")),
            "grad_clip": conf_get(conf, "max_grad_norm", 1.0),
        },
        "data": {
            "dataset": "dolma",
            "tokenizer": "EleutherAI/gpt-neox-20b",
            "eval_dataset": "c4_en",
            "seq_len": seq_len,
            "batch_tokens": batch_size * seq_len,
            "tokens_total": tokens_total,
        },
        "schedule": {
            "lr_schedule": lr_schedule,
            "warmup_steps": conf_get(conf, "warmup", 0),
            "total_steps": max_step,
            "cooldown_steps": 0,
            "final_lr_ratio": conf_get(conf, "min_lr_ratio", 0.0),
        },
        "eval_interval": eval_interval,
        "precision": "bf16",
        "n_params": n_params,  # from arch lookup
    }

    # Extra optimizer fields for mudam/muon
    opt_name = conf_get(conf, "optimizer", "")
    if conf_get(conf, "adam_lr") is not None:
        config["optimizer"]["adam_lr"] = conf_get(conf, "adam_lr")
    if conf_get(conf, "preconditioner_lr") is not None:
        config["optimizer"]["preconditioner_lr"] = conf_get(conf, "preconditioner_lr")

    return run_id, config, losses_dict


def convert_steplaw_row(row: dict, idx: int) -> tuple[str, dict, dict] | None:
    """Convert one NCPL StepLaw row → (run_id, config_dict, losses_dict)."""
    conf = row["conf"]
    name = row["name"]
    run_id = sanitize_run_id(name, "steplaw")

    # Architecture — parse from run name
    arch = parse_steplaw_arch(name, conf)
    n_params = estimate_params_from_arch(arch)

    # Losses: smoothed_loss_steps + smoothed_loss
    steps = row.get("smoothed_loss_steps", [])
    losses = row.get("smoothed_loss", [])
    if not steps or not losses or len(steps) != len(losses):
        return None

    valid = [(s, l) for s, l in zip(steps, losses) if not math.isnan(l)]
    if len(valid) < 3:
        return None

    losses_dict = {str(s): round(l, 6) for s, l in valid}

    if len(valid) >= 2:
        intervals = [valid[i+1][0] - valid[i][0] for i in range(min(10, len(valid)-1))]
        eval_interval = max(set(intervals), key=intervals.count) if intervals else 1000
    else:
        eval_interval = 1000

    # LR schedule
    lr_schedule = conf_get(conf, "lr schedule", "cosine")

    # StepLaw batch_size is number of sequences, seq_len=1024
    seq_len = 1024
    batch_sequences = conf_get(conf, "batch size", 512)
    max_step = row.get("max_step", 0)
    batch_tokens = batch_sequences * seq_len
    tokens_total = batch_tokens * max_step

    # Cross-check with NCPL's data_size (in GB).
    # data_size ≈ tokens_total * 2 bytes / 1e9 (fp16 tokens)
    # This is a rough sanity check, not exact.
    data_size_gb = conf_get(conf, "data size", 0)

    config = {
        "run_id": run_id,
        "model": {
            "arch": "transformer",
            **arch,
            "activation": "gelu",
            "norm_type": "layernorm",
            "rope": False,
        },
        "optimizer": {
            "name": conf_get(conf, "optimizer", "adamw"),
            "lr": conf_get(conf, "learning rate"),
            "weight_decay": conf_get(conf, "weight decay"),
            "beta1": conf_get(conf, "beta1", 0.9),
            "beta2": conf_get(conf, "beta2", 0.95),
            "eps": interpret_epsilon(conf_get(conf, "epsilon")),
            "grad_clip": conf_get(conf, "max_grad_norm", 1.0),
        },
        "data": {
            "dataset": "openwebtext",
            "tokenizer": "gpt2",
            "eval_dataset": "train",
            "seq_len": seq_len,
            "batch_tokens": batch_tokens,
            "tokens_total": tokens_total,
        },
        "schedule": {
            "lr_schedule": lr_schedule,
            "warmup_steps": conf_get(conf, "warmup", 0),
            "total_steps": max_step,
            "cooldown_steps": 0,
            "final_lr_ratio": conf_get(conf, "min_lr_ratio", 0.0),
        },
        "eval_interval": eval_interval,
        "precision": "bf16",
        "n_params": n_params,
    }

    return run_id, config, losses_dict


def convert_source(source: str, output_dir: Path, dry_run: bool = False, min_points: int = 5):
    """Convert all runs from one NCPL source."""
    print(f"\n{'='*60}")
    print(f"Converting NCPL {source}")
    print(f"{'='*60}")

    ds = load_dataset("zhqwqwq/NCPL-Pretraining-Logs", source, split="train")
    print(f"Loaded {len(ds)} rows")

    converter = convert_marin_row if source == "marin" else convert_steplaw_row
    success, skipped, short = 0, 0, 0

    # Stats for dry-run summary
    opt_counts = Counter()
    size_counts = Counter()
    curve_lengths = []

    for i, row in enumerate(ds):
        result = converter(row, i)
        if result is None:
            skipped += 1
            continue

        run_id, config, losses = result

        # Filter short curves
        if len(losses) < min_points:
            short += 1
            continue

        opt_counts[config["optimizer"]["name"]] += 1
        size_counts[round(config.get("n_params", 0) / 1e6)] += 1
        curve_lengths.append(len(losses))

        if dry_run:
            if success < 5:
                print(f"\n  [{run_id}]")
                print(f"    model: {config['model']['n_layers']}L d={config['model']['d_model']} "
                      f"d_ff={config['model'].get('d_ff', '?')} ({config.get('n_params', 0)/1e6:.0f}M params)")
                print(f"    optimizer: {config['optimizer']['name']} lr={config['optimizer']['lr']} "
                      f"eps={config['optimizer']['eps']}")
                print(f"    data: {config['data']['tokens_total']/1e9:.1f}B tokens, "
                      f"batch={config['data']['batch_tokens']}")
                print(f"    schedule: {config['schedule']['lr_schedule']} "
                      f"warmup={config['schedule']['warmup_steps']} "
                      f"steps={config['schedule']['total_steps']}")
                print(f"    loss: {len(losses)} points, "
                      f"range [{min(losses.values()):.3f}, {max(losses.values()):.3f}]")
            success += 1
            continue

        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        with open(run_dir / "losses.json", "w") as f:
            json.dump(losses, f, indent=2)

        success += 1
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(ds)} processed ({success} ok)")

    print(f"\n--- Summary ---")
    print(f"Converted: {success}, Skipped (no data): {skipped}, Skipped (too short <{min_points}): {short}")
    print(f"Optimizers: {dict(opt_counts.most_common())}")
    print(f"Model sizes (M): {dict(sorted(size_counts.items()))}")
    if curve_lengths:
        curve_lengths.sort()
        print(f"Curve lengths: min={min(curve_lengths)}, median={curve_lengths[len(curve_lengths)//2]}, "
              f"max={max(curve_lengths)}")

    return success


def main():
    parser = argparse.ArgumentParser(description="Convert NCPL dataset to losscast-bench format")
    parser.add_argument("--source", choices=["marin", "steplaw", "all"], default="all")
    parser.add_argument("--output", type=str, default="data/staging/")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    parser.add_argument("--min-points", type=int, default=5, help="Min loss curve points to keep")
    args = parser.parse_args()

    output = Path(args.output)
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)

    sources = ["marin", "steplaw"] if args.source == "all" else [args.source]
    for source in sources:
        convert_source(source, output, dry_run=args.dry_run, min_points=args.min_points)


if __name__ == "__main__":
    main()
