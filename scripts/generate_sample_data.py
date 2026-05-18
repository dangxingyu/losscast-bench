#!/usr/bin/env python3
"""
Generate sample data for development/testing.
Creates synthetic per-run directories with config.json and losses.json.
"""

import json
import math
import random
from pathlib import Path

random.seed(42)

EVAL_INTERVAL = 500
TOTAL_STEPS = 50000

SAMPLE_RUNS = [
    {
        "run_id": "sample_125m_c4_cosine",
        "model": {"arch": "transformer", "n_layers": 12, "d_model": 768, "n_heads": 12, "vocab_size": 32000, "activation": "swiglu", "norm_type": "rmsnorm", "rope": True},
        "optimizer": {"name": "adamw", "lr": 3e-4, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "grad_clip": 1.0},
        "data": {"dataset": "c4", "tokenizer": "gpt2", "seq_len": 2048, "batch_tokens": 524288, "tokens_total": 10e9, "eval_dataset": "c4"},
        "schedule": {"lr_schedule": "cosine", "warmup_steps": 2000, "total_steps": TOTAL_STEPS, "final_lr_ratio": 0.1},
        "precision": "bf16",
    },
    {
        "run_id": "sample_350m_pile_wsd",
        "model": {"arch": "transformer", "n_layers": 24, "d_model": 1024, "n_heads": 16, "vocab_size": 50257, "activation": "gelu", "norm_type": "layernorm", "rope": True},
        "optimizer": {"name": "adamw", "lr": 1e-4, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "grad_clip": 1.0},
        "data": {"dataset": "pile", "tokenizer": "gpt2", "seq_len": 2048, "batch_tokens": 1048576, "tokens_total": 30e9, "eval_dataset": "pile"},
        "schedule": {"lr_schedule": "wsd", "warmup_steps": 3000, "total_steps": TOTAL_STEPS, "cooldown_steps": 5000, "final_lr_ratio": 0.0},
        "precision": "bf16",
    },
    {
        "run_id": "sample_1b_dolma_cosine",
        "model": {"arch": "transformer", "n_layers": 24, "d_model": 2048, "n_heads": 16, "n_kv_heads": 4, "vocab_size": 32000, "activation": "swiglu", "norm_type": "rmsnorm", "rope": True},
        "optimizer": {"name": "adamw", "lr": 5e-5, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "grad_clip": 1.0},
        "data": {"dataset": "dolma", "tokenizer": "gpt2", "seq_len": 4096, "batch_tokens": 2097152, "tokens_total": 100e9, "eval_dataset": "dolma"},
        "schedule": {"lr_schedule": "cosine", "warmup_steps": 5000, "total_steps": TOTAL_STEPS, "final_lr_ratio": 0.1},
        "precision": "bf16",
    },
    {
        "run_id": "sample_70m_refinedweb_linear",
        "model": {"arch": "transformer", "n_layers": 6, "d_model": 512, "n_heads": 8, "vocab_size": 32000, "activation": "swiglu", "norm_type": "rmsnorm", "rope": True},
        "optimizer": {"name": "adamw", "lr": 6e-4, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "grad_clip": 1.0},
        "data": {"dataset": "refinedweb", "tokenizer": "gpt2", "seq_len": 2048, "batch_tokens": 262144, "tokens_total": 5e9, "eval_dataset": "refinedweb"},
        "schedule": {"lr_schedule": "linear", "warmup_steps": 1000, "total_steps": TOTAL_STEPS, "final_lr_ratio": 0.0},
        "precision": "bf16",
    },
]


def synthetic_loss_curve(n_params: int, tokens_total: float, total_steps: int, eval_interval: int, noise: float = 0.02) -> dict[str, float]:
    """Generate a synthetic but plausible loss curve."""
    E = 1.69
    A = 406.4
    B = 410.7
    alpha = 0.34
    beta = 0.28

    losses = {}
    for step in range(eval_interval, total_steps + 1, eval_interval):
        frac = step / total_steps
        tokens = max(frac * tokens_total, 1.0)
        base_loss = E + A / (n_params ** alpha) + B / (tokens ** beta)
        # Add some realistic noise
        noise_val = random.gauss(0, noise * base_loss)
        losses[str(step)] = round(base_loss + noise_val, 4)
    return losses


def compute_n_params(model_cfg: dict) -> int:
    d = model_cfg["d_model"]
    n = model_cfg["n_layers"]
    d_ff = model_cfg.get("d_ff") or 4 * d
    act = model_cfg.get("activation", "gelu")
    attn = 4 * d * d
    ffn_mult = 3 if act == "swiglu" else 2
    ffn = ffn_mult * d * d_ff
    return n * (attn + ffn)


def main():
    out_dir = Path(__file__).parent.parent / "data" / "val"
    out_dir.mkdir(parents=True, exist_ok=True)

    for run in SAMPLE_RUNS:
        run_id = run["run_id"]
        run_dir = out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write config.json
        run_cfg = dict(run)
        run_cfg["eval_interval"] = EVAL_INTERVAL
        with open(run_dir / "config.json", "w") as f:
            json.dump(run_cfg, f, indent=2)

        # Write losses.json
        n_params = compute_n_params(run["model"])
        losses = synthetic_loss_curve(
            n_params=n_params,
            tokens_total=run["data"]["tokens_total"],
            total_steps=run["schedule"]["total_steps"],
            eval_interval=EVAL_INTERVAL,
        )
        with open(run_dir / "losses.json", "w") as f:
            json.dump(losses, f, indent=2)

        print(f"Wrote {run_id}/ (config.json + losses.json with {len(losses)} steps)")

    print(f"\nGenerated {len(SAMPLE_RUNS)} sample runs in {out_dir}/")


if __name__ == "__main__":
    main()
