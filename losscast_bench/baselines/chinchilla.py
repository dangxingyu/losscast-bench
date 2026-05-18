"""
Chinchilla scaling law baseline.

Predicts loss based on the Chinchilla functional form:
    L(N, D) = E + A / N^alpha + B / D^beta

where N = non-embedding params, D = training tokens.

For the full curve, we interpolate by treating D as tokens seen at each step.

Default parameters from Hoffmann et al. (2022).
"""

from __future__ import annotations

import math
from ..schema import RunConfig, RunPrediction


# Chinchilla fit parameters (Table A9 from Hoffmann et al.)
E = 1.69  # irreducible loss
A = 406.4
B = 410.7
ALPHA = 0.34
BETA = 0.28


def chinchilla_loss(n_params: int, n_tokens: float) -> float:
    """Predict loss using Chinchilla scaling law."""
    return E + A / (n_params ** ALPHA) + B / (n_tokens ** BETA)


# Nanochat-specific Chinchilla parameters, fit on d8/d12 train data (590 runs)
# using scipy.optimize.curve_fit.  The Hoffmann (2022) parameters were fit on
# GPT-like models with C4 data; nanochat uses autoresearch_bpe_8k tokenizer,
# Muon optimizer, and a different architecture family, so needs its own fit.
_NC_E = 1.670352
_NC_A = 31.1596
_NC_B = 3223.1430
_NC_ALPHA = 0.239915
_NC_BETA = 0.423452


def nanochat_chinchilla_loss(n_params: int, n_tokens: float) -> float:
    """Predict loss using Chinchilla scaling law with nanochat-specific parameters."""
    return _NC_E + _NC_A / (n_params ** _NC_ALPHA) + _NC_B / (n_tokens ** _NC_BETA)


def predict_run(config: RunConfig) -> RunPrediction:
    """Generate Chinchilla baseline predictions for a single run."""
    n_params = config.model.n_params_approx
    total_tokens = config.data.tokens_total
    total_steps = config.schedule.total_steps

    predictions = {}
    for step in config.eval_steps:
        if step == 0:
            # Chinchilla can't predict init loss (0 tokens seen).
            # Use ln(vocab_size) as a reasonable init loss estimate.
            predictions[step] = math.log(config.model.vocab_size)
            continue
        # Tokens seen at this step (linear interpolation)
        frac = step / total_steps
        tokens_at_step = frac * total_tokens
        tokens_at_step = max(tokens_at_step, 1.0)  # avoid div by zero
        predictions[step] = chinchilla_loss(n_params, tokens_at_step)

    return RunPrediction(run_id=config.run_id, predictions=predictions)


def predict_batch(configs: list[RunConfig]) -> list[RunPrediction]:
    """Generate Chinchilla baseline predictions for multiple runs."""
    return [predict_run(c) for c in configs]
