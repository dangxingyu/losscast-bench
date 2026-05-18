"""
LossCast-Bench — Standard Model Template

Instructions:
  1. Fill in __init__() with your architecture definition
  2. Fill in _init_weights() with your exact initialization logic
  3. Fill in forward() with your forward pass
  4. Update MODEL_CONFIG at the bottom with your model's metadata
  5. Run this file standalone to verify: python model.py

Requirements:
  - Pure PyTorch (no external framework dependencies like HuggingFace, etc.)
  - Must be runnable standalone (see __main__ block below)
  - Parameter count from this file should match your JSON config's n_params_approx

Constraints:
  - Do NOT include optimizer, data loading, or training loop logic here
  - Do NOT include dropout (we care about the deterministic architecture)
  - Keep it self-contained: all layers defined in this single file
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MODEL DEFINITION — Fill in your architecture below
# =============================================================================

class Model(nn.Module):
    """
    Your model architecture.

    Replace this with your actual model. The example below shows a minimal
    transformer block structure for reference — delete it and write yours.
    """

    def __init__(self, config: dict):
        super().__init__()
        # Example (delete and replace with your architecture):
        #
        # self.embed = nn.Embedding(config["vocab_size"], config["d_model"])
        # self.layers = nn.ModuleList([
        #     TransformerBlock(config) for _ in range(config["n_layers"])
        # ])
        # self.norm = nn.RMSNorm(config["d_model"])
        # self.head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)

        raise NotImplementedError("Fill in your architecture")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Weight initialization.

        This is where per-layer init logic goes. Be precise — this should
        exactly reproduce what your training code does.

        Example (delete and replace with your init):
        #
        # if isinstance(module, nn.Linear):
        #     std = 0.02
        #     # Scale output projections by 1/sqrt(2*n_layers) (GPT-2 style)
        #     if hasattr(module, "_is_residual_proj"):
        #         std *= (2 * self.config["n_layers"]) ** -0.5
        #     nn.init.normal_(module.weight, mean=0.0, std=std)
        #     if module.bias is not None:
        #         nn.init.zeros_(module.bias)
        # elif isinstance(module, nn.Embedding):
        #     nn.init.normal_(module.weight, mean=0.0, std=0.02)
        """
        raise NotImplementedError("Fill in your initialization")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) — integer token IDs

        Returns:
            logits: (batch_size, seq_len, vocab_size) — raw logits (no softmax)
        """
        # Example (delete and replace with your forward pass):
        #
        # x = self.embed(input_ids)
        # for layer in self.layers:
        #     x = layer(x)
        # x = self.norm(x)
        # logits = self.head(x)
        # return logits

        raise NotImplementedError("Fill in your forward pass")


# =============================================================================
# MODEL CONFIG — Update this to match your model
# =============================================================================

MODEL_CONFIG = {
    "vocab_size": 32000,
    "d_model": 768,
    "n_layers": 12,
    "n_heads": 12,
    "d_ff": None,           # None = 4 * d_model
    "activation": "swiglu",
    "norm_type": "rmsnorm",
    "rope": True,
    "n_kv_heads": None,     # None = MHA
    "tied_embeddings": False,
}


# =============================================================================
# VERIFICATION — Do not modify below this line
# =============================================================================

def count_parameters(model: nn.Module, exclude_embeddings: bool = True) -> int:
    """Count non-embedding parameters."""
    total = 0
    for name, param in model.named_parameters():
        if exclude_embeddings and ("embed" in name or "head" in name):
            continue
        total += param.numel()
    return total


def verify():
    """Verify the model is well-formed."""
    print("=" * 60)
    print("  LossCast-Bench Model Verification")
    print("=" * 60)

    # Build model
    model = Model(MODEL_CONFIG)
    print(f"  Model built successfully")

    # Parameter count
    n_params = count_parameters(model, exclude_embeddings=True)
    n_params_total = sum(p.numel() for p in model.parameters())
    print(f"  Non-embedding params:  {n_params:,}")
    print(f"  Total params:          {n_params_total:,}")

    # Test forward pass
    batch_size = 2
    seq_len = MODEL_CONFIG.get("seq_len", 128)  # short seq for verification
    dummy_input = torch.randint(0, MODEL_CONFIG["vocab_size"], (batch_size, seq_len))

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (batch_size, seq_len, MODEL_CONFIG["vocab_size"]), \
        f"Expected shape {(batch_size, seq_len, MODEL_CONFIG['vocab_size'])}, got {logits.shape}"
    print(f"  Forward pass:          OK  (output shape: {tuple(logits.shape)})")

    # Check for NaN/Inf
    assert torch.isfinite(logits).all(), "Output contains NaN or Inf"
    print(f"  Numerical stability:   OK")

    print("=" * 60)
    print("  ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    verify()
