"""
LossCast-Bench — Example: 125M Transformer (GPT-2 style with SwiGLU + RMSNorm)

A standard decoder-only transformer with:
  - RoPE positional encoding
  - SwiGLU FFN
  - RMSNorm (pre-norm)
  - No bias in linear layers
  - GPT-2 style initialization with residual scaling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# COMPONENTS
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.w_down._is_residual_proj = True

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.wo._is_residual_proj = True

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# =============================================================================
# MODEL DEFINITION
# =============================================================================

class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        d_model = config["d_model"]
        n_layers = config["n_layers"]
        n_heads = config["n_heads"]
        d_ff = config.get("d_ff") or 4 * d_model
        vocab_size = config["vocab_size"]

        self.embed = nn.Embedding(vocab_size, d_model)
        self.rotary = RotaryEmbedding(d_model // n_heads)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if config.get("tied_embeddings", False):
            self.head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """GPT-2 style init with residual projection scaling."""
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "_is_residual_proj"):
                std *= (2 * self.config["n_layers"]) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.embed(input_ids)
        cos, sin = self.rotary(T, x.device)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.head(x)


# =============================================================================
# MODEL CONFIG
# =============================================================================

MODEL_CONFIG = {
    "vocab_size": 32000,
    "d_model": 768,
    "n_layers": 12,
    "n_heads": 12,
    "d_ff": None,
    "activation": "swiglu",
    "norm_type": "rmsnorm",
    "rope": True,
    "n_kv_heads": None,
    "tied_embeddings": False,
}


# =============================================================================
# VERIFICATION — Do not modify below this line
# =============================================================================

def count_parameters(model: nn.Module, exclude_embeddings: bool = True) -> int:
    total = 0
    for name, param in model.named_parameters():
        if exclude_embeddings and ("embed" in name or "head" in name):
            continue
        total += param.numel()
    return total


def verify():
    print("=" * 60)
    print("  LossCast-Bench Model Verification")
    print("=" * 60)

    model = Model(MODEL_CONFIG)
    print(f"  Model built successfully")

    n_params = count_parameters(model, exclude_embeddings=True)
    n_params_total = sum(p.numel() for p in model.parameters())
    print(f"  Non-embedding params:  {n_params:,}")
    print(f"  Total params:          {n_params_total:,}")

    batch_size = 2
    seq_len = 128
    dummy_input = torch.randint(0, MODEL_CONFIG["vocab_size"], (batch_size, seq_len))

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (batch_size, seq_len, MODEL_CONFIG["vocab_size"]), \
        f"Expected shape {(batch_size, seq_len, MODEL_CONFIG['vocab_size'])}, got {logits.shape}"
    print(f"  Forward pass:          OK  (output shape: {tuple(logits.shape)})")

    assert torch.isfinite(logits).all(), "Output contains NaN or Inf"
    print(f"  Numerical stability:   OK")

    print("=" * 60)
    print("  ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    verify()
