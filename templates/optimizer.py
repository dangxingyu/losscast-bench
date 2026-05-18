"""
LossCast-Bench — Standard Optimizer Template

This file is ONLY needed if you use a non-standard optimizer (i.e., not one of:
adamw, adam, sgd, lion). If you use a standard optimizer, skip this file —
the JSON config's optimizer fields are sufficient.

Instructions:
  1. Implement your optimizer as a subclass of torch.optim.Optimizer
  2. All per-step logic (including periodic operations) goes in step()
  3. step() receives step_idx so you can implement periodic behaviors
     (e.g., SOAP's preconditioner update every N steps)
  4. Update OPTIMIZER_DEFAULTS at the bottom
  5. Run this file standalone to verify: python optimizer.py

Requirements:
  - Pure PyTorch (no external dependencies)
  - Must be runnable standalone
  - step(step_idx=...) interface — all periodic logic goes here

Constraints:
  - Do NOT include model definition, data loading, or LR scheduling here
  - LR scheduling is handled externally (specified in JSON config)
  - This file only defines the optimizer's update rule
"""

import math
import torch
from torch.optim import Optimizer


# =============================================================================
# OPTIMIZER DEFINITION — Fill in your custom optimizer below
# =============================================================================

class CustomOptimizer(Optimizer):
    """
    Your custom optimizer.

    Replace this with your actual optimizer. The example below shows the
    structure for a SOAP-like optimizer with periodic preconditioner updates.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        # Add your custom hyperparameters here:
        # precondition_frequency: int = 100,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            # precondition_frequency=precondition_frequency,
        )
        super().__init__(params, defaults)

        raise NotImplementedError("Fill in your optimizer")

    @torch.no_grad()
    def step(self, closure=None, step_idx: int = 0):
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure for re-evaluating the loss.
            step_idx: Current training step index (0-based). Use this for
                      any periodic operations (e.g., preconditioner updates).

        Returns:
            loss: Loss value if closure is provided, else None.

        Example for periodic operations:
            if step_idx % self.defaults["precondition_frequency"] == 0:
                self._update_preconditioner(group, state)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state["step"] = 0
                    # Add your state initialization here:
                    # state["exp_avg"] = torch.zeros_like(p)
                    # state["exp_avg_sq"] = torch.zeros_like(p)
                    pass

                state["step"] += 1

                # === YOUR UPDATE RULE HERE ===
                #
                # Example (AdamW-like structure):
                # exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                # beta1, beta2 = group["betas"]
                #
                # # Decay
                # if group["weight_decay"] != 0:
                #     p.mul_(1 - group["lr"] * group["weight_decay"])
                #
                # # Moment updates
                # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                #
                # # Periodic operation example:
                # if step_idx % group["precondition_frequency"] == 0:
                #     self._update_preconditioner(state)
                #
                # # Parameter update
                # bias_correction1 = 1 - beta1 ** state["step"]
                # bias_correction2 = 1 - beta2 ** state["step"]
                # step_size = group["lr"] / bias_correction1
                # denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                # p.addcdiv_(exp_avg, denom, value=-step_size)

                raise NotImplementedError("Fill in your update rule")

        return loss


# =============================================================================
# OPTIMIZER DEFAULTS — Update to match your optimizer's hyperparameters
# =============================================================================

OPTIMIZER_DEFAULTS = {
    "name": "custom_optimizer",
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    # Add your custom hyperparameters here:
    # "precondition_frequency": 100,
}


# =============================================================================
# VERIFICATION — Do not modify below this line
# =============================================================================

def verify():
    """Verify the optimizer works correctly."""
    print("=" * 60)
    print("  LossCast-Bench Optimizer Verification")
    print("=" * 60)

    # Create a simple model to test with
    model = torch.nn.Linear(64, 64)
    optimizer = CustomOptimizer(model.parameters(), **OPTIMIZER_DEFAULTS)
    print(f"  Optimizer built:       {OPTIMIZER_DEFAULTS['name']}")

    # Run a few steps
    n_steps = 200
    losses = []
    for i in range(n_steps):
        x = torch.randn(4, 64)
        y = torch.randn(4, 64)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step(step_idx=i)
        optimizer.zero_grad()
        losses.append(loss.item())

    print(f"  Ran {n_steps} steps:       OK")
    print(f"  Initial loss:          {losses[0]:.4f}")
    print(f"  Final loss:            {losses[-1]:.4f}")

    # Check loss decreased
    assert losses[-1] < losses[0], "Loss did not decrease — optimizer may not be working"
    print(f"  Loss decreased:        OK")

    # Check no NaN in parameters
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), f"NaN/Inf in {name}"
    print(f"  Numerical stability:   OK")

    print("=" * 60)
    print("  ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    verify()
