"""
Input/output schema definitions for LossCast-Bench.

The schema defines a strict contract between the benchmark and participants:
- RunConfig: what participants receive (the training configuration)
- RunGroundTruth: what we hold (actual loss values at each step)
- RunPrediction: what participants submit (predicted loss values)
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import Optional
import json


# ── Model Architecture ──────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Architecture specification."""

    arch: str  # e.g. "transformer", "mamba", "rwkv", "retnet"
    n_layers: int
    d_model: int
    n_heads: Optional[int] = None  # None for non-attention archs
    head_dim: Optional[int] = None  # per-head dimension; None = d_model // n_heads
    d_ff: Optional[int] = None  # feedforward dim; defaults to 4*d_model
    vocab_size: int = 32000
    tied_embeddings: bool = False
    activation: str = "swiglu"  # "gelu", "relu", "swiglu"
    norm_type: str = "rmsnorm"  # "layernorm", "rmsnorm"
    rope: bool = True
    n_kv_heads: Optional[int] = None  # for GQA; None = MHA

    @property
    def n_params_approx(self) -> int:
        """Rough parameter count (non-embedding)."""
        d_ff = self.d_ff or 4 * self.d_model
        n_heads = self.n_heads or 1
        head_dim = self.head_dim or (self.d_model // n_heads)
        n_kv = self.n_kv_heads or n_heads
        # Q: d_model * n_heads * head_dim, K/V: d_model * n_kv * head_dim, O: n_heads * head_dim * d_model
        attn = self.d_model * (n_heads + n_kv * 2) * head_dim + n_heads * head_dim * self.d_model
        ffn_mult = 3 if self.activation == "swiglu" else 2
        ffn = ffn_mult * self.d_model * d_ff
        return self.n_layers * (attn + ffn)


# ── Optimizer ────────────────────────────────────────────────────────────────

@dataclass
class OptimizerConfig:
    """Optimizer specification."""

    name: str = "adamw"  # "adamw", "adam", "sgd", "lion", "muon", "soap"
    lr: float = 3e-4  # peak learning rate
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: Optional[float] = 1.0
    mup: bool = False  # muP / spectral scaling
    # Extra optimizer-specific params (e.g., adam_lr, preconditioner_lr for muon/mudam)
    extra: Optional[dict] = None


# ── Data Recipe ──────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Training data specification."""

    dataset: str  # e.g. "c4", "pile", "refinedweb", "fineweb", "dolma", "custom_mix"
    tokenizer: str = "gpt2"  # HuggingFace tokenizer name, e.g. "gpt2", "meta-llama/Llama-2-7b-hf"
    seq_len: int = 2048
    batch_tokens: int = 524288  # batch size in tokens (batch_size * seq_len)
    tokens_total: float = 10e9  # total training tokens
    # data mix (optional, for multi-source)
    mix: Optional[dict[str, float]] = None  # e.g. {"web": 0.7, "code": 0.2, "books": 0.1}
    # eval dataset: which dataset's validation split is used to measure loss (required)
    eval_dataset: str = "fineweb"  # e.g. "c4", "fineweb", "hellaswag"


# ── LR Schedule ──────────────────────────────────────────────────────────────

@dataclass
class ScheduleConfig:
    """Learning rate schedule and training duration."""

    lr_schedule: str = "cosine"  # "cosine", "linear", "constant", "wsd", "inverse_sqrt"
    warmup_steps: int = 2000
    total_steps: int = 50000
    cooldown_steps: int = 0  # for WSD (warmup-stable-decay)
    final_lr_ratio: float = 0.1  # final_lr / peak_lr


# ── Full Run Configuration (INPUT to participants) ──────────────────────────

@dataclass
class RunConfig:
    """
    Complete training run configuration.
    This is what participants receive as input.
    """

    run_id: str
    model: ModelConfig
    optimizer: OptimizerConfig
    data: DataConfig
    schedule: ScheduleConfig
    # Evaluation interval: loss is recorded every eval_interval steps
    eval_interval: int = 500
    # Optional metadata (not used for prediction, but useful context)
    hardware: Optional[str] = None  # e.g. "8xA100-80G"
    precision: str = "bf16"  # "fp32", "bf16", "fp16"
    # Parallelism (may affect effective batch dynamics)
    dp_size: int = 1
    tp_size: int = 1
    # Source code provenance
    code_repo: Optional[str] = None   # e.g. "karpathy/autoresearch"
    code_commit: Optional[str] = None  # git commit hash, e.g. "a3f7b2c"

    @property
    def eval_steps(self) -> list[int]:
        """Compute the list of eval steps from eval_interval and total_steps.

        Includes step 0 (initial eval) and the final step even if it doesn't
        align with eval_interval — real training runs typically log both.
        """
        total = self.schedule.total_steps
        steps = [0] + list(range(self.eval_interval, total + 1, self.eval_interval))
        # Add final step if not already included
        if total not in steps:
            steps.append(total)
        return steps

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        # Parse optimizer, collecting unknown fields into extra
        opt_d = dict(d["optimizer"])
        opt_known = {f.name for f in fields(OptimizerConfig) if f.name != "extra"}
        opt_extra = {k: opt_d.pop(k) for k in list(opt_d) if k not in opt_known}
        if opt_extra:
            opt_d["extra"] = opt_extra

        # Parse model, ignoring unknown fields
        model_d = {k: v for k, v in d["model"].items() if k in {f.name for f in fields(ModelConfig)}}

        # Parse data, ignoring unknown fields
        data_d = {k: v for k, v in d["data"].items() if k in {f.name for f in fields(DataConfig)}}

        # Parse schedule, ignoring unknown fields
        sched_d = {k: v for k, v in d["schedule"].items() if k in {f.name for f in fields(ScheduleConfig)}}

        return cls(
            run_id=d["run_id"],
            model=ModelConfig(**model_d),
            optimizer=OptimizerConfig(**opt_d),
            data=DataConfig(**data_d),
            schedule=ScheduleConfig(**sched_d),
            eval_interval=d.get("eval_interval", 500),
            hardware=d.get("hardware"),
            precision=d.get("precision", "bf16"),
            dp_size=d.get("dp_size", 1),
            tp_size=d.get("tp_size", 1),
            code_repo=d.get("code_repo"),
            code_commit=d.get("code_commit"),
        )

    @classmethod
    def from_json(cls, s: str) -> "RunConfig":
        return cls.from_dict(json.loads(s))


# ── Ground Truth (held by benchmark organizers) ─────────────────────────────

@dataclass
class RunGroundTruth:
    """
    Actual loss values at each eval step.
    This is what the benchmark holds internally.
    """

    run_id: str
    losses: dict[int, float]  # step -> loss value

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "losses": {str(k): v for k, v in self.losses.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "RunGroundTruth":
        return cls(
            run_id=d["run_id"],
            losses={int(k): v for k, v in d["losses"].items()},
        )

    @property
    def final_loss(self) -> float:
        return self.losses[max(self.losses.keys())]


# ── Prediction (submitted by participants) ──────────────────────────────────

@dataclass
class RunPrediction:
    """
    Predicted loss values at each eval step.
    This is what participants submit.
    """

    run_id: str
    predictions: dict[int, float]  # step -> predicted loss

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "predictions": {str(k): v for k, v in self.predictions.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "RunPrediction":
        raw = d.get("predictions") or d.get("losses", {})
        return cls(
            run_id=d["run_id"],
            predictions={int(k): v for k, v in raw.items()},
        )

    @property
    def final_loss(self) -> float:
        return self.predictions[max(self.predictions.keys())]


# ── Batch I/O ────────────────────────────────────────────────────────────────

def load_configs(path: str) -> list[RunConfig]:
    """Load a list of run configs from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get("runs", [])
    return [RunConfig.from_dict(d) for d in items]


def load_ground_truths(path: str) -> list[RunGroundTruth]:
    """Load ground truth from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get("runs", [])
    return [RunGroundTruth.from_dict(d) for d in items]


def load_predictions(path: str) -> list[RunPrediction]:
    """Load predictions from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get("runs", [])
    return [RunPrediction.from_dict(d) for d in items]


def save_predictions(preds: list[RunPrediction], path: str) -> None:
    """Save predictions to a JSON file."""
    with open(path, "w") as f:
        json.dump({"runs": [p.to_dict() for p in preds]}, f, indent=2)


# ── Validation ───────────────────────────────────────────────────────────────

def validate_submission(
    predictions: list[RunPrediction],
    configs: Optional[list[RunConfig]] = None,
    ground_truths: Optional[list[RunGroundTruth]] = None,
) -> list[str]:
    """
    Validate that predictions cover the expected scoring surface.

    If configs are available, their eval_steps define the submitted prediction
    grid. If ground truths are also available, steps absent from the ground
    truth are not required because they cannot be scored. If configs are not
    available, ground_truths define the required run/step coverage.

    Returns a list of error messages (empty = valid).
    """
    errors = []
    config_map = {c.run_id: c for c in configs or []}
    truth_map = {gt.run_id: gt for gt in ground_truths or []}
    expected_ids = set(config_map) or set(truth_map)

    pred_map = {}
    for pred in predictions:
        if pred.run_id in pred_map:
            errors.append(f"Duplicate predictions for run {pred.run_id}")
        pred_map[pred.run_id] = pred
    pred_ids = set(pred_map)

    # Check all runs are covered
    missing = expected_ids - pred_ids
    if missing:
        errors.append(f"Missing predictions for {len(missing)} runs: {sorted(missing)[:5]}...")

    extra = pred_ids - expected_ids
    if extra:
        errors.append(f"Extra predictions for unknown runs: {sorted(extra)[:5]}...")

    for run_id in sorted(expected_ids):
        pred = pred_map.get(run_id)
        if pred is None:
            continue

        cfg = config_map.get(run_id)
        gt = truth_map.get(run_id)
        pred_steps = set(pred.predictions)

        if cfg is not None:
            config_steps = set(cfg.eval_steps)
            required_steps = config_steps
            if gt is not None:
                required_steps = config_steps & set(gt.losses)
            unexpected_steps = pred_steps - config_steps
        else:
            required_steps = set(gt.losses) if gt is not None else set()
            unexpected_steps = pred_steps - required_steps

        if not pred_steps:
            errors.append(f"Run {run_id}: no predictions")
            continue

        missing_steps = required_steps - pred_steps
        if missing_steps:
            errors.append(
                f"Run {run_id}: missing predictions for {len(missing_steps)} "
                f"required steps: {sorted(missing_steps)[:5]}..."
            )

        if unexpected_steps:
            errors.append(
                f"Run {run_id}: predictions contain {len(unexpected_steps)} "
                f"unexpected steps outside the configured eval grid: "
                f"{sorted(unexpected_steps)[:5]}..."
            )

    # Check values are reasonable
    for pred in predictions:
        for step, val in pred.predictions.items():
            if not isinstance(val, (int, float)):
                errors.append(f"Run {pred.run_id} step {step}: non-numeric value {val}")
            elif val < 0:
                errors.append(f"Run {pred.run_id} step {step}: negative loss {val}")
            elif val > 20:
                errors.append(f"Run {pred.run_id} step {step}: suspiciously high loss {val}")

    return errors
