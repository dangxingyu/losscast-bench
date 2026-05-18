"""XGBoost feature/blend baseline v1.

This is a small calibration layer on top of ``xgboost_ultimate``.  It keeps the
same trained heads, but blends the CatBoost and PCA heads with a step-dependent
weight:

    w_cat(step) = w_final + (w_start - w_final) * (1 - step_frac) ** power

The early/mid curve gets more of the curve-accurate CatBoost head, while the
final point keeps more of the final-loss-accurate PCA head.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..schema import RunConfig, RunPrediction
from .chinchilla import chinchilla_loss
from .pca_xgboost import PCAXGBoostPredictor
from .xgboost_baseline import XGBoostPredictor as _XGBoostPredictor
from .xgboost_ultimate import XGBoostUltimatePredictor, _CatBoostHead


@dataclass
class XGBoostFeatV1Predictor(XGBoostUltimatePredictor):
    """Dynamic blend of the ultimate predictor heads."""

    # Force the parent fit() to train/load all heads used by the dynamic blend.
    marin_blend: tuple[float, float] = (1.0, 1.0)
    steplaw_blend: tuple[float, float] = (1.0, 1.0)

    marin_start_weight: float = 1.0
    marin_final_weight: float = 0.56
    marin_power: float = 1.5

    steplaw_start_weight: float = 0.20
    steplaw_final_weight: float = 0.02
    steplaw_power: float = 3.0

    nanochat_alpha: float = 0.495
    nanochat_constant: float = -0.0825
    nanochat_power: float = 1.25

    @staticmethod
    def _scheduled_weight(step_frac: float, start: float, final: float, power: float) -> float:
        frac = min(max(float(step_frac), 0.0), 1.0)
        return final + (start - final) * ((1.0 - frac) ** power)

    def _nanochat_correction(self, config: RunConfig, step: int) -> float:
        if self._nanochat_max_n <= 0 or self._nanochat_min_d <= 0 or step == 0:
            return 0.0
        n = config.model.n_params_approx
        total = max(int(config.schedule.total_steps or 1), 1)
        step_frac = step / total
        d_actual = max(step_frac * config.data.tokens_total, 1.0)
        d = max(d_actual, self._nanochat_min_d / 10.0)
        chin_actual = chinchilla_loss(n, d)
        chin_clipped = chinchilla_loss(
            min(n, self._nanochat_max_n),
            max(d, self._nanochat_min_d),
        )
        raw_delta = chin_actual - chin_clipped
        return self.nanochat_alpha * (
            self.nanochat_constant + raw_delta
        ) * (step_frac ** self.nanochat_power)

    def _blend(self, src: str, configs: list[RunConfig]) -> list[RunPrediction]:
        if src == "nanochat":
            if self._nanochat_xgb is None:
                return [RunPrediction(run_id=c.run_id, predictions={}) for c in configs]
            raw = self._nanochat_xgb.predict_batch(configs)
            out: list[RunPrediction] = []
            for c, p in zip(configs, raw):
                preds = {
                    int(s): float(v + self._nanochat_correction(c, int(s)))
                    for s, v in p.predictions.items()
                }
                out.append(RunPrediction(run_id=c.run_id, predictions=preds))
            return out

        if src == "marin":
            cat_head, pca_head = self._marin_cat, self._marin_pca
            start = self.marin_start_weight
            final = self.marin_final_weight
            power = self.marin_power
        else:
            cat_head, pca_head = self._steplaw_cat, self._steplaw_pca
            start = self.steplaw_start_weight
            final = self.steplaw_final_weight
            power = self.steplaw_power

        if cat_head is None or pca_head is None:
            return super()._blend(src, configs)

        cat_preds = {p.run_id: p for p in cat_head.predict_batch(configs)}
        pca_preds = {p.run_id: p for p in pca_head.predict_batch(configs)}

        out: list[RunPrediction] = []
        for c in configs:
            cp = cat_preds[c.run_id]
            pp = pca_preds[c.run_id]
            total_steps = max(int(c.schedule.total_steps or 1), 1)
            keys = sorted(set(cp.predictions) | set(pp.predictions))
            preds: dict[int, float] = {}
            for step in keys:
                step_frac = int(step) / total_steps
                w = self._scheduled_weight(step_frac, start, final, power)
                if step in cp.predictions and step in pp.predictions:
                    preds[int(step)] = float(
                        w * cp.predictions[step] + (1.0 - w) * pp.predictions[step]
                    )
                elif step in cp.predictions:
                    preds[int(step)] = float(cp.predictions[step])
                else:
                    preds[int(step)] = float(pp.predictions[step])
            out.append(RunPrediction(run_id=c.run_id, predictions=preds))
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        super().save(path)
        with open(path / "feat_v1_meta.json", "w") as f:
            json.dump(self._feat_v1_meta(), f, indent=2)

    def _feat_v1_meta(self) -> dict[str, float]:
        return {
            "marin_start_weight": self.marin_start_weight,
            "marin_final_weight": self.marin_final_weight,
            "marin_power": self.marin_power,
            "steplaw_start_weight": self.steplaw_start_weight,
            "steplaw_final_weight": self.steplaw_final_weight,
            "steplaw_power": self.steplaw_power,
            "nanochat_alpha": self.nanochat_alpha,
            "nanochat_constant": self.nanochat_constant,
            "nanochat_power": self.nanochat_power,
        }

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostFeatV1Predictor":
        path = Path(path)
        meta = {}
        if (path / "feat_v1_meta.json").exists():
            with open(path / "feat_v1_meta.json") as f:
                meta = json.load(f)

        inst = cls(**meta)
        with open(path / "meta.json") as f:
            base_meta = json.load(f)
        inst._nanochat_max_n = float(base_meta.get("nanochat_max_n", 0.0))
        inst._nanochat_min_d = float(base_meta.get("nanochat_min_d", 0.0))

        for fname in ("ncpl_cat.pkl", "shared_cat.pkl", "marin_cat.pkl"):
            if (path / fname).exists():
                ncpl_cat = _CatBoostHead.load(path / fname)
                inst._marin_cat = ncpl_cat
                inst._steplaw_cat = ncpl_cat
                break
        if (path / "nanochat_xgb.pkl").exists():
            inst._nanochat_xgb = _XGBoostPredictor.load(path / "nanochat_xgb.pkl")
        if (path / "marin_pca.pkl").exists():
            inst._marin_pca = PCAXGBoostPredictor.load(path / "marin_pca.pkl")
        if (path / "steplaw_pca.pkl").exists():
            inst._steplaw_pca = PCAXGBoostPredictor.load(path / "steplaw_pca.pkl")
        return inst


def predict_batch(
    configs: list[RunConfig],
    train_configs: list[RunConfig] | None = None,
    train_ground_truths: list | None = None,
    model_path: str | None = None,
) -> list[RunPrediction]:
    if model_path is not None and Path(model_path).exists():
        predictor = XGBoostFeatV1Predictor.load(model_path)
    elif train_configs is not None and train_ground_truths is not None:
        predictor = XGBoostFeatV1Predictor().fit(train_configs, train_ground_truths)
        if model_path is not None:
            predictor.save(model_path)
    else:
        raise ValueError(
            "xgboost_feat_v1 needs a saved model or training data."
        )
    return predictor.predict_batch(configs)
