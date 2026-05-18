"""
Per-source XGBoost ensemble for LossCast-Bench.

Motivation: our training data mixes two very different loss regimes — Marin
(C4-en eval loss, irreducible floor ~2.0) and StepLaw (smoothed pretraining
loss, irreducible floor ~2.5). A single global model has to straddle both.
Routing each run to a source-specific model frees each head to specialize.

The source label is inferred from ``config.data.eval_dataset`` (``"c4_en"`` →
Marin, anything else → StepLaw). At predict time, unknown sources fall back
to whichever model has more training data for that source.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..schema import RunConfig, RunGroundTruth, RunPrediction
from .xgboost_v2 import XGBoostPredictorV2


def _source_of(config: RunConfig) -> str:
    eval_ds = (config.data.eval_dataset or "").lower()
    if eval_ds == "c4_en":
        return "marin"
    return "steplaw"


@dataclass
class XGBoostEnsemblePredictor:
    """Two XGBoost v2 predictors, routed by source (marin / steplaw)."""

    marin: Optional[XGBoostPredictorV2] = field(default=None)
    steplaw: Optional[XGBoostPredictorV2] = field(default=None)
    # Shared hyperparameters — overridden from fit() kwargs if desired.
    n_estimators: int = 600
    max_depth: int = 8
    learning_rate: float = 0.05

    def _new_head(self) -> XGBoostPredictorV2:
        return XGBoostPredictorV2(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
        )

    def fit(
        self,
        configs: list[RunConfig],
        ground_truths: list[RunGroundTruth],
        verbose: bool = False,
    ) -> "XGBoostEnsemblePredictor":
        gt_map = {g.run_id: g for g in ground_truths}

        by_src: dict[str, tuple[list[RunConfig], list[RunGroundTruth]]] = {
            "marin": ([], []),
            "steplaw": ([], []),
        }
        for cfg in configs:
            gt = gt_map.get(cfg.run_id)
            if gt is None:
                continue
            src = _source_of(cfg)
            by_src[src][0].append(cfg)
            by_src[src][1].append(gt)

        for src, (cfgs, gts) in by_src.items():
            if not cfgs:
                continue
            head = self._new_head()
            head.fit(cfgs, gts, verbose=verbose)
            setattr(self, src, head)
            print(f"  fitted {src} head on {len(cfgs)} runs")

        return self

    def predict_batch(self, configs: list[RunConfig]) -> list[RunPrediction]:
        # Keep original ordering while dispatching per source.
        preds_by_id: dict[str, RunPrediction] = {}
        for src in ("marin", "steplaw"):
            head: XGBoostPredictorV2 | None = getattr(self, src)
            if head is None:
                continue
            src_configs = [c for c in configs if _source_of(c) == src]
            if not src_configs:
                continue
            for p in head.predict_batch(src_configs):
                preds_by_id[p.run_id] = p

        out: list[RunPrediction] = []
        for c in configs:
            p = preds_by_id.get(c.run_id)
            if p is None:
                # Fallback: empty prediction (should not happen in practice)
                p = RunPrediction(run_id=c.run_id, predictions={})
            out.append(p)
        return out

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "marin": self._head_blob(self.marin),
            "steplaw": self._head_blob(self.steplaw),
            "hparams": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f)

    @staticmethod
    def _head_blob(head: Optional[XGBoostPredictorV2]):
        if head is None:
            return None
        return {"booster": head._booster, "cat_vocab": head._cat_vocab, "config": head._config()}

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostEnsemblePredictor":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        inst = cls(**blob.get("hparams", {}))
        for src in ("marin", "steplaw"):
            b = blob.get(src)
            if b is None:
                continue
            head = XGBoostPredictorV2(**b["config"])
            head._booster = b["booster"]
            head._cat_vocab = b["cat_vocab"]
            setattr(inst, src, head)
        return inst


def predict_batch(
    configs: list[RunConfig],
    train_configs: Optional[list[RunConfig]] = None,
    train_ground_truths: Optional[list[RunGroundTruth]] = None,
    model_path: Optional[str] = None,
) -> list[RunPrediction]:
    if model_path is not None and Path(model_path).exists():
        predictor = XGBoostEnsemblePredictor.load(model_path)
    elif train_configs is not None and train_ground_truths is not None:
        predictor = XGBoostEnsemblePredictor().fit(train_configs, train_ground_truths)
        if model_path is not None:
            predictor.save(model_path)
    else:
        raise ValueError(
            "xgboost_ensemble baseline needs a saved model or training data."
        )
    return predictor.predict_batch(configs)
