"""Rolling correlation-style factor ranking baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_alpha_rl.envs.allocation import weights_from_scores


@dataclass(slots=True)
class RollingRankBaseline:
    top_k: int
    max_weight: float
    transaction_cost: float = 0.0
    scores: np.ndarray | None = None

    name: str = "correlation_rank"

    def fit(self, future_returns: np.ndarray, available: np.ndarray, **_: object) -> None:
        masked = np.where(available, future_returns, np.nan)
        with np.errstate(invalid="ignore"):
            mean_return = np.nanmean(masked, axis=0)
            vol = np.nanstd(masked, axis=0)
        self.scores = np.nan_to_num(mean_return / (vol + 1.0), nan=-1e6, neginf=-1e6, posinf=1e6)

    def predict(
        self,
        features: np.ndarray,
        available: np.ndarray,
        prev_weights: np.ndarray,
        **_: object,
    ) -> np.ndarray:
        del features, prev_weights
        assert self.scores is not None
        return weights_from_scores(self.scores, available, top_k=self.top_k, max_weight=self.max_weight)
