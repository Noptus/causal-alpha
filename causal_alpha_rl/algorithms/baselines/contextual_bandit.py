"""One-step contextual bandit baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from causal_alpha_rl.envs.allocation import weights_from_scores


@dataclass(slots=True)
class ContextualBanditBaseline:
    top_k: int
    max_weight: float
    factor_eye: np.ndarray
    transaction_cost: float = 0.0
    scaler: StandardScaler = field(init=False)
    model: RidgeCV = field(init=False)

    name: str = "contextual_bandit"

    def fit(self, features: np.ndarray, future_returns: np.ndarray, available: np.ndarray) -> None:
        row_mask = available.reshape(-1)
        x_numeric = features.reshape(-1, features.shape[-1])[row_mask]
        factor_idx = np.tile(np.arange(features.shape[1]), features.shape[0])[row_mask]
        x = np.concatenate([x_numeric, self.factor_eye[factor_idx]], axis=1)
        y = future_returns.reshape(-1)[row_mask]
        self.scaler = StandardScaler()
        x[:, : x_numeric.shape[1]] = self.scaler.fit_transform(x[:, : x_numeric.shape[1]])
        self.model = RidgeCV(alphas=np.logspace(-3, 2, 12))
        self.model.fit(x, y)

    def predict(
        self,
        features: np.ndarray,
        available: np.ndarray,
        prev_weights: np.ndarray,
        **_: object,
    ) -> np.ndarray:
        del prev_weights
        x = np.concatenate([features, self.factor_eye], axis=1)
        x[:, : features.shape[1]] = self.scaler.transform(x[:, : features.shape[1]])
        scores = self.model.predict(x)
        return weights_from_scores(scores, available, top_k=self.top_k, max_weight=self.max_weight)
