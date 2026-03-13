"""Elastic net factor selector baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from causal_alpha_rl.envs.allocation import weights_from_scores


@dataclass(slots=True)
class ElasticNetFactorSelector:
    top_k: int
    max_weight: float
    random_state: int
    factor_eye: np.ndarray
    transaction_cost: float = 0.0
    scaler: StandardScaler = field(init=False)
    model: ElasticNetCV = field(init=False)

    name: str = "elastic_net"

    def fit(self, features: np.ndarray, future_returns: np.ndarray, available: np.ndarray) -> None:
        row_mask = available.reshape(-1)
        x_numeric = features.reshape(-1, features.shape[-1])[row_mask]
        factor_idx = np.tile(np.arange(features.shape[1]), features.shape[0])[row_mask]
        x = np.concatenate([x_numeric, self.factor_eye[factor_idx]], axis=1)
        y = future_returns.reshape(-1)[row_mask]
        self.scaler = StandardScaler()
        x[:, : x_numeric.shape[1]] = self.scaler.fit_transform(x[:, : x_numeric.shape[1]])
        self.model = ElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8],
            alphas=np.logspace(-3, -0.3, 10),
            max_iter=5000,
            random_state=self.random_state,
        )
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
