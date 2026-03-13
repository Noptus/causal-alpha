"""Direct sequential policy learner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from causal_alpha_rl.envs.allocation import smooth_weights_from_scores_torch, weights_from_scores


@dataclass(slots=True)
class DirectPolicyNetwork:
    top_k: int
    max_weight: float
    transaction_cost: float
    random_state: int
    hidden_dim: int = 24
    epochs: int = 80
    learning_rate: float = 3e-3
    risk_lambda: float = 0.08
    persistence_scale: float = 0.15
    scaler: StandardScaler = field(init=False)
    model: torch.nn.Module = field(init=False)

    name: str = "standard_rl"

    def fit(
        self,
        *,
        train_features: np.ndarray,
        train_returns: np.ndarray,
        train_available: np.ndarray,
        val_features: np.ndarray,
        val_returns: np.ndarray,
        val_available: np.ndarray,
    ) -> None:
        self.scaler = StandardScaler()
        row_mask = train_available.reshape(-1)
        flat_features = train_features.reshape(-1, train_features.shape[-1])[row_mask]
        self.scaler.fit(flat_features)
        train_x = self._transform(train_features)
        val_x = self._transform(val_features)

        torch.manual_seed(self.random_state)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(train_x.shape[-1] + 1, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_state = deepcopy(self.model.state_dict())
        best_val = -np.inf
        patience = 0
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = -self._sequence_objective(train_x, train_returns, train_available)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                val_score = float(self._sequence_objective(val_x, val_returns, val_available).item())
            if val_score > best_val:
                best_val = val_score
                best_state = deepcopy(self.model.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= 12:
                break
        self.model.load_state_dict(best_state)

    def _transform(self, features: np.ndarray) -> torch.Tensor:
        original_shape = features.shape
        flat = features.reshape(-1, features.shape[-1])
        flat = self.scaler.transform(flat)
        return torch.tensor(flat.reshape(original_shape), dtype=torch.float32)

    def _sequence_objective(
        self,
        features: torch.Tensor,
        future_returns: np.ndarray,
        available: np.ndarray,
    ) -> torch.Tensor:
        prev_weights = torch.zeros(features.shape[1], dtype=torch.float32)
        rewards: list[torch.Tensor] = []
        future_tensor = torch.tensor(future_returns, dtype=torch.float32)
        available_tensor = torch.tensor(available.astype(float), dtype=torch.float32)
        for idx in range(features.shape[0]):
            prev_column = prev_weights.unsqueeze(-1)
            inputs = torch.cat([features[idx], prev_column], dim=1)
            scores = self.model(inputs).squeeze(-1) + self.persistence_scale * prev_weights
            weights = smooth_weights_from_scores_torch(
                scores,
                available_tensor[idx],
                max_weight=self.max_weight,
            )
            gross = torch.dot(weights, future_tensor[idx])
            turnover = torch.abs(weights - prev_weights).sum()
            reward = gross - self.transaction_cost * turnover - self.risk_lambda * gross.pow(2)
            rewards.append(reward)
            prev_weights = weights
        return torch.stack(rewards).mean()

    def predict(
        self,
        features: np.ndarray,
        available: np.ndarray,
        prev_weights: np.ndarray,
        **_: object,
    ) -> np.ndarray:
        transformed = self.scaler.transform(features)
        prev_column = prev_weights[:, None]
        inputs = torch.tensor(np.concatenate([transformed, prev_column], axis=1), dtype=torch.float32)
        with torch.no_grad():
            scores = self.model(inputs).squeeze(-1).numpy() + self.persistence_scale * prev_weights
        return weights_from_scores(scores, available, top_k=self.top_k, max_weight=self.max_weight)
