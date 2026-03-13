"""Causal regime-aware sequential policy learner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from causal_alpha_rl.envs.allocation import smooth_weights_from_scores_torch, weights_from_scores


@dataclass(slots=True)
class CausalPolicyNetwork:
    top_k: int
    max_weight: float
    transaction_cost: float
    random_state: int
    hidden_dim: int = 32
    epochs: int = 90
    learning_rate: float = 3e-3
    risk_lambda: float = 0.08
    robustness_lambda: float = 0.80
    min_regime_bonus: float = 0.50
    counterfactual_weight: float = 0.50
    persistence_scale: float = 0.15
    scaler: StandardScaler = field(init=False)
    model: torch.nn.Module = field(init=False)

    name: str = "causal_rl"

    def fit(
        self,
        *,
        train_features: np.ndarray,
        train_returns: np.ndarray,
        train_available: np.ndarray,
        val_features: np.ndarray,
        val_returns: np.ndarray,
        val_available: np.ndarray,
        train_regime_posteriors: np.ndarray,
        val_regime_posteriors: np.ndarray,
        train_regime_labels: np.ndarray,
        val_regime_labels: np.ndarray,
        train_counterfactual_returns: np.ndarray | None = None,
        val_counterfactual_returns: np.ndarray | None = None,
    ) -> None:
        self.scaler = StandardScaler()
        row_mask = train_available.reshape(-1)
        flat_features = train_features.reshape(-1, train_features.shape[-1])[row_mask]
        self.scaler.fit(flat_features)
        train_x = self._transform(train_features)
        val_x = self._transform(val_features)

        torch.manual_seed(self.random_state)
        input_dim = train_x.shape[-1] + train_regime_posteriors.shape[-1] + 1
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_state = deepcopy(self.model.state_dict())
        best_val = -np.inf
        patience = 0
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = -self._sequence_objective(
                train_x,
                train_returns,
                train_available,
                train_regime_posteriors,
                train_regime_labels,
                train_counterfactual_returns,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                val_score = float(
                    self._sequence_objective(
                        val_x,
                        val_returns,
                        val_available,
                        val_regime_posteriors,
                        val_regime_labels,
                        val_counterfactual_returns,
                    ).item()
                )
            if val_score > best_val:
                best_val = val_score
                best_state = deepcopy(self.model.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= 14:
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
        regime_posteriors: np.ndarray,
        regime_labels: np.ndarray,
        counterfactual_returns: np.ndarray | None,
    ) -> torch.Tensor:
        prev_weights = torch.zeros(features.shape[1], dtype=torch.float32)
        rewards: list[torch.Tensor] = []
        future_tensor = torch.tensor(future_returns, dtype=torch.float32)
        available_tensor = torch.tensor(available.astype(float), dtype=torch.float32)
        posterior_tensor = torch.tensor(regime_posteriors, dtype=torch.float32)
        cf_tensor = (
            torch.tensor(counterfactual_returns, dtype=torch.float32)
            if counterfactual_returns is not None
            else None
        )
        cf_terms: list[torch.Tensor] = []
        for idx in range(features.shape[0]):
            regime_repeated = posterior_tensor[idx].repeat(features.shape[1], 1)
            prev_column = prev_weights.unsqueeze(-1)
            inputs = torch.cat([features[idx], regime_repeated, prev_column], dim=1)
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
            if cf_tensor is not None:
                cf_terms.append(torch.min(torch.matmul(cf_tensor[idx], weights)))
        reward_tensor = torch.stack(rewards)
        objective = reward_tensor.mean()
        regime_means = []
        for regime in np.unique(regime_labels):
            regime_mask = torch.tensor(regime_labels == regime, dtype=torch.bool)
            if int(regime_mask.sum()) > 0:
                regime_means.append(reward_tensor[regime_mask].mean())
        if regime_means:
            stacked = torch.stack(regime_means)
            objective = objective - self.robustness_lambda * torch.std(stacked, correction=0)
            objective = objective + self.min_regime_bonus * torch.min(stacked)
        if cf_terms:
            objective = objective + self.counterfactual_weight * torch.stack(cf_terms).mean()
        return objective

    def predict(
        self,
        features: np.ndarray,
        available: np.ndarray,
        prev_weights: np.ndarray,
        regime_posterior: np.ndarray,
        **_: object,
    ) -> np.ndarray:
        transformed = self.scaler.transform(features)
        regime_repeated = np.repeat(regime_posterior[None, :], transformed.shape[0], axis=0)
        prev_column = prev_weights[:, None]
        inputs = torch.tensor(
            np.concatenate([transformed, regime_repeated, prev_column], axis=1),
            dtype=torch.float32,
        )
        with torch.no_grad():
            scores = self.model(inputs).squeeze(-1).numpy() + self.persistence_scale * prev_weights
        return weights_from_scores(scores, available, top_k=self.top_k, max_weight=self.max_weight)


@dataclass(slots=True)
class OracleRegimePolicy:
    top_k: int
    max_weight: float
    transaction_cost: float = 0.0

    name: str = "oracle_regime"

    def fit(self, **_: object) -> None:
        return

    def predict(
        self,
        features: np.ndarray,
        available: np.ndarray,
        prev_weights: np.ndarray,
        *,
        oracle_scores: np.ndarray,
        **_: object,
    ) -> np.ndarray:
        del features, prev_weights
        return weights_from_scores(oracle_scores, available, top_k=self.top_k, max_weight=self.max_weight)
