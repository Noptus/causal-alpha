"""Causal regime-aware policy learners."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from causal_alpha_rl.envs.allocation import (
    smooth_weights_from_scores_torch,
    weights_from_scores,
    weights_from_scores_torch,
)
from causal_alpha_rl.evaluation.metrics import summarize_backtest


@dataclass(slots=True)
class _RidgeScoreModel:
    scaler: StandardScaler
    model: RidgeCV
    numeric_dim: int


def _fit_ridge_score_model(
    *,
    features: np.ndarray,
    future_returns: np.ndarray,
    available: np.ndarray,
    factor_eye: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> _RidgeScoreModel:
    row_mask = available.reshape(-1)
    x_numeric = features.reshape(-1, features.shape[-1])[row_mask]
    factor_idx = np.tile(np.arange(features.shape[1]), features.shape[0])[row_mask]
    design = np.concatenate([x_numeric, factor_eye[factor_idx]], axis=1)
    y = future_returns.reshape(-1)[row_mask]
    scaler = StandardScaler()
    design[:, : x_numeric.shape[1]] = scaler.fit_transform(design[:, : x_numeric.shape[1]])
    model = RidgeCV(alphas=np.logspace(-3, 2, 12))
    fit_kwargs: dict[str, np.ndarray] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight.reshape(-1)[row_mask]
    model.fit(design, y, **fit_kwargs)
    return _RidgeScoreModel(scaler=scaler, model=model, numeric_dim=x_numeric.shape[1])


def _predict_ridge_scores(
    score_model: _RidgeScoreModel,
    features: np.ndarray,
    factor_eye: np.ndarray,
) -> np.ndarray:
    design = np.concatenate([features, factor_eye], axis=1).copy()
    design[:, : score_model.numeric_dim] = score_model.scaler.transform(design[:, : score_model.numeric_dim])
    return score_model.model.predict(design)


@dataclass(slots=True)
class LegacyCausalPolicyNetwork:
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
    fit_summary_: dict[str, float] = field(init=False, default_factory=dict)

    name: str = "causal_rl_legacy"

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
        self.fit_summary_ = {
            "selected_blend": 0.0,
            "selected_instability": float(self.robustness_lambda),
            "selected_persistence": float(self.persistence_scale),
            "validation_score": 0.0,
        }
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
class CausalPolicyNetwork:
    top_k: int
    max_weight: float
    transaction_cost: float
    random_state: int
    factor_eye: np.ndarray
    blend_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    instability_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4)
    persistence_grid: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15)
    invariant_weight_grid: tuple[float, ...] = (0.0, 0.05, 0.1)
    validation_cum_weight: float = 0.15
    validation_turnover_weight: float = 0.10
    validation_worst_regime_weight: float = 0.10
    regime_sample_floor: float = 0.20
    refit_on_train_val: bool = False
    compare_legacy: bool = True
    invariant_strength: float = 0.50
    invariant_sign_penalty: float = 0.25
    base_model_: _RidgeScoreModel = field(init=False)
    regime_models_: list[_RidgeScoreModel] = field(init=False)
    invariant_prior_: np.ndarray = field(init=False)
    selected_blend_: float = field(init=False, default=0.0)
    selected_instability_: float = field(init=False, default=0.0)
    selected_persistence_: float = field(init=False, default=0.0)
    selected_invariant_weight_: float = field(init=False, default=0.0)
    fit_summary_: dict[str, float] = field(init=False, default_factory=dict)
    legacy_candidate_: LegacyCausalPolicyNetwork | None = field(init=False, default=None)
    active_strategy_: str = field(init=False, default="hybrid")

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
        self.legacy_candidate_ = None
        self.active_strategy_ = "hybrid"
        self.base_model_, self.regime_models_ = self._fit_models(
            features=train_features,
            future_returns=train_returns,
            available=train_available,
            regime_posteriors=train_regime_posteriors,
        )
        self.invariant_prior_ = self._compute_invariant_prior(
            future_returns=train_returns,
            available=train_available,
            regime_posteriors=train_regime_posteriors,
        )
        base_scores, regime_mix_scores, instability_scores = self._path_components(
            features=val_features,
            regime_posteriors=val_regime_posteriors,
        )
        best_score = -np.inf
        best_metrics: dict[str, float] | None = None
        for blend in self.blend_grid:
            for instability in self.instability_grid:
                for persistence in self.persistence_grid:
                    for invariant_weight in self.invariant_weight_grid:
                        metrics = self._simulate_path(
                            future_returns=val_returns,
                            available=val_available,
                            base_scores=base_scores,
                            regime_mix_scores=regime_mix_scores,
                            instability_scores=instability_scores,
                            regime_labels=val_regime_labels,
                            blend=blend,
                            instability_penalty=instability,
                            persistence=persistence,
                            invariant_weight=invariant_weight,
                        )
                        score = self._validation_score(metrics)
                        if score > best_score + 1e-12 or (
                            abs(score - best_score) <= 1e-12
                            and best_metrics is not None
                            and metrics["avg_turnover"] < best_metrics["avg_turnover"]
                        ):
                            best_score = score
                            best_metrics = metrics
                            self.selected_blend_ = blend
                            self.selected_instability_ = instability
                            self.selected_persistence_ = persistence
                            self.selected_invariant_weight_ = invariant_weight
        hybrid_score = best_score
        hybrid_metrics = best_metrics
        legacy_score = -np.inf
        legacy_metrics: dict[str, float] | None = None
        force_legacy = train_counterfactual_returns is not None and val_counterfactual_returns is not None
        if self.compare_legacy or force_legacy:
            legacy_candidate = LegacyCausalPolicyNetwork(
                top_k=self.top_k,
                max_weight=self.max_weight,
                transaction_cost=self.transaction_cost,
                random_state=self.random_state,
                robustness_lambda=0.8 if train_counterfactual_returns is not None else 0.4,
                min_regime_bonus=0.5 if train_counterfactual_returns is not None else 0.2,
                counterfactual_weight=0.5 if train_counterfactual_returns is not None else 0.25,
            )
            legacy_candidate.fit(
                train_features=train_features,
                train_returns=train_returns,
                train_available=train_available,
                val_features=val_features,
                val_returns=val_returns,
                val_available=val_available,
                train_regime_posteriors=train_regime_posteriors,
                val_regime_posteriors=val_regime_posteriors,
                train_regime_labels=train_regime_labels,
                val_regime_labels=val_regime_labels,
                train_counterfactual_returns=train_counterfactual_returns,
                val_counterfactual_returns=val_counterfactual_returns,
            )
            legacy_metrics = self._evaluate_legacy_candidate(
                legacy_candidate=legacy_candidate,
                features=val_features,
                future_returns=val_returns,
                available=val_available,
                regime_posteriors=val_regime_posteriors,
                regime_labels=val_regime_labels,
            )
            legacy_score = self._validation_score(legacy_metrics)
            if force_legacy or legacy_score > hybrid_score:
                self.legacy_candidate_ = legacy_candidate
                self.active_strategy_ = "legacy"
                self.fit_summary_ = {
                    "selected_blend": 0.0,
                    "selected_instability": 0.0,
                    "selected_persistence": float(legacy_candidate.persistence_scale),
                    "selected_invariant_weight": 0.0,
                    "validation_score": float(legacy_score),
                    "validation_sharpe": float(legacy_metrics["sharpe"]),
                    "validation_cum_return": float(legacy_metrics["cum_return"]),
                    "validation_worst_regime_sharpe": float(legacy_metrics.get("worst_regime_sharpe", 0.0)),
                    "validation_avg_turnover": float(legacy_metrics["avg_turnover"]),
                    "selected_strategy_hybrid": 0.0,
                    "hybrid_validation_score": float(hybrid_score),
                    "legacy_validation_score": float(legacy_score),
                    "refit_on_train_val": float(int(self.refit_on_train_val)),
                }
                return
        if self.refit_on_train_val:
            combined_features = np.concatenate([train_features, val_features], axis=0)
            combined_returns = np.concatenate([train_returns, val_returns], axis=0)
            combined_available = np.concatenate([train_available, val_available], axis=0)
            combined_posteriors = np.concatenate([train_regime_posteriors, val_regime_posteriors], axis=0)
            self.base_model_, self.regime_models_ = self._fit_models(
                features=combined_features,
                future_returns=combined_returns,
                available=combined_available,
                regime_posteriors=combined_posteriors,
            )
            self.invariant_prior_ = self._compute_invariant_prior(
                future_returns=combined_returns,
                available=combined_available,
                regime_posteriors=combined_posteriors,
            )
        self.fit_summary_ = {
            "selected_blend": float(self.selected_blend_),
            "selected_instability": float(self.selected_instability_),
            "selected_persistence": float(self.selected_persistence_),
            "selected_invariant_weight": float(self.selected_invariant_weight_),
            "validation_score": float(best_score),
            "validation_sharpe": float(0.0 if best_metrics is None else best_metrics["sharpe"]),
            "validation_cum_return": float(0.0 if best_metrics is None else best_metrics["cum_return"]),
            "validation_worst_regime_sharpe": float(
                0.0 if best_metrics is None else best_metrics.get("worst_regime_sharpe", 0.0)
            ),
            "validation_avg_turnover": float(0.0 if best_metrics is None else best_metrics["avg_turnover"]),
            "selected_strategy_hybrid": 1.0,
            "hybrid_validation_score": float(hybrid_score),
            "legacy_validation_score": float(legacy_score),
            "refit_on_train_val": float(int(self.refit_on_train_val)),
        }

    def _fit_models(
        self,
        *,
        features: np.ndarray,
        future_returns: np.ndarray,
        available: np.ndarray,
        regime_posteriors: np.ndarray,
    ) -> tuple[_RidgeScoreModel, list[_RidgeScoreModel]]:
        base_model = _fit_ridge_score_model(
            features=features,
            future_returns=future_returns,
            available=available,
            factor_eye=self.factor_eye,
        )
        regime_models: list[_RidgeScoreModel] = []
        for regime_id in range(regime_posteriors.shape[1]):
            sample_weight = np.repeat(
                (self.regime_sample_floor + regime_posteriors[:, regime_id])[:, None],
                features.shape[1],
                axis=1,
            )
            regime_models.append(
                _fit_ridge_score_model(
                    features=features,
                    future_returns=future_returns,
                    available=available,
                    factor_eye=self.factor_eye,
                    sample_weight=sample_weight,
                )
            )
        return base_model, regime_models

    def _compute_invariant_prior(
        self,
        *,
        future_returns: np.ndarray,
        available: np.ndarray,
        regime_posteriors: np.ndarray,
    ) -> np.ndarray:
        regime_means: list[np.ndarray] = []
        availability = available.astype(float)
        for regime_id in range(regime_posteriors.shape[1]):
            weights = regime_posteriors[:, regime_id][:, None] * availability
            denom = np.maximum(weights.sum(axis=0), 1e-8)
            regime_means.append((weights * future_returns).sum(axis=0) / denom)
        stacked = np.stack(regime_means, axis=0)
        base = stacked.mean(axis=0)
        instability = stacked.std(axis=0)
        sign_penalty = (np.ptp(np.sign(stacked + 1e-12), axis=0) > 0).astype(float)
        prior = base - self.invariant_strength * instability - self.invariant_sign_penalty * sign_penalty * instability
        prior = (prior - prior.mean()) / max(prior.std(ddof=0), 1e-8)
        return prior

    def _path_components(
        self,
        *,
        features: np.ndarray,
        regime_posteriors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base_scores = np.zeros(features.shape[:2], dtype=float)
        regime_mix_scores = np.zeros_like(base_scores)
        instability_scores = np.zeros_like(base_scores)
        for idx in range(features.shape[0]):
            base_scores[idx] = _predict_ridge_scores(self.base_model_, features[idx], self.factor_eye)
            regime_scores = np.stack(
                [
                    _predict_ridge_scores(score_model, features[idx], self.factor_eye)
                    for score_model in self.regime_models_
                ],
                axis=0,
            )
            regime_mix_scores[idx] = regime_posteriors[idx] @ regime_scores
            instability_scores[idx] = regime_scores.std(axis=0)
        return base_scores, regime_mix_scores, instability_scores

    def _simulate_path(
        self,
        *,
        future_returns: np.ndarray,
        available: np.ndarray,
        base_scores: np.ndarray,
        regime_mix_scores: np.ndarray,
        instability_scores: np.ndarray,
        regime_labels: np.ndarray,
        blend: float,
        instability_penalty: float,
        persistence: float,
        invariant_weight: float,
    ) -> dict[str, float]:
        prev_weights = np.zeros(base_scores.shape[1], dtype=float)
        weight_path = np.zeros_like(base_scores)
        for idx in range(base_scores.shape[0]):
            scores = (
                base_scores[idx]
                + blend * (regime_mix_scores[idx] - base_scores[idx])
                - instability_penalty * instability_scores[idx]
                + persistence * prev_weights
                + invariant_weight * self.invariant_prior_
            )
            current_weights = weights_from_scores(
                scores,
                available[idx],
                top_k=self.top_k,
                max_weight=self.max_weight,
            )
            weight_path[idx] = current_weights
            prev_weights = current_weights
        returns = np.zeros(weight_path.shape[0], dtype=float)
        turnovers = np.zeros_like(returns)
        prev_weights = np.zeros(weight_path.shape[1], dtype=float)
        for idx in range(weight_path.shape[0]):
            turnovers[idx] = np.abs(weight_path[idx] - prev_weights).sum()
            returns[idx] = float(np.dot(weight_path[idx], future_returns[idx]) - self.transaction_cost * turnovers[idx])
            prev_weights = weight_path[idx]
        return summarize_backtest(returns, turnovers, regime_labels=regime_labels)

    def _validation_score(self, metrics: dict[str, float]) -> float:
        return float(
            metrics["sharpe"]
            + self.validation_cum_weight * metrics["cum_return"]
            + self.validation_worst_regime_weight * metrics.get("worst_regime_sharpe", 0.0)
            - self.validation_turnover_weight * metrics["avg_turnover"]
        )

    def _evaluate_legacy_candidate(
        self,
        *,
        legacy_candidate: LegacyCausalPolicyNetwork,
        features: np.ndarray,
        future_returns: np.ndarray,
        available: np.ndarray,
        regime_posteriors: np.ndarray,
        regime_labels: np.ndarray,
    ) -> dict[str, float]:
        prev_weights = np.zeros(features.shape[1], dtype=float)
        returns = np.zeros(features.shape[0], dtype=float)
        turnovers = np.zeros(features.shape[0], dtype=float)
        for idx in range(features.shape[0]):
            current_weights = legacy_candidate.predict(
                features[idx],
                available[idx],
                prev_weights,
                regime_posterior=regime_posteriors[idx],
            )
            turnover = np.abs(current_weights - prev_weights).sum()
            returns[idx] = float(np.dot(current_weights, future_returns[idx]) - self.transaction_cost * turnover)
            turnovers[idx] = turnover
            prev_weights = current_weights
        return summarize_backtest(returns, turnovers, regime_labels=regime_labels)

    def predict(
        self,
        features: np.ndarray,
        available: np.ndarray,
        prev_weights: np.ndarray,
        regime_posterior: np.ndarray,
        **_: object,
    ) -> np.ndarray:
        if self.active_strategy_ == "legacy":
            assert self.legacy_candidate_ is not None
            return self.legacy_candidate_.predict(
                features,
                available,
                prev_weights,
                regime_posterior=regime_posterior,
            )
        posterior = np.asarray(regime_posterior, dtype=float)
        posterior = np.nan_to_num(posterior, nan=1.0 / max(len(self.regime_models_), 1))
        posterior_sum = posterior.sum()
        if posterior_sum <= 0.0:
            posterior = np.full(len(self.regime_models_), 1.0 / max(len(self.regime_models_), 1))
        else:
            posterior = posterior / posterior_sum
        base_scores = _predict_ridge_scores(self.base_model_, features, self.factor_eye)
        regime_scores = np.stack(
            [_predict_ridge_scores(score_model, features, self.factor_eye) for score_model in self.regime_models_],
            axis=0,
        )
        regime_mix_scores = posterior @ regime_scores
        instability_scores = regime_scores.std(axis=0)
        scores = (
            base_scores
            + self.selected_blend_ * (regime_mix_scores - base_scores)
            - self.selected_instability_ * instability_scores
            + self.selected_persistence_ * prev_weights
            + self.selected_invariant_weight_ * self.invariant_prior_
        )
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
