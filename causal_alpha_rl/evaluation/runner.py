"""Dataset conversion and experiment execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causal_alpha_rl.algorithms.baselines.contextual_bandit import ContextualBanditBaseline
from causal_alpha_rl.algorithms.baselines.correlation import RollingRankBaseline
from causal_alpha_rl.algorithms.baselines.elastic_net import ElasticNetFactorSelector
from causal_alpha_rl.algorithms.baselines.standard_policy import DirectPolicyNetwork
from causal_alpha_rl.algorithms.causal_rl.policy import (
    CausalPolicyNetwork,
    LegacyCausalPolicyNetwork,
    OracleRegimePolicy,
)
from causal_alpha_rl.envs.allocation import evaluate_weight_path
from causal_alpha_rl.evaluation.metrics import factor_discovery_metrics, summarize_backtest
from causal_alpha_rl.scm.regime import causal_regime_posteriors, fit_regime_model


FEATURE_COLUMNS = [
    "ret_1m",
    "ret_3m",
    "ret_6m",
    "ret_12m",
    "vol_12m",
    "downside_12m",
    "drawdown_12m",
    "inflation_yoy",
    "unemployment",
    "fed_funds",
    "term_spread",
    "credit_spread",
    "factor_mean_1m",
    "factor_dispersion_1m",
    "factor_breadth_1m",
    "factor_mean_12m",
    "factor_vol_12m",
]
REGIME_FEATURE_COLUMNS = [
    "inflation_yoy",
    "unemployment",
    "fed_funds",
    "term_spread",
    "credit_spread",
    "factor_dispersion_1m",
    "factor_vol_12m",
]


@dataclass(slots=True)
class SequenceDataset:
    dates: list[pd.Timestamp]
    factor_ids: list[str]
    features: np.ndarray
    future_returns: np.ndarray
    available: np.ndarray
    date_frame: pd.DataFrame
    factor_metadata: pd.DataFrame
    true_regime: np.ndarray | None = None
    counterfactual_returns: np.ndarray | None = None


def build_sequence_dataset(panel: pd.DataFrame) -> SequenceDataset:
    dates = sorted(pd.to_datetime(panel["date"]).unique().tolist())
    factor_ids = sorted(panel["factor_id"].unique().tolist())
    date_index = pd.Index(dates, name="date")
    factor_index = pd.Index(factor_ids, name="factor_id")

    feature_arrays = []
    for column in FEATURE_COLUMNS:
        matrix = (
            panel.pivot(index="date", columns="factor_id", values=column)
            .reindex(index=date_index, columns=factor_index)
            .to_numpy()
        )
        feature_arrays.append(matrix)
    features = np.stack(feature_arrays, axis=-1)
    future_returns = (
        panel.pivot(index="date", columns="factor_id", values="ret_fwd_1m")
        .reindex(index=date_index, columns=factor_index)
        .to_numpy()
    )
    available = (
        panel.pivot(index="date", columns="factor_id", values="available")
        .reindex(index=date_index, columns=factor_index)
        .fillna(0)
        .to_numpy()
        .astype(bool)
    )
    valid_feature_mask = ~np.isnan(features).any(axis=-1)
    valid_return_mask = ~np.isnan(future_returns)
    available = available & valid_feature_mask & valid_return_mask
    features = np.nan_to_num(features, nan=0.0)
    future_returns = np.nan_to_num(future_returns, nan=0.0)

    date_columns = ["date"] + [column for column in REGIME_FEATURE_COLUMNS if column in panel.columns]
    for optional in ["true_regime", "stress_regime"]:
        if optional in panel.columns:
            date_columns.append(optional)
    date_frame = panel[date_columns].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    factor_meta_cols = ["factor_id"] + [
        column
        for column in ["factor_group", "preferred_regime", "true_causal", "true_causal_score"]
        if column in panel.columns
    ]
    factor_metadata = (
        panel[factor_meta_cols].drop_duplicates("factor_id").sort_values("factor_id").reset_index(drop=True)
    )

    cf_columns = sorted([column for column in panel.columns if column.startswith("cf_ret_regime_")])
    counterfactual_returns = None
    if cf_columns:
        cf_arrays = []
        for column in cf_columns:
            matrix = (
                panel.pivot(index="date", columns="factor_id", values=column)
                .reindex(index=date_index, columns=factor_index)
                .to_numpy()
            )
            cf_arrays.append(np.nan_to_num(matrix, nan=0.0))
        counterfactual_returns = np.stack(cf_arrays, axis=1)

    true_regime = None
    if "true_regime" in date_frame.columns:
        true_regime = date_frame["true_regime"].to_numpy(dtype=int)

    return SequenceDataset(
        dates=dates,
        factor_ids=factor_ids,
        features=features,
        future_returns=future_returns,
        available=available,
        date_frame=date_frame,
        factor_metadata=factor_metadata,
        true_regime=true_regime,
        counterfactual_returns=counterfactual_returns,
    )


def _run_policy(
    dataset: SequenceDataset,
    test_idx: np.ndarray,
    method: Any,
    *,
    regime_posteriors: np.ndarray | None = None,
    regime_labels_full: np.ndarray | None = None,
    stress_mask_full: np.ndarray | None = None,
    oracle: bool = False,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    prev_weights = np.zeros(len(dataset.factor_ids), dtype=float)
    weights = []
    returns = []
    turnovers = []
    for idx in test_idx:
        kwargs: dict[str, Any] = {}
        if regime_posteriors is not None:
            kwargs["regime_posterior"] = regime_posteriors[idx]
        if oracle and dataset.counterfactual_returns is not None and dataset.true_regime is not None:
            kwargs["oracle_scores"] = dataset.counterfactual_returns[idx, dataset.true_regime[idx]]
        current_weights = method.predict(
            dataset.features[idx],
            dataset.available[idx],
            prev_weights,
            **kwargs,
        )
        turnover = np.abs(current_weights - prev_weights).sum()
        reward = float(np.dot(current_weights, dataset.future_returns[idx]) - method.transaction_cost * turnover) if hasattr(method, "transaction_cost") else float(np.dot(current_weights, dataset.future_returns[idx]) - 0.0)
        weights.append(current_weights)
        returns.append(reward)
        turnovers.append(turnover)
        prev_weights = current_weights
    weight_matrix = np.asarray(weights)
    turnover_array = np.asarray(turnovers)
    if hasattr(method, "transaction_cost"):
        backtest = evaluate_weight_path(weight_matrix, dataset.future_returns[test_idx], transaction_cost=method.transaction_cost)
        returns_array = backtest.net_returns
        turnover_array = backtest.turnovers
    else:
        returns_array = np.asarray(returns)
    regime_labels = dataset.true_regime[test_idx] if dataset.true_regime is not None else regime_labels_full[test_idx] if regime_labels_full is not None else None
    stress_mask = (
        dataset.date_frame.loc[test_idx, "stress_regime"].to_numpy(dtype=bool)
        if "stress_regime" in dataset.date_frame.columns
        else stress_mask_full[test_idx] if stress_mask_full is not None else None
    )
    metrics = summarize_backtest(
        returns_array,
        turnover_array,
        regime_labels=regime_labels,
        stress_mask=stress_mask,
    )
    return metrics, weight_matrix, returns_array


def run_fold(
    dataset: SequenceDataset,
    split: dict[str, tuple[int, int]],
    *,
    seeds: list[int],
    top_k: int,
    max_weight: float,
    transaction_cost: float,
    kind: str,
    method_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_idx = np.arange(*split["train"])
    val_idx = np.arange(*split["val"])
    test_idx = np.arange(*split["test"])
    method_overrides = method_overrides or {}

    if kind == "synthetic" and dataset.true_regime is not None:
        n_regimes = int(dataset.true_regime.max()) + 1
        regime_posteriors = np.eye(n_regimes)[dataset.true_regime]
        regime_labels = dataset.true_regime.copy()
        stress_mask_full = regime_labels == 2
    else:
        regime_fit = fit_regime_model(
            dataset.date_frame.iloc[train_idx],
            REGIME_FEATURE_COLUMNS,
            n_regimes=3,
            seed=seeds[0],
        )
        regime_frame = causal_regime_posteriors(regime_fit, dataset.date_frame)
        posterior_cols = [column for column in regime_frame.columns if column.startswith("regime_p_")]
        regime_posteriors = regime_frame[posterior_cols].fillna(1.0 / len(posterior_cols)).to_numpy()
        regime_labels = regime_posteriors.argmax(axis=1)
        stress_state = (
            dataset.date_frame.iloc[train_idx]
            .assign(regime_state=regime_labels[train_idx])
            .groupby("regime_state")["factor_dispersion_1m"]
            .mean()
            .idxmax()
        )
        stress_mask_full = regime_labels == stress_state
        if stress_mask_full[test_idx].sum() < max(6, int(0.1 * len(test_idx))):
            stress_prob = regime_posteriors[:, stress_state]
            threshold = np.quantile(stress_prob[test_idx], 0.75)
            stress_mask_full = stress_prob >= threshold

    factor_eye = np.eye(len(dataset.factor_ids))

    def _causal_overrides(method_name: str) -> dict[str, Any]:
        overrides = dict(method_overrides.get(method_name, {}))
        for grid_name in ["blend_grid", "instability_grid", "persistence_grid", "invariant_weight_grid"]:
            if grid_name in overrides:
                overrides[grid_name] = tuple(float(value) for value in overrides[grid_name])
        return overrides

    deterministic_methods = [
        RollingRankBaseline(top_k=top_k, max_weight=max_weight, transaction_cost=transaction_cost),
        ElasticNetFactorSelector(top_k=top_k, max_weight=max_weight, random_state=seeds[0], factor_eye=factor_eye, transaction_cost=transaction_cost),
        ContextualBanditBaseline(top_k=top_k, max_weight=max_weight, factor_eye=factor_eye, transaction_cost=transaction_cost),
    ]
    causal_methods = []
    primary_causal_kwargs = {"name": "causal_rl"}
    primary_causal_kwargs.update(_causal_overrides("causal_rl"))
    causal_methods.append(
        CausalPolicyNetwork(
            top_k=top_k,
            max_weight=max_weight,
            transaction_cost=transaction_cost,
            random_state=seeds[0],
            factor_eye=factor_eye,
            **primary_causal_kwargs,
        )
    )
    no_instability_kwargs = {"name": "causal_rl_no_instability", "instability_grid": (0.0,), "compare_legacy": False}
    no_instability_kwargs.update(_causal_overrides("causal_rl_no_instability"))
    causal_methods.append(
        CausalPolicyNetwork(
            top_k=top_k,
            max_weight=max_weight,
            transaction_cost=transaction_cost,
            random_state=seeds[0],
            factor_eye=factor_eye,
            **no_instability_kwargs,
        )
    )
    results: list[dict[str, Any]] = []

    for method in deterministic_methods:
        if hasattr(method, "fit"):
            if method.name == "correlation_rank":
                method.fit(dataset.future_returns[train_idx], dataset.available[train_idx])
            else:
                method.fit(
                    dataset.features[train_idx],
                    dataset.future_returns[train_idx],
                    dataset.available[train_idx],
                )
        metrics, weights, returns = _run_policy(
            dataset,
            test_idx,
            method,
            regime_labels_full=regime_labels,
            stress_mask_full=stress_mask_full,
        )
        results.append(
            {
                "method": method.name,
                "seed": seeds[0],
                "metrics": metrics,
                "weights": weights,
                "returns": returns,
                "fit_summary": getattr(method, "fit_summary_", {}),
            }
        )

    for seed in seeds:
        standard = DirectPolicyNetwork(
            top_k=top_k,
            max_weight=max_weight,
            transaction_cost=transaction_cost,
            random_state=seed,
        )
        standard.fit(
            train_features=dataset.features[train_idx],
            train_returns=dataset.future_returns[train_idx],
            train_available=dataset.available[train_idx],
            val_features=dataset.features[val_idx],
            val_returns=dataset.future_returns[val_idx],
            val_available=dataset.available[val_idx],
        )
        metrics, weights, returns = _run_policy(
            dataset,
            test_idx,
            standard,
            regime_labels_full=regime_labels,
            stress_mask_full=stress_mask_full,
        )
        results.append(
            {
                "method": standard.name,
                "seed": seed,
                "metrics": metrics,
                "weights": weights,
                "returns": returns,
                "fit_summary": getattr(standard, "fit_summary_", {}),
            }
        )

        legacy_causal = LegacyCausalPolicyNetwork(
            top_k=top_k,
            max_weight=max_weight,
            transaction_cost=transaction_cost,
            random_state=seed,
            **_causal_overrides("causal_rl_legacy"),
        )
        legacy_causal.fit(
            train_features=dataset.features[train_idx],
            train_returns=dataset.future_returns[train_idx],
            train_available=dataset.available[train_idx],
            val_features=dataset.features[val_idx],
            val_returns=dataset.future_returns[val_idx],
            val_available=dataset.available[val_idx],
            train_regime_posteriors=regime_posteriors[train_idx],
            val_regime_posteriors=regime_posteriors[val_idx],
            train_regime_labels=regime_labels[train_idx],
            val_regime_labels=regime_labels[val_idx],
            train_counterfactual_returns=dataset.counterfactual_returns[train_idx] if dataset.counterfactual_returns is not None else None,
            val_counterfactual_returns=dataset.counterfactual_returns[val_idx] if dataset.counterfactual_returns is not None else None,
        )
        metrics, weights, returns = _run_policy(
            dataset,
            test_idx,
            legacy_causal,
            regime_posteriors=regime_posteriors,
            regime_labels_full=regime_labels,
            stress_mask_full=stress_mask_full,
        )
        results.append(
            {
                "method": legacy_causal.name,
                "seed": seed,
                "metrics": metrics,
                "weights": weights,
                "returns": returns,
                "fit_summary": getattr(legacy_causal, "fit_summary_", {}),
            }
        )

    for causal_method in causal_methods:
        causal_method.fit(
            train_features=dataset.features[train_idx],
            train_returns=dataset.future_returns[train_idx],
            train_available=dataset.available[train_idx],
            val_features=dataset.features[val_idx],
            val_returns=dataset.future_returns[val_idx],
            val_available=dataset.available[val_idx],
            train_regime_posteriors=regime_posteriors[train_idx],
            val_regime_posteriors=regime_posteriors[val_idx],
            train_regime_labels=regime_labels[train_idx],
            val_regime_labels=regime_labels[val_idx],
            train_counterfactual_returns=dataset.counterfactual_returns[train_idx] if dataset.counterfactual_returns is not None else None,
            val_counterfactual_returns=dataset.counterfactual_returns[val_idx] if dataset.counterfactual_returns is not None else None,
        )
        metrics, weights, returns = _run_policy(
            dataset,
            test_idx,
            causal_method,
            regime_posteriors=regime_posteriors,
            regime_labels_full=regime_labels,
            stress_mask_full=stress_mask_full,
        )
        results.append(
            {
                "method": causal_method.name,
                "seed": seeds[0],
                "metrics": metrics,
                "weights": weights,
                "returns": returns,
                "fit_summary": getattr(causal_method, "fit_summary_", {}),
            }
        )

    if kind == "synthetic" and dataset.counterfactual_returns is not None and dataset.true_regime is not None:
        oracle = OracleRegimePolicy(top_k=top_k, max_weight=max_weight, transaction_cost=transaction_cost)
        oracle.fit()
        metrics, weights, returns = _run_policy(
            dataset,
            test_idx,
            oracle,
            regime_labels_full=regime_labels,
            stress_mask_full=stress_mask_full,
            oracle=True,
        )
        results.append({"method": oracle.name, "seed": seeds[0], "metrics": metrics, "weights": weights, "returns": returns})

    summary_records = []
    importance_records = []
    return_series = pd.DataFrame(index=[dataset.dates[idx] for idx in test_idx])
    for record in results:
        summary = {"method": record["method"], "seed": record["seed"], **record["metrics"]}
        for key, value in record.get("fit_summary", {}).items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                summary[key] = float(value)
        summary_records.append(summary)
        importance = pd.Series(record["weights"].mean(axis=0), index=dataset.factor_ids, name=record["method"])
        importance_records.append(importance)
        return_series[f"{record['method']}_seed{record['seed']}"] = record["returns"]
    summary_frame = pd.DataFrame(summary_records)
    importance_frame = pd.concat(importance_records, axis=1)
    outputs: dict[str, Any] = {
        "summary": summary_frame,
        "importance": importance_frame,
        "return_series": return_series,
        "test_dates": [str(pd.Timestamp(dataset.dates[idx]).date()) for idx in test_idx],
        "stress_mask": dataset.date_frame.loc[test_idx, "stress_regime"].tolist()
        if "stress_regime" in dataset.date_frame.columns
        else None,
        "regime_labels": dataset.true_regime[test_idx].tolist() if dataset.true_regime is not None else None,
    }
    if kind == "synthetic":
        discovery_rows = []
        for method_name in importance_frame.columns.unique():
            method_importances = importance_frame[method_name]
            if isinstance(method_importances, pd.DataFrame):
                method_importances = method_importances.mean(axis=1)
            discovery_rows.append(
                {"method": method_name, **factor_discovery_metrics(method_importances, dataset.factor_metadata)}
            )
        outputs["discovery"] = pd.DataFrame(discovery_rows)
    return outputs


def aggregate_fold_outputs(fold_outputs: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    summaries = []
    discoveries = []
    for fold_id, fold_output in enumerate(fold_outputs):
        summary = fold_output["summary"].copy()
        summary["fold"] = fold_id
        summaries.append(summary)
        if "discovery" in fold_output:
            discovery = fold_output["discovery"].copy()
            discovery["fold"] = fold_id
            discoveries.append(discovery)
    result = {"summary": pd.concat(summaries, ignore_index=True)}
    if discoveries:
        result["discovery"] = pd.concat(discoveries, ignore_index=True)
    return result
