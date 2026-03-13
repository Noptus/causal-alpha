"""Backtest and factor discovery metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _annualize(monthly_mean: float) -> float:
    return (1.0 + monthly_mean) ** 12 - 1.0


def sharpe_ratio(returns: np.ndarray) -> float:
    volatility = returns.std(ddof=1)
    if volatility <= 1e-8:
        return 0.0
    return float(np.sqrt(12.0) * returns.mean() / volatility)


def sortino_ratio(returns: np.ndarray) -> float:
    downside = returns[returns < 0]
    downside_vol = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if downside_vol <= 1e-8:
        return 0.0
    return float(np.sqrt(12.0) * returns.mean() / downside_vol)


def max_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(wealth)
    drawdowns = wealth / np.maximum(running_max, 1e-8) - 1.0
    return float(drawdowns.min())


def summarize_backtest(
    returns: np.ndarray,
    turnovers: np.ndarray,
    *,
    regime_labels: np.ndarray | None = None,
    stress_mask: np.ndarray | None = None,
) -> dict[str, float]:
    summary = {
        "annual_return": _annualize(float(returns.mean())),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "avg_turnover": float(turnovers.mean()),
        "cum_return": float(np.prod(1.0 + returns) - 1.0),
    }
    if regime_labels is not None:
        regime_metrics = []
        for regime in np.unique(regime_labels):
            regime_returns = returns[regime_labels == regime]
            if len(regime_returns) >= 6:
                regime_metrics.append(sharpe_ratio(regime_returns))
        if regime_metrics:
            summary["worst_regime_sharpe"] = float(min(regime_metrics))
    if stress_mask is not None and stress_mask.any():
        summary["stress_sharpe"] = sharpe_ratio(returns[stress_mask])
    return summary


def factor_discovery_metrics(importances: pd.Series, factor_metadata: pd.DataFrame) -> dict[str, float]:
    ranked = importances.sort_values(ascending=False)
    metadata = factor_metadata.set_index("factor_id")
    top5 = metadata.loc[ranked.head(5).index]
    top10 = metadata.loc[ranked.head(10).index]
    truth = metadata["true_causal"].astype(float)
    aligned = truth.reindex(ranked.index).fillna(0.0)
    rho = spearmanr(ranked.to_numpy(), aligned.to_numpy()).statistic
    return {
        "precision_at_5": float(top5["true_causal"].mean()),
        "fdr_at_10": float(1.0 - top10["true_causal"].mean()),
        "rank_corr": float(0.0 if np.isnan(rho) else rho),
    }
