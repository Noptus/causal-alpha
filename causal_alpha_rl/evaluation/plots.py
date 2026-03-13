"""Plotting utilities for experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def _method_from_column(column: str) -> str:
    method = column.rsplit("_seed", 1)[0]
    if method.startswith("fold") and "_" in method:
        method = method.split("_", 1)[1]
    return method


def aggregate_return_series(return_series: pd.DataFrame) -> pd.DataFrame:
    grouped: dict[str, list[str]] = {}
    for column in return_series.columns:
        grouped.setdefault(_method_from_column(column), []).append(column)
    aggregated = {}
    for method, columns in grouped.items():
        aggregated[method] = return_series[columns].mean(axis=1)
    return pd.DataFrame(aggregated, index=return_series.index)


def plot_cumulative_returns(return_series: pd.DataFrame, output_path: Path, *, title: str) -> None:
    cumulative = (1.0 + aggregate_return_series(return_series)).cumprod() - 1.0
    plt.figure(figsize=(12, 7))
    for column in cumulative.columns:
        plt.plot(cumulative.index, cumulative[column], label=column)
    plt.title(title)
    plt.ylabel("Cumulative Net Return")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_seed_boxplot(summary: pd.DataFrame, metric: str, output_path: Path, *, title: str) -> None:
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=summary, x="method", y=metric)
    plt.xticks(rotation=20)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_turnover_scatter(summary: pd.DataFrame, output_path: Path, *, title: str) -> None:
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=summary, x="avg_turnover", y="sharpe", hue="method", s=120)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_importance_heatmap(importance: pd.DataFrame, output_path: Path, *, title: str) -> None:
    matrix = importance.copy()
    if isinstance(matrix.columns, pd.MultiIndex):
        matrix = matrix.groupby(level=0, axis=1).mean()
    top_factors = matrix.abs().mean(axis=1).sort_values(ascending=False).head(15).index
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix.loc[top_factors], cmap="RdBu_r", center=0.0)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
