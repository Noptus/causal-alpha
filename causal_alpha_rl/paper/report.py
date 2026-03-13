"""Markdown report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _markdown_table(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: format(value, floatfmt))
    headers = list(formatted.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def write_results_report(
    output_path: Path,
    *,
    synthetic_summary: pd.DataFrame,
    synthetic_discovery: pd.DataFrame,
    real_summary: pd.DataFrame,
    notes: list[str],
) -> None:
    synthetic_table = (
        synthetic_summary.groupby("method")[["annual_return", "sharpe", "worst_regime_sharpe", "avg_turnover"]]
        .mean()
        .reset_index()
        .sort_values("sharpe", ascending=False)
    )
    discovery_table = (
        synthetic_discovery.groupby("method")[["precision_at_5", "fdr_at_10", "rank_corr"]]
        .mean()
        .reset_index()
        .sort_values("precision_at_5", ascending=False)
    )
    real_table = (
        real_summary.groupby("method")[["annual_return", "sharpe", "stress_sharpe", "max_drawdown", "avg_turnover"]]
        .mean()
        .reset_index()
        .sort_values("stress_sharpe", ascending=False)
    )
    content = "\n".join(
        [
            "# Causal RL for Robust Alpha Discovery",
            "",
            "## Claim",
            "Causal regime-aware policy learning is strongest on the synthetic SCM benchmark, where it recovers invariant factors and nearly matches the oracle Sharpe. On the real factor-library benchmark, the causal policy improves stress-regime Sharpe versus the standard policy learner, but the simple correlation ranking baseline remains the strongest overall performer.",
            "",
            "## Synthetic Performance",
            _markdown_table(synthetic_table),
            "",
            "## Synthetic Factor Discovery",
            _markdown_table(discovery_table),
            "",
            "## Real Data Performance",
            _markdown_table(real_table),
            "",
            "## Figures",
            "- `figures/generated/synthetic_cumulative_returns.png`",
            "- `figures/generated/synthetic_seed_boxplot.png`",
            "- `figures/generated/real_cumulative_returns.png`",
            "- `figures/generated/real_turnover_scatter.png`",
            "- `figures/generated/factor_importance_heatmap.png`",
            "",
            "## Notes",
            *[f"- {note}" for note in notes],
            "- The real benchmark should be interpreted as external validity with mixed evidence, not as definitive dominance over simpler factor-ranking methods.",
            "",
        ]
    )
    output_path.write_text(content, encoding="utf-8")
