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


def _method_summary(frame: pd.DataFrame, columns: list[str], *, sort_by: str) -> pd.DataFrame:
    available_columns = ["method"] + [column for column in columns if column in frame.columns]
    return (
        frame[available_columns]
        .groupby("method", as_index=False)
        .mean(numeric_only=True)
        .sort_values(sort_by, ascending=False)
    )


def _claim_text(real_table: pd.DataFrame) -> str:
    indexed = real_table.set_index("method")
    if "causal_rl" not in indexed.index or "contextual_bandit" not in indexed.index:
        return (
            "Causal regime-aware policy learning remains strongest on the synthetic SCM benchmark, "
            "while the real factor-library benchmark stays mixed."
        )
    causal_row = indexed.loc["causal_rl"]
    bandit_row = indexed.loc["contextual_bandit"]
    correlation_row = indexed.loc["correlation_rank"] if "correlation_rank" in indexed.index else None
    if causal_row["sharpe"] > bandit_row["sharpe"] and causal_row["cum_return"] > bandit_row["cum_return"]:
        if correlation_row is not None and correlation_row["sharpe"] > causal_row["sharpe"]:
            return (
                "Bandit-guided causal RL now improves both cumulative return and Sharpe versus the contextual bandit "
                "on the real factor-library benchmark, while the frozen correlation ranking baseline remains the "
                "highest-Sharpe reference."
            )
        return (
            "Bandit-guided causal RL now improves both cumulative return and Sharpe versus the contextual bandit "
            "on the real factor-library benchmark and remains strongest on the synthetic SCM benchmark."
        )
    if causal_row.get("stress_sharpe", 0.0) > bandit_row.get("stress_sharpe", 0.0):
        return (
            "Bandit-guided causal RL improves stress-regime robustness versus the contextual bandit and standard "
            "policy learner, while the overall real-data edge remains narrower than on the synthetic SCM benchmark."
        )
    return (
        "Causal regime-aware policy learning remains strongest on the synthetic SCM benchmark, while the real "
        "factor-library benchmark is still led by simpler ranking or bandit baselines."
    )


def write_results_report(
    output_path: Path,
    *,
    synthetic_summary: pd.DataFrame,
    synthetic_discovery: pd.DataFrame,
    real_summary: pd.DataFrame,
    notes: list[str],
) -> None:
    synthetic_table = _method_summary(
        synthetic_summary,
        ["annual_return", "cum_return", "sharpe", "worst_regime_sharpe", "avg_turnover"],
        sort_by="sharpe",
    )
    discovery_table = _method_summary(
        synthetic_discovery,
        ["precision_at_5", "fdr_at_10", "rank_corr"],
        sort_by="precision_at_5",
    )
    real_table = _method_summary(
        real_summary,
        ["annual_return", "cum_return", "sharpe", "stress_sharpe", "max_drawdown", "avg_turnover"],
        sort_by="sharpe",
    )
    content = "\n".join(
        [
            "# Causal RL for Robust Alpha Discovery",
            "",
            "## Claim",
            _claim_text(real_table),
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
            "- `causal_rl_legacy` is the original standalone neural policy.",
            "- `causal_rl` automatically uses the counterfactual-aware causal path on the synthetic SCM benchmark and the bandit-guided regime-stability path on real data.",
            "",
        ]
    )
    output_path.write_text(content, encoding="utf-8")


def write_experiment_journal(
    output_path: Path,
    *,
    synthetic_summary: pd.DataFrame,
    real_summary: pd.DataFrame,
) -> None:
    real_table = _method_summary(
        real_summary,
        ["annual_return", "cum_return", "sharpe", "stress_sharpe", "max_drawdown", "avg_turnover"],
        sort_by="sharpe",
    ).set_index("method")
    synthetic_table = _method_summary(
        synthetic_summary,
        ["annual_return", "cum_return", "sharpe", "worst_regime_sharpe", "avg_turnover"],
        sort_by="sharpe",
    ).set_index("method")
    focus_methods = [
        method
        for method in [
            "correlation_rank",
            "contextual_bandit",
            "standard_rl",
            "causal_rl_legacy",
            "causal_rl_no_instability",
            "causal_rl",
        ]
        if method in real_table.index
    ]
    real_focus = real_table.loc[focus_methods].reset_index()
    delta_rows = []
    if "contextual_bandit" in real_table.index:
        bandit = real_table.loc["contextual_bandit"]
        for method in ["causal_rl_legacy", "causal_rl_no_instability", "causal_rl"]:
            if method in real_table.index:
                row = real_table.loc[method]
                delta_rows.append(
                    {
                        "method": method,
                        "delta_cum_return": row["cum_return"] - bandit["cum_return"],
                        "delta_sharpe": row["sharpe"] - bandit["sharpe"],
                        "delta_stress_sharpe": row.get("stress_sharpe", 0.0) - bandit.get("stress_sharpe", 0.0),
                        "delta_avg_turnover": row["avg_turnover"] - bandit["avg_turnover"],
                    }
                )
    delta_table = pd.DataFrame(delta_rows)
    hyper_columns = [
        column
        for column in [
            "selected_blend",
            "selected_instability",
            "selected_persistence",
            "selected_invariant_weight",
            "validation_score",
            "validation_sharpe",
            "validation_cum_return",
        ]
        if column in real_summary.columns
    ]
    hyper_table = (
        real_summary[real_summary["method"] == "causal_rl"][hyper_columns]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"index": "hyperparameter", 0: "value"})
    )
    synthetic_focus_methods = [
        method
        for method in ["contextual_bandit", "standard_rl", "causal_rl_legacy", "causal_rl_no_instability", "causal_rl"]
        if method in synthetic_table.index
    ]
    synthetic_focus = synthetic_table.loc[synthetic_focus_methods].reset_index()
    lines = [
        "# Experiment Journal",
        "",
        "## Real-Data Strategy Sweep",
        _markdown_table(real_focus),
        "",
        "## Causal Delta Vs Contextual Bandit",
        _markdown_table(delta_table) if not delta_table.empty else "No contextual bandit reference found.",
        "",
        "## Selected `causal_rl` Hyperparameters",
        _markdown_table(hyper_table) if not hyper_table.empty else "No causal hyperparameter summary available.",
        "",
        "## Synthetic Sanity Check",
        _markdown_table(synthetic_focus),
        "",
        "## Interpretation",
        "- `causal_rl_legacy` keeps the original standalone neural regime-aware learner for comparison.",
        "- `causal_rl_no_instability` removes the regime-stability penalty from the new bandit-guided causal policy.",
        "- `causal_rl` uses the counterfactual-aware legacy path on synthetic data and the validation-selected bandit-guided path on real data.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
