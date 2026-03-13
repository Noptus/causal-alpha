"""High-level experiment pipeline orchestration."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from causal_alpha_rl.data.fetch import fetch_all_data
from causal_alpha_rl.data.real import build_real_panel
from causal_alpha_rl.evaluation.plots import (
    plot_cumulative_returns,
    plot_importance_heatmap,
    plot_seed_boxplot,
    plot_turnover_scatter,
)
from causal_alpha_rl.evaluation.runner import aggregate_fold_outputs, build_sequence_dataset, run_fold
from causal_alpha_rl.evaluation.splits import walk_forward_splits
from causal_alpha_rl.paper.report import write_results_report
from causal_alpha_rl.scm.synthetic import generate_synthetic_panel
from causal_alpha_rl.utils.logging import RunContext
from causal_alpha_rl.utils.paths import project_paths
from causal_alpha_rl.utils.retry import retry_call


def run_pipeline(config: dict[str, Any], run_context: RunContext, stage: str) -> dict[str, Any]:
    paths = project_paths()
    outputs: dict[str, Any] = {"config_name": config["name"], "stage": stage, "run_id": run_context.run_id}
    manifest_path = paths.root / "data" / "source_manifest.json"

    if stage in {"all", "data", "train", "report"}:
        fetch_all_data(paths.data_raw, manifest_path)
        build_real_panel(paths.data_raw, paths.data_processed)
        outputs["manifest_path"] = str(manifest_path)

    if config["kind"] == "synthetic":
        synthetic_bundle = generate_synthetic_panel(config["seeds"][0])
        dataset = build_sequence_dataset(synthetic_bundle.panel)
        fold_outputs = [
            retry_call(
                lambda: run_fold(
                    dataset,
                    synthetic_bundle.split,
                    seeds=config["seeds"],
                    top_k=config.get("top_k", 5),
                    max_weight=config.get("max_weight", 0.2),
                    transaction_cost=config.get("transaction_cost", 0.001),
                    kind="synthetic",
                )
            )
        ]
        aggregated = aggregate_fold_outputs(fold_outputs)
        synthetic_dir = paths.figures / "generated"
        plot_cumulative_returns(fold_outputs[0]["return_series"], synthetic_dir / "synthetic_cumulative_returns.png", title="Synthetic Cumulative Returns")
        plot_seed_boxplot(aggregated["summary"], "sharpe", synthetic_dir / "synthetic_seed_boxplot.png", title="Synthetic Sharpe Across Seeds")
        plot_importance_heatmap(fold_outputs[0]["importance"], synthetic_dir / "factor_importance_heatmap.png", title="Average Factor Importances")
        aggregated["summary"].to_csv(paths.paper / "generated_synthetic_summary.csv", index=False)
        aggregated["discovery"].to_csv(paths.paper / "generated_synthetic_discovery.csv", index=False)
        outputs["synthetic_summary_path"] = str(paths.paper / "generated_synthetic_summary.csv")
        outputs["synthetic_discovery_path"] = str(paths.paper / "generated_synthetic_discovery.csv")
        outputs["summary_rows"] = int(len(aggregated["summary"]))
        return outputs

    if config["kind"] == "real":
        panel = pd.read_parquet(paths.data_processed / "real_factor_panel.parquet")
        valid_dates = (
            panel.groupby("date")
            .agg(
                n_available=("available", "sum"),
                inflation_yoy=("inflation_yoy", "first"),
                unemployment=("unemployment", "first"),
                fed_funds=("fed_funds", "first"),
                term_spread=("term_spread", "first"),
                credit_spread=("credit_spread", "first"),
                factor_dispersion_1m=("factor_dispersion_1m", "first"),
                factor_vol_12m=("factor_vol_12m", "first"),
            )
            .reset_index()
        )
        valid_dates = valid_dates[
            (valid_dates["n_available"] >= config.get("min_available", 120))
            & valid_dates[
                [
                    "inflation_yoy",
                    "unemployment",
                    "fed_funds",
                    "term_spread",
                    "credit_spread",
                    "factor_dispersion_1m",
                    "factor_vol_12m",
                ]
            ]
            .notna()
            .all(axis=1)
        ]["date"]
        panel = panel[panel["date"].isin(valid_dates)].copy()
        dataset = build_sequence_dataset(panel)
        splits = walk_forward_splits(
            dataset.dates,
            train_months=config.get("train_months", 180),
            val_months=config.get("val_months", 60),
            test_months=config.get("test_months", 60),
            step_months=config.get("step_months", 60),
        )
        fold_outputs = []
        for split in splits:
            fold_outputs.append(
                retry_call(
                    lambda split=split: run_fold(
                        dataset,
                        split,
                        seeds=config["seeds"],
                        top_k=config.get("top_k", 10),
                        max_weight=config.get("max_weight", 0.2),
                        transaction_cost=config.get("transaction_cost", 0.0015),
                        kind="real",
                    )
                )
            )
        aggregated = aggregate_fold_outputs(fold_outputs)
        return_frame = pd.concat(
            [
                fold_output["return_series"].add_prefix(f"fold{fold_id}_")
                for fold_id, fold_output in enumerate(fold_outputs)
            ],
            axis=1,
        )
        generated_dir = paths.figures / "generated"
        plot_cumulative_returns(return_frame, generated_dir / "real_cumulative_returns.png", title="Real Data Walk-Forward Cumulative Returns")
        plot_turnover_scatter(aggregated["summary"], generated_dir / "real_turnover_scatter.png", title="Sharpe vs Turnover on Real Data")
        aggregated["summary"].to_csv(paths.paper / "generated_real_summary.csv", index=False)
        outputs["real_summary_path"] = str(paths.paper / "generated_real_summary.csv")
        outputs["summary_rows"] = int(len(aggregated["summary"]))
        return outputs

    if config["kind"] == "paper_bundle":
        synthetic_outputs = run_pipeline({"name": "synthetic_main", "kind": "synthetic", **config["synthetic"]}, run_context, stage)
        real_outputs = run_pipeline({"name": "real_main", "kind": "real", **config["real"]}, run_context, stage)
        synthetic_summary = pd.read_csv(paths.paper / "generated_synthetic_summary.csv")
        synthetic_discovery = pd.read_csv(paths.paper / "generated_synthetic_discovery.csv")
        real_summary = pd.read_csv(paths.paper / "generated_real_summary.csv")
        write_results_report(
            paths.paper / "results.md",
            synthetic_summary=synthetic_summary,
            synthetic_discovery=synthetic_discovery,
            real_summary=real_summary,
            notes=[
                "Synthetic causal RL uses the correctly specified latent regime to test the SCM claim.",
                "Real data uses inferred regimes from an HMM fitted on lagged macro and factor-dispersion proxies.",
                "Raw downloads are cached outside git; generated tables and figures are committed.",
            ],
        )
        outputs["synthetic"] = synthetic_outputs
        outputs["real"] = real_outputs
        outputs["report_path"] = str(paths.paper / "results.md")
        return outputs

    raise ValueError(f"Unsupported config kind: {config['kind']}")
