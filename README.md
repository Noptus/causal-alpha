# causal-alpha

Reproducible experiments for causal regime-aware alpha discovery under regime shifts.

The repo contains:

- open-data fetchers for OpenAssetPricing, FRED, and Kenneth French factors
- a synthetic SCM benchmark with latent regimes and counterfactual factor returns
- correlation, contextual bandit, standard sequential policy, and causal regime-aware policy baselines
- walk-forward evaluation, plots, and a paper-style Markdown report

## Environment

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

## Main Commands

Synthetic benchmark only:

```bash
.venv/bin/python -m experiments.run --config synthetic_main
```

Real-data benchmark only:

```bash
.venv/bin/python -m experiments.run --config real_main
```

Full paper bundle:

```bash
.venv/bin/python -m experiments.run --config paper_bundle
```

## Outputs

- raw source manifest: `data/source_manifest.json`
- run logs and snapshots: `artifacts/runs/<run_id>/`
- generated figures: `figures/generated/`
- tables: `paper/generated_synthetic_summary.csv`, `paper/generated_synthetic_discovery.csv`, `paper/generated_real_summary.csv`
- paper-style writeup: `paper/results.md`
- experiment journal: `paper/experiment_journal.md`

## Current Findings

- Synthetic SCM: `causal_rl` reaches `precision@5 = 1.0`, posts `Sharpe = 2.083`, and beats the oracle on worst-regime Sharpe while the frozen correlation ranking baseline collapses under regime shift.
- Real factor library: `causal_rl` now edges the contextual bandit on both cumulative return (`1.585` vs `1.577`) and Sharpe (`1.276` vs `1.271`) with materially lower turnover (`0.597` vs `0.796`), while the frozen correlation ranking baseline remains the highest-Sharpe reference overall.

## Data Sources

- OpenAssetPricing monthly long-short predictor returns
- FRED macro series: `CPIAUCSL`, `UNRATE`, `FEDFUNDS`, `GS10`, `TB3MS`, `BAA`
- Kenneth French research factors
