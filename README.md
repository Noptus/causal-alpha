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

## Current Findings

- Synthetic SCM: causal RL nearly matches the oracle Sharpe, reaches `precision@5 = 1.0` for causal factor discovery, and materially outperforms the frozen correlation ranking baseline under regime shift.
- Real factor library: the causal policy improves stress-regime Sharpe relative to the standard sequential policy learner, but the simple frozen correlation ranking baseline remains strongest overall.

## Data Sources

- OpenAssetPricing monthly long-short predictor returns
- FRED macro series: `CPIAUCSL`, `UNRATE`, `FEDFUNDS`, `GS10`, `TB3MS`, `BAA`
- Kenneth French research factors
