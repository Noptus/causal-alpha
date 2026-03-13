You are a senior research engineer + applied scientist building a NeurIPS-grade finance ML paper prototype.
Goal: Frame alpha discovery (factor selection/weighting) as an OFFLINE CAUSAL RL problem under confounding (latent regimes, macro shocks),
and implement counterfactual policy optimization that prefers factors with causal (regime-robust) effects over spurious correlations.

You will produce:
(1) a clean, reproducible Python research repo,
(2) datasets + environment code + baselines,
(3) at least one novel causal-offline-RL method that uses an SCM for counterfactual augmentation and/or IV-style deconfounding,
(4) experiments + plots + tables that support a NeurIPS paper claim about robustness under regime shift,
(5) a short paper-style report (markdown is fine) describing methods and results.

Hard constraints:
- No future-looking “TODO later.” Implement runnable code now.
- Reproducibility: fixed seeds, deterministic configs, one-command experiment runs.
- No data leakage: strict time-based splits and walk-forward evaluation.
- Realistic trading constraints: turnover penalty and transaction costs; no lookahead.
- Focus on “robust alpha discovery,” NOT “maximize backtest Sharpe by any means.”

Repo layout (create exactly these top-level folders):
- causal_alpha_rl/
  - data/                (download + preprocess scripts; cached data files with checksums)
  - envs/                (Gym-style environment for factor allocation/selection)
  - scm/                 (structural causal model definition + learning + counterfactual sampling)
  - algorithms/
      - baselines/       (non-causal offline RL, non-RL factor selection baselines)
      - causal_rl/       (your method(s))
  - evaluation/          (metrics, regime splits, robustness tests, OPE estimators)
  - experiments/         (hydra or json configs; scripts to run sweeps)
  - notebooks/           (minimal; keep core logic in modules)
  - figures/             (auto-generated output)
  - paper/               (paper-style markdown with figs + tables)
  - tests/               (unit tests for env transitions, SCM sampling, leakage checks)
  - README.md

Data requirement (implement open-by-default; optionally support WRDS if user has credentials):
A. OpenAssetPricing:
- Download monthly long-short returns for ~200+ predictors (factor/anomaly strategies).
- Use their “monthly long-short returns of predictors” dataset as the candidate factor library.
B. Macro confounders from FRED:
- Download a small set of monthly macro series (e.g., CPI inflation, unemployment, Fed funds, term spread).
- Align dates; forward-fill carefully and document.
C. Optional: Fama-French factors as additional benchmarks.

Environment design (factor policy MDP):
- Time index t is monthly.
- State s_t includes:
  - macro vector M_t (standardized; lagged to avoid leakage),
  - recent factor returns summary (e.g., last 12m returns per factor via low-rank embedding, NOT raw full vector),
  - previous portfolio weights w_{t-1},
  - (optional) a learned regime posterior q(z_t | history) from an HMM or VAE.
- Action a_t is a K-dimensional weight vector over K candidate factors:
  - Use a subset K=50–200 depending on compute.
  - Enforce constraints: simplex weights (sum=1), max weight per factor, turnover limit.
- Reward r_{t+1}:
  - portfolio factor return next month: dot(w_t, factor_returns_{t+1})
  - minus transaction cost: tc * ||w_t - w_{t-1}||_1
  - minus risk penalty: lambda * (rolling volatility or drawdown proxy)
- Transition:
  - next state uses t+1 macro and updated history; market evolution is exogenous to action.
- Hidden confounder/regime:
  - For synthetic experiments, generate a latent regime z_t that affects both macro and factor return distributions.
  - For real data, estimate regimes (HMM on volatility/macro) and treat them as partially observed confounders.

SCM component (this is core):
- Define an SCM where:
  - z_t -> macro M_t
  - z_t -> factor returns R^k_{t+1} (factor performance depends on regime)
  - (optionally) z_t -> behavior policy a_t in the logged dataset (to create confounded offline data)
- Implement TWO settings:
  1) Synthetic SCM with known ground truth (so we can measure causal identification quality).
  2) Learned SCM on real data:
     - Use a flexible conditional model p(R_{t+1} | M_t, z_t, w_t) or p(factor_returns_{t+1} | M_t, z_t)
     - Use latent z_t (HMM) or amortized inference network.

Offline dataset construction:
- Create logged trajectories (s_t, a_t, r_{t+1}, s_{t+1}) from a behavior policy pi_b:
  - pi_b sees (M_t + latent z_t) and chooses weights; BUT the learner does NOT observe z_t (confounding).
  - Ensure enough action diversity (e.g., Dirichlet noise) to make offline learning feasible.
- Store dataset as parquet/csv with schema validation.

Algorithms to implement (baselines):
1) Non-RL baseline:
   - Rolling mean-variance / risk parity over factor return series.
   - LASSO-style factor selection (predict next-month returns of factors with macro controls).
2) Offline RL baseline:
   - Conservative offline RL / behavior-cloning-style policy (simple versions OK if full CQL is heavy).
   - A contextual bandit baseline (one-step) to show why sequential matters with costs.
3) Causal baseline (not RL):
   - Regime-conditioned factor selection (oracle with observed regimes in synthetic setting).
   - Causal discovery / causal network factor grouping is optional if feasible.

Novel causal RL method (must implement at least one):
Option A: Counterfactual-augmented offline RL
- Learn the SCM; generate counterfactual factor-return outcomes for alternative weight actions w'_t.
- Augment dataset with (s_t, w'_t, r'_{t+1}, s_{t+1}) where r' is counterfactual reward.
- Train an offline RL agent on factual + counterfactual data, with conservative regularization to avoid exploiting model errors.
- Provide ablations:
  - without SCM (random augmentation),
  - without confounding,
  - varying SCM misspecification.

Option B: IV-style deconfounded value learning
- Implement an instrumental-variable value iteration style estimator on the synthetic confounded dataset:
  - Use lagged macro/history features as candidate instruments (validated in synthetic setting).
  - Show that naive OPE is biased under confounding and IV reduces bias.
- Combine with policy optimization (even simple policy gradient over a parametrized weight simplex).

Evaluation checklist (must be automated):
- Standard performance: CAGR, Sharpe, Sortino, max drawdown, turnover, average leverage, tx-cost-adjusted return.
- Robustness:
  - Train on regime set A, test on regime set B (held-out regimes in synthetic setting).
  - Rolling walk-forward: train N years, test next 1 year; slide window.
  - Stress test: high-vol periods / inflation regimes.
- Causal quality metrics (synthetic ground truth):
  - ability to recover regime-invariant “true causal factors” vs spurious ones,
  - policy value gap to oracle that knows z_t.
- Overfitting controls:
  - multiple random seeds,
  - report sensitivity to hyperparameters,
  - report “number of tried configs” and avoid cherry-picking.

Outputs:
- A single command `python -m experiments.run --config <name>` that:
  - downloads data (if missing),
  - builds datasets,
  - trains models,
  - writes metrics to `figures/` and a results table to `paper/results.md`.
- Include plots:
  - cumulative returns across train/test regimes,
  - performance distribution across seeds,
  - factor weight stability and turnover,
  - ablation bars for causal components.

Testing:
- Unit tests for:
  - no lookahead in feature construction,
  - action constraints (simplex, max weight),
  - SCM sampling shape/NaN checks,
  - deterministic run output hash given fixed seed.

Write the code with clear docstrings and type hints.
Prefer small, readable implementations over heavy frameworks.
If something is computationally expensive (full deep offline RL), implement a lighter but conceptually faithful version and document tradeoffs.