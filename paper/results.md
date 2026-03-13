# Causal RL for Robust Alpha Discovery

## Claim
Bandit-guided causal RL now improves both cumulative return and Sharpe versus the contextual bandit on the real factor-library benchmark, while the frozen correlation ranking baseline remains the highest-Sharpe reference.

## Synthetic Performance
| method | annual_return | cum_return | sharpe | worst_regime_sharpe | avg_turnover |
| --- | --- | --- | --- | --- | --- |
| causal_rl | 0.055 | 13.117 | 2.083 | 1.698 | 0.784 |
| causal_rl_no_instability | 0.055 | 13.117 | 2.083 | 1.698 | 0.784 |
| oracle_regime | 0.087 | 59.837 | 2.066 | 1.691 | 0.908 |
| causal_rl_legacy | 0.046 | 8.863 | 2.059 | 1.691 | 0.546 |
| elastic_net | 0.074 | 33.241 | 1.991 | 1.682 | 0.997 |
| standard_rl | 0.076 | 36.770 | 1.889 | 1.727 | 0.862 |
| contextual_bandit | 0.066 | 21.973 | 1.635 | 1.708 | 1.173 |
| correlation_rank | -0.047 | -0.917 | -0.768 | -3.769 | 0.002 |

## Synthetic Factor Discovery
| method | precision_at_5 | fdr_at_10 | rank_corr |
| --- | --- | --- | --- |
| causal_rl | 1.000 | 0.400 | 0.602 |
| causal_rl_legacy | 1.000 | 0.400 | 0.620 |
| causal_rl_no_instability | 1.000 | 0.400 | 0.602 |
| contextual_bandit | 0.400 | 0.400 | 0.534 |
| elastic_net | 0.400 | 0.500 | 0.491 |
| oracle_regime | 0.400 | 0.600 | 0.233 |
| standard_rl | 0.400 | 0.500 | 0.479 |
| correlation_rank | 0.000 | 1.000 | -0.159 |

## Real Data Performance
| method | annual_return | cum_return | sharpe | stress_sharpe | max_drawdown | avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| correlation_rank | 0.146 | 1.022 | 2.106 | 1.989 | -0.093 | 0.017 |
| causal_rl | 0.201 | 1.585 | 1.276 | 1.256 | -0.186 | 0.597 |
| contextual_bandit | 0.197 | 1.577 | 1.271 | 1.435 | -0.146 | 0.796 |
| causal_rl_no_instability | 0.190 | 1.462 | 1.222 | 1.381 | -0.163 | 0.617 |
| causal_rl_legacy | 0.213 | 1.721 | 1.178 | 1.288 | -0.183 | 0.867 |
| elastic_net | 0.191 | 1.439 | 1.081 | 1.144 | -0.195 | 0.747 |
| standard_rl | 0.187 | 1.326 | 1.033 | 1.029 | -0.185 | 0.843 |

## Figures
- `figures/generated/synthetic_cumulative_returns.png`
- `figures/generated/synthetic_seed_boxplot.png`
- `figures/generated/real_cumulative_returns.png`
- `figures/generated/real_turnover_scatter.png`
- `figures/generated/factor_importance_heatmap.png`

## Notes
- Synthetic causal RL uses the correctly specified latent regime to test the SCM claim.
- Real data uses inferred regimes from an HMM fitted on lagged macro and factor-dispersion proxies.
- The main `causal_rl` method uses counterfactual-aware causal training on synthetic data and a bandit-guided selector on real data.
- Raw downloads are cached outside git; generated tables and figures are committed.
- `causal_rl_legacy` is the original standalone neural policy.
- `causal_rl` automatically uses the counterfactual-aware causal path on the synthetic SCM benchmark and the bandit-guided regime-stability path on real data.
