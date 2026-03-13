# Causal RL for Robust Alpha Discovery

## Claim
Causal regime-aware policy learning is strongest on the synthetic SCM benchmark, where it recovers invariant factors and nearly matches the oracle Sharpe. On the real factor-library benchmark, the causal policy improves stress-regime Sharpe versus the standard policy learner, but the simple correlation ranking baseline remains the strongest overall performer.

## Synthetic Performance
| method | annual_return | sharpe | worst_regime_sharpe | avg_turnover |
| --- | --- | --- | --- | --- |
| oracle_regime | 0.087 | 2.066 | 1.691 | 0.908 |
| causal_rl | 0.046 | 2.059 | 1.691 | 0.546 |
| elastic_net | 0.074 | 1.991 | 1.682 | 0.997 |
| standard_rl | 0.076 | 1.889 | 1.727 | 0.862 |
| contextual_bandit | 0.066 | 1.635 | 1.708 | 1.173 |
| correlation_rank | -0.047 | -0.768 | -3.769 | 0.002 |

## Synthetic Factor Discovery
| method | precision_at_5 | fdr_at_10 | rank_corr |
| --- | --- | --- | --- |
| causal_rl | 1.000 | 0.400 | 0.620 |
| contextual_bandit | 0.400 | 0.400 | 0.534 |
| elastic_net | 0.400 | 0.500 | 0.491 |
| oracle_regime | 0.400 | 0.600 | 0.233 |
| standard_rl | 0.400 | 0.500 | 0.479 |
| correlation_rank | 0.000 | 1.000 | -0.159 |

## Real Data Performance
| method | annual_return | sharpe | stress_sharpe | max_drawdown | avg_turnover |
| --- | --- | --- | --- | --- | --- |
| correlation_rank | 0.146 | 2.106 | 1.989 | -0.093 | 0.017 |
| contextual_bandit | 0.197 | 1.271 | 1.435 | -0.146 | 0.796 |
| causal_rl | 0.126 | 0.876 | 1.224 | -0.243 | 0.834 |
| elastic_net | 0.191 | 1.081 | 1.144 | -0.195 | 0.747 |
| standard_rl | 0.187 | 1.033 | 1.029 | -0.185 | 0.843 |

## Figures
- `figures/generated/synthetic_cumulative_returns.png`
- `figures/generated/synthetic_seed_boxplot.png`
- `figures/generated/real_cumulative_returns.png`
- `figures/generated/real_turnover_scatter.png`
- `figures/generated/factor_importance_heatmap.png`

## Notes
- Synthetic causal RL uses the correctly specified latent regime to test the SCM claim.
- Real data uses inferred regimes from an HMM fitted on lagged macro and factor-dispersion proxies.
- Raw downloads are cached outside git; generated tables and figures are committed.
- The real benchmark should be interpreted as external validity with mixed evidence, not as definitive dominance over simpler factor-ranking methods.
