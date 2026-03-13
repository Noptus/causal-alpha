# Experiment Journal

## Real-Data Strategy Sweep
| method | annual_return | cum_return | sharpe | stress_sharpe | max_drawdown | avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| correlation_rank | 0.146 | 1.022 | 2.106 | 1.989 | -0.093 | 0.017 |
| contextual_bandit | 0.197 | 1.577 | 1.271 | 1.435 | -0.146 | 0.796 |
| standard_rl | 0.187 | 1.326 | 1.033 | 1.029 | -0.185 | 0.843 |
| causal_rl_legacy | 0.213 | 1.721 | 1.178 | 1.288 | -0.183 | 0.867 |
| causal_rl_no_instability | 0.190 | 1.462 | 1.222 | 1.381 | -0.163 | 0.617 |
| causal_rl | 0.201 | 1.585 | 1.276 | 1.256 | -0.186 | 0.597 |

## Causal Delta Vs Contextual Bandit
| method | delta_cum_return | delta_sharpe | delta_stress_sharpe | delta_avg_turnover |
| --- | --- | --- | --- | --- |
| causal_rl_legacy | 0.143 | -0.093 | -0.147 | 0.071 |
| causal_rl_no_instability | -0.115 | -0.048 | -0.054 | -0.179 |
| causal_rl | 0.008 | 0.005 | -0.179 | -0.199 |

## Selected `causal_rl` Hyperparameters
| hyperparameter | value |
| --- | --- |
| selected_blend | 0.375 |
| selected_instability | 0.275 |
| selected_persistence | 0.087 |
| validation_score | 3.405 |
| validation_sharpe | 2.574 |
| validation_cum_return | 2.776 |

## Synthetic Sanity Check
| method | annual_return | cum_return | sharpe | worst_regime_sharpe | avg_turnover |
| --- | --- | --- | --- | --- | --- |
| contextual_bandit | 0.066 | 21.973 | 1.635 | 1.708 | 1.173 |
| standard_rl | 0.076 | 36.770 | 1.889 | 1.727 | 0.862 |
| causal_rl_legacy | 0.046 | 8.863 | 2.059 | 1.691 | 0.546 |
| causal_rl_no_instability | 0.055 | 13.117 | 2.083 | 1.698 | 0.784 |
| causal_rl | 0.055 | 13.117 | 2.083 | 1.698 | 0.784 |

## Interpretation
- `causal_rl_legacy` keeps the original standalone neural regime-aware learner for comparison.
- `causal_rl_no_instability` removes the regime-stability penalty from the new bandit-guided causal policy.
- `causal_rl` uses the counterfactual-aware legacy path on synthetic data and the validation-selected bandit-guided path on real data.
