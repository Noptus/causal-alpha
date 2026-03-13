"""Synthetic structural causal model for regime-confounded factor returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_alpha_rl.data.panel import build_factor_panel_from_returns


MACRO_COLUMNS = [
    "inflation_yoy",
    "unemployment",
    "fed_funds",
    "term_spread",
    "credit_spread",
]


@dataclass(slots=True)
class SyntheticBundle:
    panel: pd.DataFrame
    factor_metadata: pd.DataFrame
    split: dict[str, tuple[int, int]]
    transition_shift_idx: int


def _sample_regimes(
    rng: np.random.Generator,
    n_months: int,
    transition_a: np.ndarray,
    transition_b: np.ndarray,
    shift_idx: int,
) -> np.ndarray:
    regimes = np.zeros(n_months, dtype=int)
    regimes[0] = 0
    for t in range(1, n_months):
        transition = transition_a if t < shift_idx else transition_b
        regimes[t] = rng.choice(transition.shape[0], p=transition[regimes[t - 1]])
    return regimes


def generate_synthetic_panel(
    seed: int,
    *,
    n_months: int = 2400,
    n_factors: int = 40,
    n_regimes: int = 3,
    shift_idx: int = 1800,
) -> SyntheticBundle:
    rng = np.random.default_rng(seed)
    factor_ids = [f"F{i:02d}" for i in range(n_factors)]
    groups = ["causal"] * 6 + ["conditional"] * 10 + ["spurious"] * (n_factors - 16)
    group_array = np.array(groups)

    transition_train = np.array(
        [
            [0.90, 0.08, 0.02],
            [0.08, 0.88, 0.04],
            [0.18, 0.12, 0.70],
        ]
    )
    transition_test = np.array(
        [
            [0.72, 0.10, 0.18],
            [0.12, 0.70, 0.18],
            [0.12, 0.08, 0.80],
        ]
    )
    regimes = _sample_regimes(rng, n_months, transition_train, transition_test, shift_idx)

    regime_centers = np.array(
        [
            [0.01, 0.04, 0.02, 0.015, 0.010],
            [0.03, 0.05, 0.03, 0.008, 0.015],
            [0.07, 0.09, 0.06, -0.010, 0.040],
        ]
    )
    macro_noise = rng.normal(scale=0.015, size=(n_months, len(MACRO_COLUMNS)))
    market_noise = rng.normal(scale=0.8, size=n_months)
    factor_noise = rng.normal(scale=0.55, size=(n_months, n_factors))

    macro = np.zeros((n_months, len(MACRO_COLUMNS)))
    macro_cf = np.zeros((n_months, n_regimes, len(MACRO_COLUMNS)))
    for t in range(n_months):
        prev_macro = macro[t - 1] if t > 0 else np.zeros(len(MACRO_COLUMNS))
        for regime in range(n_regimes):
            macro_cf[t, regime] = 0.55 * prev_macro + regime_centers[regime] + macro_noise[t]
        macro[t] = macro_cf[t, regimes[t]]

    market_cf = 1.5 * macro_cf[:, :, 0] - 0.9 * macro_cf[:, :, 1] + 1.2 * macro_cf[:, :, 3] + market_noise[:, None]

    preferred_regime = rng.integers(0, n_regimes, size=n_factors)
    causal_strength = np.zeros(n_factors)
    regime_loadings = np.zeros((n_factors, n_regimes))
    market_loading = rng.uniform(0.2, 0.8, size=n_factors)

    for idx, group in enumerate(groups):
        if group == "causal":
            base = rng.uniform(0.18, 0.34)
            causal_strength[idx] = base
            regime_loadings[idx] = base + rng.normal(0.0, 0.05, size=n_regimes)
        elif group == "conditional":
            hot = preferred_regime[idx]
            regime_loadings[idx] = -0.70
            regime_loadings[idx, hot] = 1.25
            causal_strength[idx] = 0.05
        else:
            hot = preferred_regime[idx] if preferred_regime[idx] != 2 else 0
            regime_loadings[idx] = np.array([1.60 if hot == 0 else 1.10, 1.60 if hot == 1 else 1.10, -2.10])
            causal_strength[idx] = -0.25

    expected_cf = np.zeros((n_months, n_regimes, n_factors))
    for regime in range(n_regimes):
        expected_cf[:, regime, :] = regime_loadings[:, regime] + market_cf[:, regime, None] * market_loading
    expected_cf = expected_cf / 100.0
    realized_returns = expected_cf[np.arange(n_months), regimes] + (factor_noise / 100.0)

    dates = pd.date_range("1900-01-31", periods=n_months, freq="ME")
    returns = pd.DataFrame(realized_returns, index=dates, columns=factor_ids)
    macro_frame = pd.DataFrame(macro, columns=MACRO_COLUMNS)
    macro_frame["date"] = dates
    macro_frame[MACRO_COLUMNS] = macro_frame[MACRO_COLUMNS].shift(1)

    extra_date = pd.DataFrame(
        {
            "date": dates,
            "true_regime": regimes,
            "stress_regime": (regimes == 2).astype(int),
        }
    )
    factor_metadata = pd.DataFrame(
        {
            "factor_id": factor_ids,
            "factor_group": group_array,
            "preferred_regime": preferred_regime,
            "true_causal": (group_array == "causal").astype(int),
            "true_causal_score": causal_strength,
        }
    )
    panel = build_factor_panel_from_returns(
        returns,
        macro_frame,
        extra_date_features=extra_date,
        factor_metadata=factor_metadata,
    )
    for regime in range(n_regimes):
        cf_frame = pd.DataFrame(expected_cf[:, regime, :], index=dates, columns=factor_ids)
        long_cf = cf_frame.stack(future_stack=True).rename(f"cf_ret_regime_{regime}").reset_index()
        long_cf.columns = ["date", "factor_id", f"cf_ret_regime_{regime}"]
        panel = panel.merge(long_cf, on=["date", "factor_id"], how="left")

    split = {
        "train": (0, 1500),
        "val": (1500, shift_idx),
        "test": (shift_idx, n_months - 1),
    }
    return SyntheticBundle(panel=panel, factor_metadata=factor_metadata, split=split, transition_shift_idx=shift_idx)
