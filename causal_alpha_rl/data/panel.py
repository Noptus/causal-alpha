"""Shared factor panel construction."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def build_factor_panel_from_returns(
    returns: pd.DataFrame,
    macro_features: pd.DataFrame,
    *,
    extra_date_features: pd.DataFrame | None = None,
    factor_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    returns = returns.sort_index()
    returns.index.name = "date"
    trailing_1m = returns
    trailing_3m = returns.rolling(3, min_periods=1).mean()
    trailing_6m = returns.rolling(6, min_periods=1).mean()
    trailing_12m = returns.rolling(12, min_periods=3).mean()
    vol_12m = returns.rolling(12, min_periods=3).std()
    downside_12m = returns.clip(upper=0).rolling(12, min_periods=3).std()
    drawdown_12m = returns.rolling(12, min_periods=3).apply(
        lambda values: float((values.cumsum() - np.maximum.accumulate(values.cumsum())).min()),
        raw=False,
    )
    next_returns = returns.shift(-1)
    availability = returns.notna().astype(int)

    cross_section = pd.DataFrame(index=returns.index)
    cross_section["factor_mean_1m"] = returns.mean(axis=1)
    cross_section["factor_dispersion_1m"] = returns.std(axis=1)
    cross_section["factor_breadth_1m"] = (returns > 0).mean(axis=1)
    cross_section["factor_mean_12m"] = cross_section["factor_mean_1m"].rolling(12, min_periods=3).mean()
    cross_section["factor_vol_12m"] = cross_section["factor_mean_1m"].rolling(12, min_periods=3).std()

    long_frames: list[pd.DataFrame] = []
    feature_map = {
        "ret_1m": trailing_1m,
        "ret_3m": trailing_3m,
        "ret_6m": trailing_6m,
        "ret_12m": trailing_12m,
        "vol_12m": vol_12m,
        "downside_12m": downside_12m,
        "drawdown_12m": drawdown_12m,
        "ret_fwd_1m": next_returns,
        "available": availability,
    }
    for feature_name, feature_frame in feature_map.items():
        stacked = feature_frame.stack(future_stack=True).rename(feature_name).reset_index()
        stacked.columns = ["date", "factor_id", feature_name]
        long_frames.append(stacked)

    panel = long_frames[0]
    for frame in long_frames[1:]:
        panel = panel.merge(frame, on=["date", "factor_id"], how="left")

    macro = macro_features.copy()
    panel = panel.merge(macro, on="date", how="left")
    panel = panel.merge(cross_section.reset_index(), on="date", how="left")
    if extra_date_features is not None:
        panel = panel.merge(extra_date_features, on="date", how="left")
    if factor_metadata is not None:
        panel = panel.merge(factor_metadata, on="factor_id", how="left")
    panel = panel.sort_values(["date", "factor_id"]).reset_index(drop=True)
    panel["available"] = panel["available"].fillna(0).astype(int)
    return panel
