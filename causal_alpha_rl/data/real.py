"""Real-data panel construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from causal_alpha_rl.data.fetch import load_fred_frame, load_open_asset_pricing, parse_french_factors
from causal_alpha_rl.data.panel import build_factor_panel_from_returns


def build_macro_features(raw_dir: Path) -> pd.DataFrame:
    macro = load_fred_frame(raw_dir)
    macro = macro.sort_values("date").reset_index(drop=True)
    macro["inflation_yoy"] = macro["CPIAUCSL"].pct_change(12)
    macro["unemployment"] = macro["UNRATE"]
    macro["fed_funds"] = macro["FEDFUNDS"]
    macro["term_spread"] = macro["GS10"] - macro["TB3MS"]
    macro["credit_spread"] = macro["BAA"] - macro["GS10"]
    features = macro[
        ["date", "inflation_yoy", "unemployment", "fed_funds", "term_spread", "credit_spread"]
    ].copy()
    lagged_columns = [column for column in features.columns if column != "date"]
    features[lagged_columns] = features[lagged_columns].shift(1)
    return features


def build_real_panel(raw_dir: Path, processed_dir: Path) -> dict[str, Path]:
    oap = load_open_asset_pricing(raw_dir / "open_asset_pricing_212_predictors.csv")
    factors = [column for column in oap.columns if column != "date"]
    returns = (oap.set_index("date")[factors].sort_index().copy()) / 100.0
    macro = build_macro_features(raw_dir)
    panel = build_factor_panel_from_returns(returns, macro)

    french = parse_french_factors(raw_dir / "ken_french" / "F-F_Research_Data_Factors_CSV.zip")
    processed_dir.mkdir(parents=True, exist_ok=True)

    oap_path = processed_dir / "oap_returns.parquet"
    macro_path = processed_dir / "macro_features.parquet"
    french_path = processed_dir / "french_factors.parquet"
    panel_path = processed_dir / "real_factor_panel.parquet"

    returns.reset_index().copy().to_parquet(oap_path, index=False)
    macro.to_parquet(macro_path, index=False)
    french.to_parquet(french_path, index=False)
    panel.to_parquet(panel_path, index=False)

    return {
        "oap_returns": oap_path,
        "macro_features": macro_path,
        "french_factors": french_path,
        "real_factor_panel": panel_path,
    }
