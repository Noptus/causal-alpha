import pandas as pd

from causal_alpha_rl.data.panel import build_factor_panel_from_returns


def test_panel_builder_uses_forward_returns() -> None:
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    returns = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [0.5, 0.4, 0.3, 0.2]}, index=dates)
    macro = pd.DataFrame(
        {
            "date": dates,
            "inflation_yoy": [0.01, 0.02, 0.03, 0.04],
            "unemployment": [5.0, 5.1, 5.2, 5.3],
            "fed_funds": [1.0, 1.1, 1.2, 1.3],
            "term_spread": [0.5, 0.4, 0.3, 0.2],
            "credit_spread": [1.5, 1.6, 1.7, 1.8],
        }
    )
    panel = build_factor_panel_from_returns(returns, macro)
    first_row = panel[(panel["date"] == dates[0]) & (panel["factor_id"] == "A")].iloc[0]
    assert first_row["ret_1m"] == 1.0
    assert first_row["ret_fwd_1m"] == 2.0
    last_row = panel[(panel["date"] == dates[-1]) & (panel["factor_id"] == "A")].iloc[0]
    assert last_row["ret_fwd_1m"] == 0.0 or pd.isna(last_row["ret_fwd_1m"])

