"""Time-based evaluation splits."""

from __future__ import annotations

import pandas as pd


def walk_forward_splits(
    dates: list[pd.Timestamp],
    *,
    train_months: int = 180,
    val_months: int = 60,
    test_months: int = 60,
    step_months: int = 60,
) -> list[dict[str, tuple[int, int]]]:
    splits: list[dict[str, tuple[int, int]]] = []
    start = 0
    n_dates = len(dates)
    while start + train_months + val_months + test_months <= n_dates:
        splits.append(
            {
                "train": (start, start + train_months),
                "val": (start + train_months, start + train_months + val_months),
                "test": (
                    start + train_months + val_months,
                    start + train_months + val_months + test_months,
                ),
            }
        )
        start += step_months
    return splits

