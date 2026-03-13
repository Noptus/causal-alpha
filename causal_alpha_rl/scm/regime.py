"""Latent regime estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class RegimeFit:
    model: GaussianHMM
    scaler: StandardScaler
    feature_columns: list[str]


def fit_regime_model(
    date_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    n_regimes: int,
    seed: int,
) -> RegimeFit:
    valid = date_frame.dropna(subset=feature_columns).copy()
    scaler = StandardScaler()
    matrix = scaler.fit_transform(valid[feature_columns])
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=200,
        random_state=seed,
    )
    model.fit(matrix)
    return RegimeFit(model=model, scaler=scaler, feature_columns=feature_columns)


def causal_regime_posteriors(fit: RegimeFit, date_frame: pd.DataFrame) -> pd.DataFrame:
    valid = date_frame.dropna(subset=fit.feature_columns).copy()
    matrix = fit.scaler.transform(valid[fit.feature_columns])
    posteriors = np.zeros((len(valid), fit.model.n_components), dtype=float)
    for idx in range(len(valid)):
        _, smoothed = fit.model.score_samples(matrix[: idx + 1])
        posteriors[idx] = smoothed[-1]
    posterior_frame = pd.DataFrame(
        posteriors,
        columns=[f"regime_p_{i}" for i in range(fit.model.n_components)],
    )
    posterior_frame["date"] = valid["date"].to_numpy()
    posterior_frame["regime_state"] = posterior_frame[
        [f"regime_p_{i}" for i in range(fit.model.n_components)]
    ].to_numpy().argmax(axis=1)
    return date_frame[["date"]].merge(posterior_frame, on="date", how="left")

