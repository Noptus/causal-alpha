import numpy as np

from causal_alpha_rl.algorithms.causal_rl.policy import CausalPolicyNetwork


def test_bandit_guided_causal_policy_fits_and_respects_constraints() -> None:
    rng = np.random.default_rng(7)
    n_factors = 4
    train_features = rng.normal(size=(8, n_factors, 3))
    val_features = rng.normal(size=(4, n_factors, 3))
    train_returns = rng.normal(scale=0.02, size=(8, n_factors))
    val_returns = rng.normal(scale=0.02, size=(4, n_factors))
    train_available = np.ones((8, n_factors), dtype=bool)
    val_available = np.ones((4, n_factors), dtype=bool)
    train_posteriors = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    val_posteriors = np.array(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )
    train_labels = train_posteriors.argmax(axis=1)
    val_labels = val_posteriors.argmax(axis=1)

    policy = CausalPolicyNetwork(
        top_k=2,
        max_weight=0.6,
        transaction_cost=0.001,
        random_state=7,
        factor_eye=np.eye(n_factors),
        blend_grid=(0.0, 0.5),
        instability_grid=(0.0, 0.2),
        persistence_grid=(0.0, 0.1),
    )
    policy.fit(
        train_features=train_features,
        train_returns=train_returns,
        train_available=train_available,
        val_features=val_features,
        val_returns=val_returns,
        val_available=val_available,
        train_regime_posteriors=train_posteriors,
        val_regime_posteriors=val_posteriors,
        train_regime_labels=train_labels,
        val_regime_labels=val_labels,
    )

    weights = policy.predict(
        val_features[0],
        val_available[0],
        np.zeros(n_factors, dtype=float),
        regime_posterior=val_posteriors[0],
    )

    assert np.isclose(weights.sum(), 1.0)
    assert np.count_nonzero(weights) <= 2
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 0.6 + 1e-8)
    assert "selected_blend" in policy.fit_summary_
    assert "validation_score" in policy.fit_summary_
