import numpy as np

from causal_alpha_rl.envs.allocation import weights_from_scores


def test_weight_projection_respects_constraints() -> None:
    scores = np.array([4.0, 3.0, 2.0, 1.0, 0.0, -1.0])
    available = np.array([1, 1, 1, 1, 1, 0], dtype=bool)
    weights = weights_from_scores(scores, available, top_k=5, max_weight=0.2)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 0.2 + 1e-8)
    assert np.count_nonzero(weights) <= 5
    assert weights[-1] == 0.0

