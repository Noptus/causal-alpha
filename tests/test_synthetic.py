import numpy as np

from causal_alpha_rl.evaluation.runner import build_sequence_dataset
from causal_alpha_rl.scm.synthetic import generate_synthetic_panel


def test_synthetic_generator_shapes_and_metadata() -> None:
    bundle = generate_synthetic_panel(7)
    dataset = build_sequence_dataset(bundle.panel)
    assert dataset.features.shape[1] == 40
    assert dataset.counterfactual_returns is not None
    assert dataset.counterfactual_returns.shape[1] == 3
    assert dataset.factor_metadata["true_causal"].sum() == 6
    assert np.isfinite(dataset.future_returns).all()


def test_synthetic_generation_is_deterministic() -> None:
    bundle_a = generate_synthetic_panel(11)
    bundle_b = generate_synthetic_panel(11)
    assert bundle_a.panel.equals(bundle_b.panel)
