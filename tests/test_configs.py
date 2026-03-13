from causal_alpha_rl.config import load_config


def test_load_config() -> None:
    config = load_config("synthetic_main")
    assert config["name"] == "synthetic_main"
