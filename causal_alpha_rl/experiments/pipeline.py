"""High-level experiment pipeline orchestration."""

from __future__ import annotations

from typing import Any

from causal_alpha_rl.utils.logging import RunContext


def run_pipeline(config: dict[str, Any], run_context: RunContext, stage: str) -> dict[str, Any]:
    """Placeholder pipeline that is populated by later implementation steps."""
    return {
        "config_name": config["name"],
        "stage": stage,
        "run_id": run_context.run_id,
    }

