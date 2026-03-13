"""CLI entrypoint for experiment execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_alpha_rl.config import load_config
from causal_alpha_rl.utils.logging import get_logger, init_run_context, log_event
from causal_alpha_rl.utils.paths import ensure_directories, project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run causal alpha experiments")
    parser.add_argument("--config", required=True, help="Name of the JSON config in experiments/configs")
    parser.add_argument("--stage", default="all", choices=["all", "data", "train", "report"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = project_paths()
    ensure_directories(paths)
    config = load_config(args.config)
    run_context = init_run_context(paths.artifacts, config.name)
    logger = get_logger(run_context)
    logger.info("Starting run %s for config %s", run_context.run_id, config.name)
    log_event(run_context, "run_started", config=config.payload, stage=args.stage)

    manifest_path = run_context.run_dir / "config_snapshot.json"
    manifest_path.write_text(json.dumps(config.payload, indent=2, sort_keys=True), encoding="utf-8")

    from causal_alpha_rl.experiments.pipeline import run_pipeline

    outputs = run_pipeline(config=config.payload, run_context=run_context, stage=args.stage)
    summary_path = run_context.run_dir / "summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Completed run %s", run_context.run_id)
    log_event(run_context, "run_completed", outputs=outputs)


if __name__ == "__main__":
    main()

