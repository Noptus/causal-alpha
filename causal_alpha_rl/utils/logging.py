"""Structured run logging."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass(slots=True)
class RunContext:
    run_id: str
    run_dir: Path
    log_path: Path


def init_run_context(artifacts_dir: Path, name: str) -> RunContext:
    run_id = f"{name}-{utc_now()}"
    run_dir = artifacts_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(run_id=run_id, run_dir=run_dir, log_path=run_dir / "events.jsonl")


def get_logger(run_context: RunContext) -> logging.Logger:
    logger = logging.getLogger(run_context.run_id)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(run_context.run_dir / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(stream)
    return logger


def log_event(run_context: RunContext, event: str, **payload: Any) -> None:
    record = {
        "ts": utc_now(),
        "event": event,
        **payload,
    }
    with run_context.log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

