"""Filesystem paths used across the project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    root: Path
    artifacts: Path
    data_raw: Path
    data_processed: Path
    figures: Path
    paper: Path


def project_paths(root: Path | None = None) -> ProjectPaths:
    repo_root = (root or Path(__file__).resolve().parents[2]).resolve()
    return ProjectPaths(
        root=repo_root,
        artifacts=repo_root / "artifacts",
        data_raw=repo_root / "data" / "raw",
        data_processed=repo_root / "data" / "processed",
        figures=repo_root / "figures",
        paper=repo_root / "paper",
    )


def ensure_directories(paths: ProjectPaths) -> None:
    for directory in (
        paths.artifacts,
        paths.data_raw,
        paths.data_processed,
        paths.figures,
        paths.paper,
    ):
        directory.mkdir(parents=True, exist_ok=True)

