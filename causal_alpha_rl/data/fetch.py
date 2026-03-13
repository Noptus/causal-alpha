"""Data fetching for open finance sources."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from causal_alpha_rl.data.manifest import load_manifest, sha256sum, write_manifest
from causal_alpha_rl.utils.retry import retry_call


OPEN_ASSET_PRICING_FILE_ID = "10sOryk_ddjkXagaajTKUk1nwJs2ZLRiI"
OPEN_ASSET_PRICING_URL = f"https://drive.google.com/uc?export=download&id={OPEN_ASSET_PRICING_FILE_ID}"
FRENCH_FACTORS_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
)
FRED_SERIES = {
    "CPIAUCSL": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL",
    "UNRATE": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE",
    "FEDFUNDS": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS",
    "GS10": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GS10",
    "TB3MS": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS",
    "BAA": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA",
}


@dataclass(slots=True)
class DownloadResult:
    path: Path
    source: str
    sha256: str


def _download_bytes(url: str, timeout: int = 60) -> bytes:
    def _impl() -> bytes:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response.content

    return retry_call(_impl, retries=3, transient_exceptions=(requests.RequestException,))


def download_file(url: str, destination: Path) -> DownloadResult:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(_download_bytes(url))
    return DownloadResult(path=destination, source=url, sha256=sha256sum(destination))


def fetch_open_asset_pricing(raw_dir: Path) -> DownloadResult:
    return download_file(OPEN_ASSET_PRICING_URL, raw_dir / "open_asset_pricing_212_predictors.csv")


def fetch_fred_series(raw_dir: Path) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    for series_id, url in FRED_SERIES.items():
        results.append(download_file(url, raw_dir / "fred" / f"{series_id}.csv"))
    return results


def fetch_french_factors(raw_dir: Path) -> DownloadResult:
    zip_path = raw_dir / "ken_french" / "F-F_Research_Data_Factors_CSV.zip"
    return download_file(FRENCH_FACTORS_URL, zip_path)


def parse_french_factors(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        file_name = archive.namelist()[0]
        text = archive.read(file_name).decode("latin1")
    rows: list[tuple[str, float, float, float, float]] = []
    in_monthly = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if in_monthly and rows:
                break
            continue
        if stripped.startswith(",Mkt-RF"):
            in_monthly = True
            continue
        if not in_monthly:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) != 5 or len(parts[0]) != 6 or not parts[0].isdigit():
            continue
        rows.append((parts[0], *(float(value) for value in parts[1:])))
    frame = pd.DataFrame(rows, columns=["yyyymm", "MktRF", "SMB", "HML", "RF"])
    for column in ["MktRF", "SMB", "HML", "RF"]:
        frame[column] = frame[column] / 100.0
    frame["date"] = pd.to_datetime(frame["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    return frame.drop(columns=["yyyymm"]).sort_values("date").reset_index(drop=True)


def fetch_all_data(raw_dir: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    open_asset_pricing = fetch_open_asset_pricing(raw_dir)
    fred_results = fetch_fred_series(raw_dir)
    french_factors = fetch_french_factors(raw_dir)
    manifest.update(
        {
            "open_asset_pricing": {
                "path": str(open_asset_pricing.path),
                "sha256": open_asset_pricing.sha256,
                "source": open_asset_pricing.source,
            },
            "fred": {
                result.path.stem: {
                    "path": str(result.path),
                    "sha256": result.sha256,
                    "source": result.source,
                }
                for result in fred_results
            },
            "ken_french": {
                "path": str(french_factors.path),
                "sha256": french_factors.sha256,
                "source": french_factors.source,
            },
        }
    )
    write_manifest(manifest_path, manifest)
    return manifest


def load_open_asset_pricing(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame


def load_fred_frame(raw_dir: Path) -> pd.DataFrame:
    series_frames: list[pd.DataFrame] = []
    for series_id in FRED_SERIES:
        frame = pd.read_csv(raw_dir / "fred" / f"{series_id}.csv")
        frame["observation_date"] = pd.to_datetime(frame["observation_date"])
        frame[series_id] = pd.to_numeric(frame[series_id], errors="coerce")
        monthly = (
            frame.set_index("observation_date")
            .resample("ME")
            .last()
            .rename_axis("date")
            .reset_index()[["date", series_id]]
        )
        series_frames.append(monthly)
    merged = series_frames[0]
    for frame in series_frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    return merged.sort_values("date").reset_index(drop=True)
