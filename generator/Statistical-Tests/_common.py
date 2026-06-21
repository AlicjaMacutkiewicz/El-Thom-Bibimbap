"""Small shared helpers for independently runnable statistical tests"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TEST_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = TEST_DIR.parent / "src" / "output"
DEFAULT_OUTPUT = TEST_DIR / "plots_generalized"
DATE_PATTERN = re.compile(r"flight_(\d{4})-\d{2}-\d{2}(?:_.*)?\.parquet$")

FEATURE_COLUMNS = [
    "Best_Acc_X", "Best_Acc_Y", "Best_Acc_Z",
    "Best_AngVel_X", "Best_AngVel_Y", "Best_AngVel_Z",
    "Barometer_Value", "Sensor_Value", "Thrust", "Mass",
    "Position_X", "Position_Y", "Position_Z",
    "Acceleration_X", "Acceleration_Y", "Acceleration_Z",
]


def test_parser(description: str, samples: int = 512) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=1990)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--representative-flights", type=int, default=35)
    parser.add_argument("--samples-per-flight", type=int, default=samples)
    parser.add_argument("--seed", type=int, default=41)
    return parser


def discover_files(input_dir: Path, start_year: int, end_year: int) -> list[Path]:
    files = []
    for path in sorted(input_dir.glob("flight_*.parquet")):
        match = DATE_PATTERN.match(path.name)
        if match and start_year <= int(match.group(1)) <= end_year:
            files.append(path)
    return files


def choose_representatives(files: list[Path], count: int, seed: int) -> list[Path]:
    rng = np.random.default_rng(seed)
    by_year: dict[int, list[Path]] = {}
    for path in files:
        year = int(DATE_PATTERN.match(path.name).group(1))
        by_year.setdefault(year, []).append(path)
    years = sorted(by_year)
    if count < len(years):
        years = [years[index] for index in np.linspace(0, len(years) - 1, count, dtype=int)]
    selected = [by_year[year][int(rng.integers(len(by_year[year])))] for year in years]
    return selected[:count]


def load_representatives(args: argparse.Namespace) -> tuple[list[Path], list[np.ndarray]]:
    files = discover_files(args.input_dir, args.start_year, args.end_year)
    selected = choose_representatives(files, args.representative_flights, args.seed)
    if not selected:
        raise SystemExit("No dated flight files found in the requested range.")
    samples = []
    for path in selected:
        values = pd.read_parquet(path, columns=FEATURE_COLUMNS).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {path.name}")
        indices = np.linspace(0, len(values) - 1, args.samples_per_flight, dtype=int)
        samples.append(values[indices])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using {len(selected)} representative flights from {len(files)} files.")
    return selected, samples


def standardize(values: np.ndarray) -> np.ndarray:
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (values - values.mean(axis=0)) / scale


def save_heatmap(
    matrix: np.ndarray,
    paths: list[Path],
    title: str,
    output_path: Path,
    fmt: str = ".2f",
) -> None:
    labels = [path.name[7:11] for path in paths]
    figure, axis = plt.subplots(figsize=(11, 9))
    sns.heatmap(matrix, cmap="coolwarm", xticklabels=labels, yticklabels=labels, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Representative flight year")
    axis.set_ylabel("Representative flight year")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

