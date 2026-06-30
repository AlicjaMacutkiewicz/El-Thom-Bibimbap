from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6378137.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a coarse 3D real-flight trajectory from GNSS horizontal "
            "position and filtered barometric altitude."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing gnssInfo.csv and filteredDataInfo.csv.",
    )
    parser.add_argument(
        "--gnss",
        type=Path,
        default=None,
        help="GNSS CSV path. Defaults to <input-dir>/gnssInfo.csv.",
    )
    parser.add_argument(
        "--filtered",
        type=Path,
        default=None,
        help="Filtered altitude CSV path. Defaults to <input-dir>/filteredDataInfo.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input-dir>/converted/trajectory_gnss.csv.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Optional static 3D trajectory plot. Defaults to <output>.png.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report. Defaults to <output>.report.json.",
    )
    parser.add_argument(
        "--min-satellites",
        type=int,
        default=6,
        help="Discard GNSS samples with fewer satellites. Set to 0 to disable.",
    )
    parser.add_argument(
        "--origin-time",
        type=float,
        default=0.0,
        help=(
            "Reference time for the local origin. The nearest remaining GNSS "
            "sample defines X/Y origin; altitude is zeroed at this timestamp."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Write CSV/report only.",
    )
    return parser.parse_args()


def read_csv(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["ts"]).sort_values("ts")
    return data.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)


def latlon_to_local(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: float,
    lon0: float,
) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    north = (lat_rad - lat0_rad) * EARTH_RADIUS_M
    east = (lon_rad - lon0_rad) * EARTH_RADIUS_M * np.cos(lat0_rad)
    return east.astype(np.float32), north.astype(np.float32)


def interpolate_column(times: np.ndarray, source: pd.DataFrame, column: str) -> np.ndarray:
    values = source[column].to_numpy(dtype=np.float64)
    source_times = source["ts"].to_numpy(dtype=np.float64)
    finite = np.isfinite(source_times) & np.isfinite(values)
    if finite.sum() == 0:
        raise ValueError(f"No finite values available for {column}.")
    return np.interp(times, source_times[finite], values[finite]).astype(np.float32)


def build_trajectory(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    input_dir = args.input_dir.expanduser().resolve()
    gnss_path = args.gnss.expanduser().resolve() if args.gnss else input_dir / "gnssInfo.csv"
    filtered_path = (
        args.filtered.expanduser().resolve()
        if args.filtered
        else input_dir / "filteredDataInfo.csv"
    )

    gnss_raw = read_csv(gnss_path, ["ts", "latitude", "longitude", "satellites"])
    filtered = read_csv(filtered_path, ["ts", "filteredAltitudeAGL"])

    gnss = gnss_raw.copy()
    if args.min_satellites > 0 and "satellites" in gnss.columns:
        gnss = gnss[gnss["satellites"] >= args.min_satellites].copy()
    gnss = gnss.dropna(subset=["latitude", "longitude"])
    if gnss.empty:
        raise RuntimeError("No GNSS samples remain after filtering.")

    origin_index = int(np.abs(gnss["ts"].to_numpy(dtype=np.float64) - args.origin_time).argmin())
    origin = gnss.iloc[origin_index]
    lat0 = float(origin["latitude"])
    lon0 = float(origin["longitude"])
    origin_time = float(origin["ts"])

    east, north = latlon_to_local(
        gnss["latitude"].to_numpy(dtype=np.float64),
        gnss["longitude"].to_numpy(dtype=np.float64),
        lat0,
        lon0,
    )
    altitude = interpolate_column(
        gnss["ts"].to_numpy(dtype=np.float64),
        filtered,
        "filteredAltitudeAGL",
    )
    altitude_origin = float(
        np.interp(
            args.origin_time,
            filtered["ts"].to_numpy(dtype=np.float64),
            filtered["filteredAltitudeAGL"].to_numpy(dtype=np.float64),
        )
    )

    trajectory = pd.DataFrame(
        {
            "Time": gnss["ts"].to_numpy(dtype=np.float32),
            "Position_X": east,
            "Position_Y": north,
            "Position_Z": altitude - altitude_origin,
            "latitude": gnss["latitude"].to_numpy(dtype=np.float64),
            "longitude": gnss["longitude"].to_numpy(dtype=np.float64),
            "satellites": gnss["satellites"].to_numpy(dtype=np.float32),
        }
    )

    report = {
        "scope": (
            "Coarse GNSS-derived 3D reference. X/Y come from low-rate GNSS; "
            "Z is interpolated filtered barometric altitude. This is not exact "
            "high-rate 3D ground truth."
        ),
        "gnss_path": str(gnss_path),
        "filtered_path": str(filtered_path),
        "raw_gnss_rows": len(gnss_raw),
        "used_gnss_rows": len(gnss),
        "min_satellites": int(args.min_satellites),
        "requested_origin_time_s": float(args.origin_time),
        "origin_gnss_time_s": origin_time,
        "origin_latitude": lat0,
        "origin_longitude": lon0,
        "time_start_s": float(trajectory["Time"].iloc[0]),
        "time_end_s": float(trajectory["Time"].iloc[-1]),
        "x_range_m": [
            float(trajectory["Position_X"].min()),
            float(trajectory["Position_X"].max()),
        ],
        "y_range_m": [
            float(trajectory["Position_Y"].min()),
            float(trajectory["Position_Y"].max()),
        ],
        "z_range_m": [
            float(trajectory["Position_Z"].min()),
            float(trajectory["Position_Z"].max()),
        ],
        "warnings": [
            "GNSS is low-rate relative to IMU/barometer telemetry.",
            "GNSS latency/noise can dominate horizontal error metrics.",
            "Use this reference for coarse diagnostics, not exact 3D validation.",
        ],
    }
    return trajectory, report


def render_plot(trajectory: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        trajectory["Position_X"],
        trajectory["Position_Y"],
        trajectory["Position_Z"],
        color="green",
        linewidth=2,
        label="GNSS + baro trajectory",
    )
    ax.scatter(
        trajectory["Position_X"].iloc[0],
        trajectory["Position_Y"].iloc[0],
        trajectory["Position_Z"].iloc[0],
        color="green",
        s=50,
        label="start",
    )
    ax.scatter(
        trajectory["Position_X"].iloc[-1],
        trajectory["Position_Y"].iloc[-1],
        trajectory["Position_Z"].iloc[-1],
        color="red",
        s=50,
        label="end",
    )
    ax.set_xlabel("East / X [m]")
    ax.set_ylabel("North / Y [m]")
    ax.set_zlabel("Altitude / Z [m]")
    ax.set_title("Coarse GNSS-Derived 3D Trajectory")
    ax.legend()
    ax.view_init(elev=30, azim=-60)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = args.input_dir / "converted" / "trajectory_gnss.csv"
    args.output = args.output.expanduser().resolve()
    if args.plot_output is None:
        args.plot_output = args.output.with_suffix(".png")
    args.plot_output = args.plot_output.expanduser().resolve()
    if args.report is None:
        args.report = args.output.with_suffix(args.output.suffix + ".report.json")
    args.report = args.report.expanduser().resolve()

    trajectory, report = build_trajectory(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(args.output, index=False)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.no_plot:
        render_plot(trajectory, args.plot_output)

    print(json.dumps({**report, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
