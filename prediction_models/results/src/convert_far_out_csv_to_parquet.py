from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SENSOR_COLUMNS = [
    "Best_Acc_X",
    "Best_Acc_Y",
    "Best_Acc_Z",
    "Best_AngVel_X",
    "Best_AngVel_Y",
    "Best_AngVel_Z",
    "Barometer_Value",
    "Sensor_Value",
]

STRUCTURE_COLUMNS = [
    *SENSOR_COLUMNS,
    "Thrust",
    "Mass",
    "Position_X",
    "Position_Y",
    "Position_Z",
    "Acceleration_X",
    "Acceleration_Y",
    "Acceleration_Z",
    "flight_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CSV telemetry into a flight_*.parquet file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("far_out_26_data"),
        help="Directory containing imu.csv, baro.csv, and flightInfo.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("far_out_26_data/converted/flight_far_out_26.parquet"),
        help="Output parquet path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path. Defaults to <output>.report.json.",
    )
    parser.add_argument("--flight-id", default="far_out_26_actual")
    parser.add_argument(
        "--timeline",
        choices=("imu", "baro", "union"),
        default="imu",
        help="Master timeline for the converted parquet.",
    )
    parser.add_argument(
        "--temperature-unit",
        choices=("celsius", "kelvin", "raw"),
        default="celsius",
        help="Unit in baro.csv column T. Celsius is converted to Kelvin for Sensor_Value.",
    )
    parser.add_argument(
        "--barometer-source",
        choices=("pressure", "filtered_altitude"),
        default="pressure",
        help="Use pressure P from baro.csv or filteredAltitudeAGL as Barometer_Value.",
    )
    parser.add_argument(
        "--vertical-label-source",
        choices=("flight_info", "filtered_data"),
        default="flight_info",
        help="Source for Position_Z and Acceleration_Z label columns.",
    )
    parser.add_argument(
        "--include-label-placeholders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include target/position columns needed by existing eval loaders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print report without writing parquet/report files.",
    )
    return parser.parse_args()


def read_csv(input_dir: Path, name: str, required: list[str]) -> pd.DataFrame:
    path = input_dir / name
    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_csv(path)
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    for col in data.columns:
        if col != "id":
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["ts"]).sort_values("ts")
    data = data.drop_duplicates(subset=["ts"], keep="last")
    return data.reset_index(drop=True)


def interpolate(times: np.ndarray, source: pd.DataFrame, column: str) -> np.ndarray:
    values = source[column].to_numpy(dtype=np.float64)
    source_times = source["ts"].to_numpy(dtype=np.float64)
    valid = np.isfinite(source_times) & np.isfinite(values)
    if valid.sum() == 0:
        raise ValueError(f"No finite values available for {column}")
    return np.interp(times, source_times[valid], values[valid]).astype(np.float32)


def convert_temperature(values: np.ndarray, unit: str) -> np.ndarray:
    if unit == "celsius":
        return (values + 273.15).astype(np.float32)
    return values.astype(np.float32)


def build_timeline(args: argparse.Namespace, imu: pd.DataFrame, baro: pd.DataFrame) -> np.ndarray:
    if args.timeline == "imu":
        times = imu["ts"].to_numpy(dtype=np.float64)
    elif args.timeline == "baro":
        times = baro["ts"].to_numpy(dtype=np.float64)
    else:
        times = np.union1d(
            imu["ts"].to_numpy(dtype=np.float64),
            baro["ts"].to_numpy(dtype=np.float64),
        )
    return np.unique(times[np.isfinite(times)])


def build_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    input_dir = args.input_dir.expanduser().resolve()
    imu = read_csv(input_dir, "imu.csv", ["ts", "Ax", "Ay", "Az", "Gx", "Gy", "Gz"])
    baro = read_csv(input_dir, "baro.csv", ["ts", "T", "P"])
    flight_info = read_csv(input_dir, "flightInfo.csv", ["ts", "height", "acceleration"])
    filtered_data = read_csv(
        input_dir,
        "filteredDataInfo.csv",
        ["ts", "filteredAltitudeAGL", "filteredAcceleration"],
    )

    times = build_timeline(args, imu, baro)
    frame = pd.DataFrame(index=pd.Index(times.astype(np.float32), name="Time"))

    frame["Best_Acc_X"] = interpolate(times, imu, "Ax")
    frame["Best_Acc_Y"] = interpolate(times, imu, "Ay")
    frame["Best_Acc_Z"] = interpolate(times, imu, "Az")
    frame["Best_AngVel_X"] = interpolate(times, imu, "Gx")
    frame["Best_AngVel_Y"] = interpolate(times, imu, "Gy")
    frame["Best_AngVel_Z"] = interpolate(times, imu, "Gz")

    if args.barometer_source == "pressure":
        frame["Barometer_Value"] = interpolate(times, baro, "P")
    else:
        frame["Barometer_Value"] = interpolate(times, filtered_data, "filteredAltitudeAGL")

    frame["Sensor_Value"] = convert_temperature(
        interpolate(times, baro, "T"),
        args.temperature_unit,
    )

    if args.include_label_placeholders:
        if args.vertical_label_source == "flight_info":
            position_z = interpolate(times, flight_info, "height")
            acceleration_z = interpolate(times, flight_info, "acceleration")
        else:
            position_z = interpolate(times, filtered_data, "filteredAltitudeAGL")
            acceleration_z = interpolate(times, filtered_data, "filteredAcceleration")

        # The real export does not contain full 3D inertial truth. These columns
        # keep the parquet compatible with existing loaders but must not be used
        # as full 3D ground truth.
        frame["Thrust"] = np.nan
        frame["Mass"] = np.nan
        frame["Position_X"] = 0.0
        frame["Position_Y"] = 0.0
        frame["Position_Z"] = position_z
        frame["Acceleration_X"] = 0.0
        frame["Acceleration_Y"] = 0.0
        frame["Acceleration_Z"] = acceleration_z
        frame["flight_id"] = args.flight_id
        frame = frame[STRUCTURE_COLUMNS]
    else:
        frame = frame[SENSOR_COLUMNS]

    report = make_report(args, input_dir, frame, imu, baro, flight_info, filtered_data)
    return frame, report


def column_stats(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | int | None]]:
    stats: dict[str, dict[str, float | int | None]] = {}
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce")
        finite = values[np.isfinite(values)]
        stats[col] = {
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
            "mean": float(finite.mean()) if len(finite) else None,
            "nan_count": int(values.isna().sum()),
        }
    return stats


def make_report(
    args: argparse.Namespace,
    input_dir: Path,
    frame: pd.DataFrame,
    imu: pd.DataFrame,
    baro: pd.DataFrame,
    flight_info: pd.DataFrame,
    filtered_data: pd.DataFrame,
) -> dict[str, object]:
    time_values = frame.index.to_numpy(dtype=np.float64)
    dt = np.diff(time_values)
    finite_dt = dt[np.isfinite(dt) & (dt > 0)]
    warnings = [
        "Position_X/Y and Acceleration_X/Y are placeholders if label columns are included.",
    ]
    if args.temperature_unit == "celsius":
        warnings.append("baro.csv T was converted from Celsius to Kelvin for Sensor_Value.")
    if args.barometer_source != "pressure":
        warnings.append("Barometer_Value is altitude, not pressure; this differs from the synthetic barometer.")

    return {
        "input_dir": str(input_dir),
        "output": str(args.output.expanduser().resolve()),
        "rows": len(frame),
        "columns": list(frame.columns),
        "sensor_column_order": SENSOR_COLUMNS,
        "time_start_s": float(time_values[0]) if len(time_values) else None,
        "time_end_s": float(time_values[-1]) if len(time_values) else None,
        "dt_median_s": float(np.median(finite_dt)) if len(finite_dt) else None,
        "dt_min_s": float(np.min(finite_dt)) if len(finite_dt) else None,
        "dt_max_s": float(np.max(finite_dt)) if len(finite_dt) else None,
        "source_rows": {
            "imu.csv": len(imu),
            "baro.csv": len(baro),
            "flightInfo.csv": len(flight_info),
            "filteredDataInfo.csv": len(filtered_data),
        },
        "options": {
            "timeline": args.timeline,
            "temperature_unit": args.temperature_unit,
            "barometer_source": args.barometer_source,
            "vertical_label_source": args.vertical_label_source,
            "include_label_placeholders": bool(args.include_label_placeholders),
        },
        "stats": column_stats(frame, [c for c in frame.columns if c != "flight_id"]),
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    frame, report = build_frame(args)

    print(json.dumps(report, indent=2))
    if args.dry_run:
        return

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=True)

    report_path = args.report
    if report_path is None:
        report_path = output.with_suffix(output.suffix + ".report.json")
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nWrote parquet: {output}")
    print(f"Wrote report:  {report_path}")


if __name__ == "__main__":
    main()
