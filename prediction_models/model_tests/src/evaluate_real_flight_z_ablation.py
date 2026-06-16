#!/usr/bin/env python3
"""Real-flight Z-axis replay for GRU ablation checkpoints.

This evaluator is intentionally scoped to the available FAR-OUT 2026 references:

* model input uses real IMU/gyro/barometer/temperature telemetry,
* acceleration is evaluated on the available vertical filtered acceleration,
* position is evaluated on the available vertical altitude/height reference,
* X/Y and full 3D trajectory accuracy are not evaluated.

The script compares the same three neural variants used in the synthetic
ablation test:

* Plain GRU,
* GRU-RK4,
* GRU-RK4 + physics.

It also reports the non-learning baselines used in the synthetic evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex-cache"))

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_gru import (  # noqa: E402  # type: ignore
    GRU_RK4_METHOD,
    GRU_RK4_PHYS_METHOD,
    PLAIN_GRU_METHOD,
    SENSOR_COLUMNS,
    ModelSpec,
    baseline_acceleration,
    configure_imports,
    load_networks,
    predict_model_accelerations,
)


Z_POSITION_COLUMN = "Position_Z"
Z_ACCELERATION_COLUMN = "Acceleration_Z"
POSITION_METHODS = [
    PLAIN_GRU_METHOD,
    GRU_RK4_METHOD,
    GRU_RK4_PHYS_METHOD,
    "Polynomial",
    "RK4 only",
    "Last acceleration",
    "Oracle acceleration",
]
ACCELERATION_METHODS = [
    PLAIN_GRU_METHOD,
    GRU_RK4_METHOD,
    GRU_RK4_PHYS_METHOD,
    "RK4 only",
    "Last acceleration",
]
COLORS = {
    PLAIN_GRU_METHOD: "#17becf",
    GRU_RK4_METHOD: "#d62728",
    GRU_RK4_PHYS_METHOD: "#8c564b",
    "Polynomial": "#1f77b4",
    "RK4 only": "#9467bd",
    "Last acceleration": "#ff7f0e",
    "Oracle acceleration": "#2ca02c",
}


@dataclass
class OneDimMetric:
    squared_sum: float = 0.0
    absolute_sum: float = 0.0
    point_count: int = 0
    window_rmse: list[np.ndarray] = field(default_factory=list)
    failures: int = 0

    def add(self, prediction: np.ndarray, truth: np.ndarray, threshold: float) -> np.ndarray:
        error = prediction - truth
        window = np.sqrt(np.mean(np.square(error), axis=1))
        self.squared_sum += float(np.square(error).sum())
        self.absolute_sum += float(np.abs(error).sum())
        self.point_count += int(error.size)
        self.window_rmse.append(window.astype(np.float32, copy=False))
        self.failures += int((window > threshold).sum())
        return window

    def summarize(self) -> dict[str, float | int | None]:
        if not self.window_rmse or self.point_count == 0:
            return {
                "point_rmse": None,
                "point_mae": None,
                "mean_window_rmse": None,
                "median_window_rmse": None,
                "p95_window_rmse": None,
                "p99_window_rmse": None,
                "max_window_rmse": None,
                "failures_over_threshold": 0,
                "windows": 0,
                "failure_rate_pct": 0.0,
            }
        windows = np.concatenate(self.window_rmse)
        return {
            "point_rmse": float(np.sqrt(self.squared_sum / self.point_count)),
            "point_mae": float(self.absolute_sum / self.point_count),
            "mean_window_rmse": float(windows.mean()),
            "median_window_rmse": float(np.median(windows)),
            "p95_window_rmse": float(np.quantile(windows, 0.95)),
            "p99_window_rmse": float(np.quantile(windows, 0.99)),
            "max_window_rmse": float(windows.max()),
            "failures_over_threshold": self.failures,
            "windows": int(windows.size),
            "failure_rate_pct": float(100.0 * self.failures / windows.size),
        }


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    repo_default = Path.cwd()
    parser = argparse.ArgumentParser(
        description="Evaluate non-retrained GRU ablation models on one real flight, Z axis only."
    )
    parser.add_argument("--repo", type=Path, default=repo_default)
    parser.add_argument(
        "--flight",
        type=Path,
        default=repo_default / "far_out_26_data" / "converted" / "flight_far_out_26.parquet",
        help="Converted real-flight parquet from convert_far_out_csv_to_parquet.py.",
    )
    parser.add_argument(
        "--gru-model",
        type=Path,
        default=None,
        help="Plain GRU direct-acceleration checkpoint.",
    )
    parser.add_argument(
        "--gru-res-model",
        type=Path,
        default=None,
        help="Residual GRU checkpoint without trajectory-consistency loss.",
    )
    parser.add_argument(
        "--gru-res-phys-model",
        type=Path,
        default=None,
        help="Residual GRU checkpoint with trajectory-consistency loss.",
    )
    parser.add_argument(
        "--scaler-npz",
        type=Path,
        default=None,
        help=(
            "Scaler file with mean_in/std_in/mean_acc/std_acc/mean_xs/std_xs. "
            "Defaults to prediction_models/model_tests/gru_ablation_eval/reconstructed_scalers.npz."
        ),
    )
    parser.add_argument("--parameters", type=Path, default=None)
    parser.add_argument("--thrust-curve", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--pred-len", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--position-threshold", type=float, default=10.0)
    parser.add_argument("--acc-threshold", type=float, default=5.0)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="auto uses CUDA, then MPS, then CPU.",
    )
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--min-start-time",
        type=float,
        default=0.0,
        help="Only evaluate windows whose prediction starts at or after this time.",
    )
    parser.add_argument(
        "--max-start-time",
        type=float,
        default=None,
        help="Optional upper bound for evaluated prediction start times.",
    )
    parser.add_argument(
        "--acc-axis-map",
        default="X,Z,-Y",
        help=(
            "Map source acceleration axes into model X/Y/Z. "
            "Examples: X,Y,Z or X,Z,-Y."
        ),
    )
    parser.add_argument(
        "--gyro-axis-map",
        default="X,Z,-Y",
        help="Map source gyro axes into model X/Y/Z, with the same syntax as --acc-axis-map.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    args.repo = args.repo.expanduser().resolve()
    args.flight = args.flight.expanduser().resolve()

    model_root = args.repo / "prediction_models" / "model_tests" / "final_model"
    if args.gru_model is None:
        args.gru_model = model_root / "gru_model.pth"
    else:
        args.gru_model = args.gru_model.expanduser().resolve()
    if args.gru_res_model is None:
        args.gru_res_model = model_root / "gru_res_model.pth"
    else:
        args.gru_res_model = args.gru_res_model.expanduser().resolve()
    if args.gru_res_phys_model is None:
        args.gru_res_phys_model = model_root / "gru_res_phys_model.pth"
    else:
        args.gru_res_phys_model = args.gru_res_phys_model.expanduser().resolve()

    if args.scaler_npz is None:
        args.scaler_npz = (
            args.repo
            / "prediction_models"
            / "model_tests"
            / "gru_ablation_eval"
            / "reconstructed_scalers.npz"
        )
    else:
        args.scaler_npz = args.scaler_npz.expanduser().resolve()

    farout_root = args.repo / "source_model" / "R7_SIMLE" / "R7_FAROUT_26"
    if args.parameters is None:
        args.parameters = farout_root / "parameters.json"
    else:
        args.parameters = args.parameters.expanduser().resolve()
    if args.thrust_curve is None:
        args.thrust_curve = farout_root / "thrust_source_pressure85.csv"
    else:
        args.thrust_curve = args.thrust_curve.expanduser().resolve()

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = args.repo / "far_out_26_data" / f"real_flight_z_ablation_{stamp}"
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)


def parse_axis_map(text: str) -> list[tuple[int, float]]:
    lookup = {"X": 0, "Y": 1, "Z": 2}
    parts = [part.strip().upper() for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("Axis maps must contain three comma-separated entries, e.g. X,Y,Z.")
    result: list[tuple[int, float]] = []
    for part in parts:
        sign = -1.0 if part.startswith("-") else 1.0
        axis = part[1:] if part.startswith(("-", "+")) else part
        if axis not in lookup:
            raise ValueError(f"Unknown axis mapping entry: {part!r}")
        result.append((lookup[axis], sign))
    return result


def apply_axis_map(values: np.ndarray, mapping: list[tuple[int, float]]) -> np.ndarray:
    return np.stack([sign * values[:, source] for source, sign in mapping], axis=1).astype(np.float32)


def load_real_flight(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_parquet(args.flight)
    required = set(SENSOR_COLUMNS + [Z_POSITION_COLUMN, Z_ACCELERATION_COLUMN])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{args.flight} is missing required columns: {missing}")

    source_inputs = data[SENSOR_COLUMNS].to_numpy(dtype=np.float32)
    acc_map = parse_axis_map(args.acc_axis_map)
    gyro_map = parse_axis_map(args.gyro_axis_map)
    inputs = np.empty_like(source_inputs, dtype=np.float32)
    inputs[:, :3] = apply_axis_map(source_inputs[:, :3], acc_map)
    inputs[:, 3:6] = apply_axis_map(source_inputs[:, 3:6], gyro_map)
    inputs[:, 6:] = source_inputs[:, 6:]

    position_z = data[Z_POSITION_COLUMN].to_numpy(dtype=np.float32)
    acceleration_z = data[Z_ACCELERATION_COLUMN].to_numpy(dtype=np.float32)
    if "Time" in data.columns:
        times = data["Time"].to_numpy(dtype=np.float32)
    else:
        times = data.index.to_numpy(dtype=np.float32)

    inputs = inputs[:: args.downsample]
    position_z = position_z[:: args.downsample]
    acceleration_z = acceleration_z[:: args.downsample]
    times = times[:: args.downsample]

    order = np.argsort(times)
    inputs = inputs[order]
    position_z = position_z[order]
    acceleration_z = acceleration_z[order]
    times = times[order]

    finite = np.isfinite(times) & np.isfinite(position_z) & np.isfinite(acceleration_z)
    finite &= np.all(np.isfinite(inputs), axis=1)
    return inputs[finite], position_z[finite], acceleration_z[finite], times[finite]


def make_windows(
    inputs: np.ndarray,
    position_z: np.ndarray,
    acceleration_z: np.ndarray,
    times: np.ndarray,
    seq_len: int,
    pred_len: int,
    min_start_time: float,
    max_start_time: float | None,
) -> dict[str, np.ndarray]:
    starts = np.arange(seq_len, len(times) - pred_len, dtype=np.int64)
    start_times = times[starts]
    keep = start_times >= min_start_time
    if max_start_time is not None:
        keep &= start_times <= max_start_time
    starts = starts[keep]
    if len(starts) == 0:
        raise RuntimeError("No valid windows for this seq_len/pred_len/time range.")

    lookback = starts[:, None] - seq_len + np.arange(seq_len)[None, :]
    future = starts[:, None] + np.arange(pred_len)[None, :]
    previous = starts - 1
    before_previous = starts - 2
    dt = np.maximum(times[previous] - times[before_previous], 1e-6)

    return {
        "starts": starts,
        "input_windows": inputs[lookback],
        "lookback_times": times[lookback],
        "lookback_position_z": position_z[lookback],
        "future_times": times[future],
        "actual_position_z": position_z[future],
        "actual_acceleration_z": acceleration_z[future],
        "initial_position_z": position_z[previous],
        "initial_velocity_z": (position_z[previous] - position_z[before_previous]) / dt,
        "initial_time": times[previous],
        "previous_acceleration_z": acceleration_z[previous],
    }


def integrate_position_z(
    acceleration_z: np.ndarray,
    future_times: np.ndarray,
    initial_position_z: np.ndarray,
    initial_velocity_z: np.ndarray,
    initial_time: np.ndarray,
) -> np.ndarray:
    positions = np.empty_like(acceleration_z, dtype=np.float32)
    position = initial_position_z.astype(np.float32, copy=True)
    velocity = initial_velocity_z.astype(np.float32, copy=True)
    previous_time = initial_time.astype(np.float32, copy=True)
    for step in range(acceleration_z.shape[1]):
        dt = np.maximum(future_times[:, step] - previous_time, 0.0)
        current_acc = acceleration_z[:, step]
        position = position + velocity * dt + 0.5 * current_acc * dt * dt
        velocity = velocity + current_acc * dt
        positions[:, step] = position
        previous_time = future_times[:, step]
    return positions


def polynomial_prediction_z(
    lookback_times: np.ndarray,
    lookback_position_z: np.ndarray,
    future_times: np.ndarray,
) -> np.ndarray:
    x = (lookback_times - lookback_times[:, :1]).astype(np.float64)
    future = (future_times - lookback_times[:, :1]).astype(np.float64)
    s0 = np.full(len(x), x.shape[1], dtype=np.float64)
    s1 = x.sum(axis=1)
    s2 = np.square(x).sum(axis=1)
    s3 = np.power(x, 3).sum(axis=1)
    s4 = np.power(x, 4).sum(axis=1)
    matrix = np.stack(
        [
            np.stack([s0, s1, s2], axis=1),
            np.stack([s1, s2, s3], axis=1),
            np.stack([s2, s3, s4], axis=1),
        ],
        axis=1,
    )
    y = lookback_position_z.astype(np.float64)
    rhs = np.stack(
        [y.sum(axis=1), (x * y).sum(axis=1), (np.square(x) * y).sum(axis=1)],
        axis=1,
    )[..., None]
    coefficients = np.linalg.solve(matrix, rhs)[..., 0]
    return (
        coefficients[:, 0, None]
        + coefficients[:, 1, None] * future
        + coefficients[:, 2, None] * np.square(future)
    ).astype(np.float32)


def method_key(method: str) -> str:
    key = "".join(character.lower() if character.isalnum() else "_" for character in method)
    return "_".join(part for part in key.split("_") if part)


def select_device(args: argparse.Namespace) -> tuple[torch.device, list[int]]:
    if args.device in {"auto", "cuda"} and torch.cuda.is_available():
        ids = [int(part.strip()) for part in args.gpu_ids.split(",") if part.strip()]
        if not ids:
            ids = [0]
        device = torch.device(f"cuda:{ids[0]}")
        torch.cuda.set_device(device)
        log(f"Using CUDA device {device}: {torch.cuda.get_device_name(device)}")
        return device, ids
    if args.device == "cuda":
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    if args.device in {"auto", "mps"} and torch.backends.mps.is_available():
        log("Using Apple MPS device.")
        return torch.device("mps"), []
    if args.device == "mps":
        raise RuntimeError("--device mps requested but MPS is not available.")
    log("Using CPU.")
    return torch.device("cpu"), []


def load_scalers(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Scaler file not found: {path}")
    scalers = np.load(path)
    required = ["mean_in", "std_in", "mean_acc", "std_acc", "mean_xs", "std_xs"]
    missing = [key for key in required if key not in scalers.files]
    if missing:
        raise KeyError(
            f"{path} is missing {missing}. The real-flight ablation evaluator needs the "
            "ablation scaler file with direct-acceleration and residual statistics."
        )
    return tuple(scalers[key].astype(np.float32) for key in required)  # type: ignore[return-value]


def evaluate(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    GRU, calculate_x_b, load_parameters, load_thrust_curve = configure_imports(args.repo)
    parameters = load_parameters(args.parameters)
    thrust_curve = load_thrust_curve(args.thrust_curve)

    inputs, position_z, acceleration_z, times = load_real_flight(args)
    if len(times) < args.seq_len + args.pred_len + 2:
        raise RuntimeError(
            f"Not enough rows after downsampling: {len(times)} rows, "
            f"need at least {args.seq_len + args.pred_len + 2}."
        )

    dt = np.diff(times)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    sampling_rate = float(1.0 / np.median(dt))

    model_specs = [
        ModelSpec(PLAIN_GRU_METHOD, args.gru_model, "direct"),
        ModelSpec(GRU_RK4_METHOD, args.gru_res_model, "residual"),
        ModelSpec(GRU_RK4_PHYS_METHOD, args.gru_res_phys_model, "residual"),
    ]
    for spec in model_specs:
        if not spec.path.exists():
            raise FileNotFoundError(f"{spec.name} checkpoint not found: {spec.path}")

    mean_in, std_in, mean_acc, std_acc, mean_xs, std_xs = load_scalers(args.scaler_npz)
    device, gpu_ids = select_device(args)
    models = load_networks(GRU, model_specs, device, gpu_ids)

    windows = make_windows(
        inputs,
        position_z,
        acceleration_z,
        times,
        args.seq_len,
        args.pred_len,
        args.min_start_time,
        args.max_start_time,
    )
    base_acc = baseline_acceleration(
        calculate_x_b,
        windows["future_times"].astype(np.float32),
        parameters,
        thrust_curve,
        sampling_rate,
    )
    model_acc_3d = predict_model_accelerations(
        model_specs,
        models,
        windows["input_windows"],
        base_acc,
        mean_in,
        std_in,
        mean_acc,
        std_acc,
        mean_xs,
        std_xs,
        args.pred_len,
        args.batch_size,
        device,
        args.amp,
    )

    acceleration_predictions: dict[str, np.ndarray] = {
        spec.name: model_acc_3d[spec.name][:, :, 2] for spec in model_specs
    }
    acceleration_predictions["RK4 only"] = base_acc[:, :, 2]
    acceleration_predictions["Last acceleration"] = np.repeat(
        windows["previous_acceleration_z"][:, None], args.pred_len, axis=1
    )
    acceleration_predictions["Oracle acceleration"] = windows["actual_acceleration_z"]

    position_predictions: dict[str, np.ndarray] = {
        method: integrate_position_z(
            acceleration,
            windows["future_times"],
            windows["initial_position_z"],
            windows["initial_velocity_z"],
            windows["initial_time"],
        )
        for method, acceleration in acceleration_predictions.items()
    }
    position_predictions["Polynomial"] = polynomial_prediction_z(
        windows["lookback_times"],
        windows["lookback_position_z"],
        windows["future_times"],
    )

    position_metrics = {method: OneDimMetric() for method in POSITION_METHODS}
    acceleration_metrics = {method: OneDimMetric() for method in ACCELERATION_METHODS}
    window_rmse_by_method: dict[str, np.ndarray] = {}
    acc_rmse_by_method: dict[str, np.ndarray] = {}
    for method in POSITION_METHODS:
        window_rmse_by_method[method] = position_metrics[method].add(
            position_predictions[method],
            windows["actual_position_z"],
            args.position_threshold,
        )
    for method in ACCELERATION_METHODS:
        acc_rmse_by_method[method] = acceleration_metrics[method].add(
            acceleration_predictions[method],
            windows["actual_acceleration_z"],
            args.acc_threshold,
        )

    lead_seconds = windows["future_times"][:, -1] - windows["future_times"][:, 0]
    rows: list[dict] = []
    for index, start in enumerate(windows["starts"]):
        row: dict[str, object] = {
            "start_sample_after_downsample": int(start),
            "start_time_s": float(windows["future_times"][index, 0]),
            "lead_time_s": float(lead_seconds[index]),
        }
        for method in POSITION_METHODS:
            key = method_key(method)
            row[f"{key}_position_z_window_rmse_m"] = float(window_rmse_by_method[method][index])
            row[f"{key}_position_z_endpoint_error_m"] = float(
                position_predictions[method][index, -1]
                - windows["actual_position_z"][index, -1]
            )
        for method in ACCELERATION_METHODS:
            key = method_key(method)
            row[f"{key}_acceleration_z_window_rmse"] = float(acc_rmse_by_method[method][index])
        rows.append(row)

    summary = {
        "flight": str(args.flight),
        "models": [
            {"name": spec.name, "path": str(spec.path), "output_mode": spec.output_mode}
            for spec in model_specs
        ],
        "parameters": str(args.parameters),
        "thrust_curve": str(args.thrust_curve),
        "scaler_npz": str(args.scaler_npz),
        "output_dir": str(args.output_dir),
        "axis_maps": {"acc": args.acc_axis_map, "gyro": args.gyro_axis_map},
        "rows_after_downsample": int(len(times)),
        "time_start_s": float(times[0]),
        "time_end_s": float(times[-1]),
        "dt_median_s": float(np.median(dt)),
        "effective_sampling_rate_hz": sampling_rate,
        "downsample": args.downsample,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "horizon_seconds_median": float(np.median(lead_seconds)),
        "horizon_seconds_min": float(np.min(lead_seconds)),
        "horizon_seconds_max": float(np.max(lead_seconds)),
        "windows_evaluated": int(len(windows["starts"])),
        "position_threshold_m": args.position_threshold,
        "acc_threshold": args.acc_threshold,
        "position_z_metrics": {
            method: position_metrics[method].summarize() for method in POSITION_METHODS
        },
        "acceleration_z_metrics": {
            method: acceleration_metrics[method].summarize() for method in ACCELERATION_METHODS
        },
        "validation_scope": {
            "validated": [
                "real telemetry ingestion",
                "Z acceleration replay against telemetry-derived filtered acceleration",
                "integrated Z position replay against telemetry-derived altitude/height",
            ],
            "not_validated": [
                "exact X/Y position accuracy",
                "exact full 3D trajectory accuracy",
                "landing spot prediction",
                "live operational dropout handling",
            ],
        },
    }
    return summary, rows


def write_outputs(args: argparse.Namespace, summary: dict, rows: list[dict]) -> None:
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if rows:
        with (args.output_dir / "per_window_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "REAL FLIGHT REPLAY SUMMARY - Z AXIS ONLY - ALL NON-RETRAINED MODELS",
        f"Flight: {summary['flight']}",
        f"Parameters: {summary['parameters']}",
        f"Thrust curve: {summary['thrust_curve']}",
        f"Scalers: {summary['scaler_npz']}",
        f"Axis maps: acc={summary['axis_maps']['acc']}, gyro={summary['axis_maps']['gyro']}",
        f"Rows after downsampling: {summary['rows_after_downsample']:,}",
        f"Downsample: {summary['downsample']}  dt_median={summary['dt_median_s']:.5f}s",
        f"Prediction window: {summary['pred_len']} samples "
        f"(~{summary['horizon_seconds_median']:.2f}s)",
        f"Windows evaluated: {summary['windows_evaluated']:,}",
        f"Position threshold: >{summary['position_threshold_m']:.1f} m window RMSE",
        f"Acceleration threshold: >{summary['acc_threshold']:.1f} window RMSE",
        "",
        "Z POSITION FORECAST METRICS",
    ]
    for method in POSITION_METHODS:
        item = summary["position_z_metrics"][method]
        lines.append(
            f"{method:20s} point_RMSE={item['point_rmse']:.3f} m  "
            f"mean_window={item['mean_window_rmse']:.3f} m  "
            f"p95={item['p95_window_rmse']:.3f} m  "
            f"p99={item['p99_window_rmse']:.3f} m  "
            f">threshold={item['failures_over_threshold']:,}/{item['windows']:,} "
            f"({item['failure_rate_pct']:.3f}%)"
        )
    lines.extend(["", "Z ACCELERATION FORECAST METRICS"])
    for method in ACCELERATION_METHODS:
        item = summary["acceleration_z_metrics"][method]
        lines.append(
            f"{method:20s} point_RMSE={item['point_rmse']:.3f}  "
            f"point_MAE={item['point_mae']:.3f}  "
            f"mean_window={item['mean_window_rmse']:.3f}  "
            f"p95={item['p95_window_rmse']:.3f}  "
            f">threshold={item['failures_over_threshold']:,}/{item['windows']:,} "
            f"({item['failure_rate_pct']:.3f}%)"
        )
    lines.extend(
        [
            "",
            "INTERPRETATION WARNING",
            "This is a real-flight proof-of-concept replay using vertical telemetry-derived references.",
            "It does not validate X/Y or full 3D trajectory accuracy.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (args.output_dir / "summary.txt").write_text(report, encoding="utf-8")
    log("\n" + report)


def render_plots(args: argparse.Namespace, summary: dict, rows: list[dict]) -> None:
    if args.no_plots:
        return
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    position = summary["position_z_metrics"]
    acceleration = summary["acceleration_z_metrics"]
    comparison = [
        PLAIN_GRU_METHOD,
        GRU_RK4_METHOD,
        GRU_RK4_PHYS_METHOD,
        "Polynomial",
        "RK4 only",
        "Last acceleration",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(
        comparison,
        [position[method]["mean_window_rmse"] for method in comparison],
        color=[COLORS[method] for method in comparison],
    )
    axes[0].set_title("Real Flight Z Position Mean Window RMSE")
    axes[0].set_ylabel("Mean window RMSE (m)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(
        comparison,
        [position[method]["failure_rate_pct"] for method in comparison],
        color=[COLORS[method] for method in comparison],
    )
    axes[1].set_title("Z Position Threshold Failure Rate")
    axes[1].set_ylabel(f"Windows > {args.position_threshold:g} m (%)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "z_position_ablation_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        ACCELERATION_METHODS,
        [acceleration[method]["mean_window_rmse"] for method in ACCELERATION_METHODS],
        color=[COLORS[method] for method in ACCELERATION_METHODS],
    )
    ax.set_title("Real Flight Z Acceleration Mean Window RMSE")
    ax.set_ylabel("Mean window RMSE")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "z_acceleration_ablation_comparison.png", dpi=180)
    plt.close(fig)

    if rows:
        data = pd.DataFrame(rows)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for method in [GRU_RK4_PHYS_METHOD, PLAIN_GRU_METHOD, "Polynomial", "Last acceleration"]:
            axes[0].plot(
                data["start_time_s"],
                data[f"{method_key(method)}_position_z_window_rmse_m"],
                color=COLORS[method],
                linewidth=1.4,
                alpha=0.9,
                label=method,
            )
        axes[0].axhline(args.position_threshold, color="black", linestyle="--", linewidth=1)
        axes[0].set_title("Window Error Over Real Flight")
        axes[0].set_ylabel("Z position RMSE (m)")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        for method in [GRU_RK4_PHYS_METHOD, PLAIN_GRU_METHOD, "Last acceleration"]:
            axes[1].plot(
                data["start_time_s"],
                data[f"{method_key(method)}_acceleration_z_window_rmse"],
                color=COLORS[method],
                linewidth=1.4,
                alpha=0.9,
                label=method,
            )
        axes[1].axhline(args.acc_threshold, color="black", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Prediction start time (s)")
        axes[1].set_ylabel("Z acceleration RMSE")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "z_window_error_timeline.png", dpi=180)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    resolve_paths(args)

    for path in [
        args.flight,
        args.gru_model,
        args.gru_res_model,
        args.gru_res_phys_model,
        args.scaler_npz,
        args.parameters,
        args.thrust_curve,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    log(f"Writing real-flight Z ablation outputs to {args.output_dir}")
    log(f"Flight: {args.flight}")
    log(f"Parameters: {args.parameters}")
    log(f"Thrust curve: {args.thrust_curve}")
    log(f"Axis maps: acc={args.acc_axis_map}, gyro={args.gyro_axis_map}")

    start_time = time.time()
    summary, rows = evaluate(args)
    summary["runtime_seconds"] = time.time() - start_time
    write_outputs(args, summary, rows)
    render_plots(args, summary, rows)
    log(f"Finished. Open summary.txt in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
