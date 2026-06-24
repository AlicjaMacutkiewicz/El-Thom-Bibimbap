"""Real-flight Z-axis replay test for the GRU/RK4 trajectory model.

This is intentionally scoped to the data quality of the FAR OUT 2026 export:

* model input uses real IMU/gyro/barometer/temperature telemetry,
* acceleration is evaluated on the available vertical filtered acceleration,
* position is evaluated on the available vertical altitude/height reference,
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_gru import (  # noqa: E402 # type: ignore
    CONDITION_COLUMNS,
    INPUT_COLUMNS,
    SENSOR_COLUMNS,
    baseline_acceleration,
    compute_scalers,
    configure_imports,
    load_network,
    predict_normalized,
)


METHODS = ["GRU + RK4", "Polynomial", "RK4 only", "Last acceleration", "Oracle acceleration"]
ACC_METHODS = ["GRU + RK4", "RK4 only", "Last acceleration"]
COLORS = {
    "GRU + RK4": "#d62728",
    "Polynomial": "#1f77b4",
    "RK4 only": "#9467bd",
    "Last acceleration": "#ff7f0e",
    "Oracle acceleration": "#2ca02c",
}
Z_POSITION_COLUMN = "Position_Z"
Z_ACCELERATION_COLUMN = "Acceleration_Z"


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

    def summarize(self) -> dict[str, float | int]:
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


@dataclass(order=True)
class PlotWindow:
    score: float
    start: int = field(compare=False)
    start_time: float = field(compare=False)
    relative_time: np.ndarray = field(compare=False)
    actual_position: np.ndarray = field(compare=False)
    predicted_positions: dict[str, np.ndarray] = field(compare=False)
    actual_acceleration: np.ndarray = field(compare=False)
    predicted_accelerations: dict[str, np.ndarray] = field(compare=False)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    repo_default = Path.cwd()
    parser = argparse.ArgumentParser(
        description="Evaluate the current GRU/RK4 model on one real FAR OUT flight, Z axis only."
    )
    parser.add_argument("--repo", type=Path, default=repo_default)
    parser.add_argument(
        "--flight",
        type=Path,
        default=repo_default / "far_out_26_data" / "converted" / "flight_far_out_26.parquet",
        help="Converted real-flight parquet from convert_far_out_csv_to_parquet.py.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="GRU checkpoint. Defaults to prediction_models/GRU/src/best_gru_model_seq100.pth.",
    )
    parser.add_argument(
        "--scaler-npz",
        type=Path,
        default=None,
        help=(
            "Scaler file with mean_in/std_in/mean_xs/std_xs. "
            "Defaults to prediction_models/model_tests/test_results/reconstructed_scalers.npz if present."
        ),
    )
    parser.add_argument(
        "--scaler-data-dir",
        type=Path,
        default=None,
        help="Optional original training-data directory used to reconstruct scalers if --scaler-npz is absent.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to far_out_26_data/real_flight_z_eval_TIMESTAMP.",
    )
    parser.add_argument("--parameters", type=Path, default=None)
    parser.add_argument("--thrust-curve", type=Path, default=None)
    parser.add_argument(
        "--oxidizer-fraction",
        type=float,
        required=True,
        help="Launch oxidizer mass divided by the nominal oxidizer mass used by the model.",
    )
    parser.add_argument(
        "--pressure-scale",
        type=float,
        required=True,
        help="Launch-day chamber-pressure/thrust scale relative to nominal.",
    )
    parser.add_argument(
        "--rocket-mass-scale",
        type=float,
        required=True,
        help="Launch-day non-propellant rocket mass scale relative to the robustness baseline.",
    )
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--pred-len", type=int, default=100)
    parser.add_argument(
        "--horizons",
        default="25,50,100,150,200",
        help="Comma-separated rolling forecast lengths for the horizon sweep.",
    )
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
        default="X,Y,Z",
        help=(
            "Map source acceleration axes into model X,Y/Z. "
            "Examples: X,Y,Z or X,Z,-Y if the real vertical axis is -Y."
        ),
    )
    parser.add_argument(
        "--gyro-axis-map",
        default="X,Y,Z",
        help="Map source gyro axes into model X/Y/Z, with the same syntax as --acc-axis-map.",
    )
    parser.add_argument("--training-num-flights", type=int, default=1652)
    parser.add_argument("--training-start-flight", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=41)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--allow-incomplete-scaler-data", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    args.repo = args.repo.expanduser().resolve()
    args.flight = args.flight.expanduser().resolve()
    if args.model is None:
        args.model = args.repo / "prediction_models" / "GRU" / "src" / "best_gru_model_seq100.pth"
    else:
        args.model = args.model.expanduser().resolve()

    default_scaler = (
        args.repo / "prediction_models" / "model_tests" / "test_results" / "reconstructed_scalers.npz"
    )
    if args.scaler_npz is None and default_scaler.exists():
        args.scaler_npz = default_scaler
    elif args.scaler_npz is not None:
        args.scaler_npz = args.scaler_npz.expanduser().resolve()

    if args.parameters is None:
        args.parameters = (
            args.repo / "source_model" / "R7_SIMLE" / "R7_ROBUSTNESS" / "parameters.json"
        )
    else:
        args.parameters = args.parameters.expanduser().resolve()
    if args.thrust_curve is None:
        args.thrust_curve = (
            args.repo / "source_model" / "R7_SIMLE" / "R7_OUTPUT" / "thrust_source.csv"
        )
    else:
        args.thrust_curve = args.thrust_curve.expanduser().resolve()

    if args.scaler_data_dir is not None:
        args.scaler_data_dir = args.scaler_data_dir.expanduser().resolve()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = args.repo / "far_out_26_data" / f"real_flight_z_eval_{stamp}"
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 < args.oxidizer_fraction <= 1.0:
        raise ValueError("--oxidizer-fraction must be in (0, 1].")
    if args.pressure_scale <= 0.0 or args.rocket_mass_scale <= 0.0:
        raise ValueError("--pressure-scale and --rocket-mass-scale must be positive.")


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
    mapped_sensors = np.empty_like(source_inputs, dtype=np.float32)
    mapped_sensors[:, :3] = apply_axis_map(source_inputs[:, :3], acc_map)
    mapped_sensors[:, 3:6] = apply_axis_map(source_inputs[:, 3:6], gyro_map)
    mapped_sensors[:, 6:] = source_inputs[:, 6:]
    condition = np.array(
        [args.oxidizer_fraction, args.pressure_scale, args.rocket_mass_scale],
        dtype=np.float32,
    )
    conditions = np.broadcast_to(condition, (len(mapped_sensors), len(condition))).copy()
    inputs = np.concatenate([mapped_sensors, conditions], axis=1)

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
    inputs = inputs[finite]
    position_z = position_z[finite]
    acceleration_z = acceleration_z[finite]
    times = times[finite]

    return inputs, position_z, acceleration_z, times


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


def load_scalers(args: argparse.Namespace, calculate_x_b, parameters, thrust_curve, sampling_rate):
    if args.scaler_npz is not None and args.scaler_npz.exists():
        scalers = np.load(args.scaler_npz)
        metadata = {"source": str(args.scaler_npz), "reconstructed": False}
        result = (
            scalers["mean_in"].astype(np.float32),
            scalers["std_in"].astype(np.float32),
            scalers["mean_xs"].astype(np.float32),
            scalers["std_xs"].astype(np.float32),
            metadata,
        )
        if result[0].shape != (len(INPUT_COLUMNS),):
            raise ValueError(
                f"Scaler input schema has {result[0].size} columns, but this conditioned "
                f"evaluation requires {len(INPUT_COLUMNS)}: {INPUT_COLUMNS}"
            )
        return result
    if args.scaler_data_dir is None:
        raise FileNotFoundError(
            "No scaler file found. Pass --scaler-npz or --scaler-data-dir. "
            f"Default scaler path was: {args.scaler_npz}"
        )
    mean_in, std_in, _mean_acc, _std_acc, mean_xs, std_xs, metadata = compute_scalers(
        args, calculate_x_b, parameters, thrust_curve, sampling_rate
    )
    metadata["source"] = str(args.scaler_data_dir)
    metadata["reconstructed"] = True
    return mean_in, std_in, mean_xs, std_xs, metadata


def parse_horizons(text: str) -> list[int]:
    horizons = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("--horizons must contain positive integers.")
    return horizons


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
        "condition_context": inputs[previous, -len(CONDITION_COLUMNS) :],
    }


def predict_for_windows(
    args: argparse.Namespace,
    model,
    device: torch.device,
    calculate_x_b,
    parameters,
    thrust_curve,
    sampling_rate: float,
    mean_in: np.ndarray,
    std_in: np.ndarray,
    mean_xs: np.ndarray,
    std_xs: np.ndarray,
    windows: dict[str, np.ndarray],
    pred_len: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    residual_norm = predict_normalized(
        model,
        windows["input_windows"],
        mean_in,
        std_in,
        pred_len,
        args.batch_size,
        device,
        args.amp,
    )
    residual = residual_norm * std_xs + mean_xs
    base = baseline_acceleration(
        calculate_x_b,
        windows["future_times"].astype(np.float32),
        parameters,
        thrust_curve,
        sampling_rate,
        windows["condition_context"],
    )
    base_z = base[:, :, 2]
    hybrid_acc_z = residual[:, :, 2] + base_z
    last_acc_z = np.repeat(windows["previous_acceleration_z"][:, None], pred_len, axis=1)

    acceleration_predictions = {
        "GRU + RK4": hybrid_acc_z,
        "RK4 only": base_z,
        "Last acceleration": last_acc_z,
        "Oracle acceleration": windows["actual_acceleration_z"],
    }
    position_predictions = {
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
    return acceleration_predictions, position_predictions


def retain_worst(
    heap: list[PlotWindow],
    count: int,
    rmse: np.ndarray,
    windows: dict[str, np.ndarray],
    acceleration_predictions: dict[str, np.ndarray],
    position_predictions: dict[str, np.ndarray],
) -> None:
    candidate_count = min(count, len(rmse))
    candidate_indices = np.argpartition(rmse, -candidate_count)[-candidate_count:]
    for index in candidate_indices:
        item = PlotWindow(
            score=float(rmse[index]),
            start=int(windows["starts"][index]),
            start_time=float(windows["future_times"][index, 0]),
            relative_time=windows["future_times"][index] - windows["future_times"][index, 0],
            actual_position=windows["actual_position_z"][index].copy(),
            predicted_positions={
                method: prediction[index].copy() for method, prediction in position_predictions.items()
            },
            actual_acceleration=windows["actual_acceleration_z"][index].copy(),
            predicted_accelerations={
                method: prediction[index].copy()
                for method, prediction in acceleration_predictions.items()
                if method in ACC_METHODS
            },
        )
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item.score > heap[0].score:
            heapq.heapreplace(heap, item)


def evaluate_horizon(
    args: argparse.Namespace,
    model,
    device: torch.device,
    calculate_x_b,
    parameters,
    thrust_curve,
    sampling_rate: float,
    mean_in: np.ndarray,
    std_in: np.ndarray,
    mean_xs: np.ndarray,
    std_xs: np.ndarray,
    inputs: np.ndarray,
    position_z: np.ndarray,
    acceleration_z: np.ndarray,
    times: np.ndarray,
    pred_len: int,
    collect_rows: bool = False,
    collect_plots: bool = False,
) -> tuple[dict, list[dict], list[PlotWindow]]:
    windows = make_windows(
        inputs,
        position_z,
        acceleration_z,
        times,
        args.seq_len,
        pred_len,
        args.min_start_time,
        args.max_start_time,
    )
    acceleration_predictions, position_predictions = predict_for_windows(
        args,
        model,
        device,
        calculate_x_b,
        parameters,
        thrust_curve,
        sampling_rate,
        mean_in,
        std_in,
        mean_xs,
        std_xs,
        windows,
        pred_len,
    )

    position_metrics = {method: OneDimMetric() for method in METHODS}
    acceleration_metrics = {method: OneDimMetric() for method in ACC_METHODS}
    window_rmse_by_method: dict[str, np.ndarray] = {}
    acc_rmse_by_method: dict[str, np.ndarray] = {}
    for method in METHODS:
        window_rmse_by_method[method] = position_metrics[method].add(
            position_predictions[method],
            windows["actual_position_z"],
            args.position_threshold,
        )
    for method in ACC_METHODS:
        acc_rmse_by_method[method] = acceleration_metrics[method].add(
            acceleration_predictions[method],
            windows["actual_acceleration_z"],
            args.acc_threshold,
        )

    rows: list[dict] = []
    if collect_rows:
        for index, start in enumerate(windows["starts"]):
            row: dict[str, object] = {
                "start_sample_after_downsample": int(start),
                "start_time_s": float(windows["future_times"][index, 0]),
                "lead_time_s": float(windows["future_times"][index, -1] - windows["future_times"][index, 0]),
            }
            for method in METHODS:
                key = method.lower().replace(" ", "_").replace("+", "plus")
                row[f"{key}_position_z_window_rmse_m"] = float(window_rmse_by_method[method][index])
                row[f"{key}_position_z_endpoint_error_m"] = float(
                    position_predictions[method][index, -1]
                    - windows["actual_position_z"][index, -1]
                )
            for method in ACC_METHODS:
                key = method.lower().replace(" ", "_").replace("+", "plus")
                row[f"{key}_acceleration_z_window_rmse"] = float(acc_rmse_by_method[method][index])
            rows.append(row)

    worst: list[PlotWindow] = []
    if collect_plots:
        retain_worst(
            worst,
            6,
            window_rmse_by_method["GRU + RK4"],
            windows,
            acceleration_predictions,
            position_predictions,
        )

    lead_seconds = windows["future_times"][:, -1] - windows["future_times"][:, 0]
    summary = {
        "horizon_samples": int(pred_len),
        "lead_time_seconds_median": float(np.median(lead_seconds)),
        "lead_time_seconds_min": float(np.min(lead_seconds)),
        "lead_time_seconds_max": float(np.max(lead_seconds)),
        "windows": int(len(windows["starts"])),
        "start_time_min_s": float(windows["future_times"][:, 0].min()),
        "start_time_max_s": float(windows["future_times"][:, 0].max()),
        "position_z_metrics": {
            method: position_metrics[method].summarize() for method in METHODS
        },
        "acceleration_z_metrics": {
            method: acceleration_metrics[method].summarize() for method in ACC_METHODS
        },
    }
    return summary, rows, sorted(worst, reverse=True)


def write_summary(args: argparse.Namespace, summary: dict, rows: list[dict], worst: list[PlotWindow]) -> None:
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if rows:
        with (args.output_dir / "per_window_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with (args.output_dir / "worst_gru_z_windows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "start_sample_after_downsample", "start_time_s", "gru_position_z_rmse_m"])
        for rank, item in enumerate(worst, 1):
            writer.writerow([rank, item.start, item.start_time, item.score])

    main = summary["main_horizon"]
    lines = [
        "REAL FLIGHT REPLAY SUMMARY - Z AXIS ONLY",
        f"Flight: {summary['flight']}",
        f"Model: {summary['model']}",
        f"Launch conditions: {summary['launch_conditions']}",
        f"Rows after downsampling: {summary['rows_after_downsample']:,}",
        f"Downsample: {summary['downsample']}  dt_median={summary['dt_median_s']:.5f}s",
        f"Prediction window: {main['horizon_samples']} samples "
        f"(~{main['lead_time_seconds_median']:.2f}s)",
        f"Windows evaluated: {main['windows']:,}",
        f"Runtime: {summary['runtime_seconds']:.2f}s",
        f"Position threshold: >{summary['position_threshold_m']:.1f} m window RMSE",
        f"Acceleration threshold: >{summary['acc_threshold']:.1f} window RMSE",
        "",
        "Z POSITION FORECAST METRICS",
    ]
    for method in METHODS:
        item = main["position_z_metrics"][method]
        lines.append(
            f"{method:20s} point_RMSE={item['point_rmse']:.3f} m  "
            f"mean_window={item['mean_window_rmse']:.3f} m  "
            f"p95={item['p95_window_rmse']:.3f} m  "
            f"p99={item['p99_window_rmse']:.3f} m  "
            f">threshold={item['failures_over_threshold']:,}/{item['windows']:,} "
            f"({item['failure_rate_pct']:.3f}%)"
        )
    lines.extend(["", "Z ACCELERATION FORECAST METRICS"])
    for method in ACC_METHODS:
        item = main["acceleration_z_metrics"][method]
        lines.append(
            f"{method:20s} point_RMSE={item['point_rmse']:.3f}  "
            f"point_MAE={item['point_mae']:.3f}  "
            f"mean_window={item['mean_window_rmse']:.3f}  "
            f"p95={item['p95_window_rmse']:.3f}  "
            f">threshold={item['failures_over_threshold']:,}/{item['windows']:,} "
            f"({item['failure_rate_pct']:.3f}%)"
        )
    lines.extend(["", "HORIZON SWEEP - GRU + RK4 Z POSITION"])
    for horizon, report in summary["horizon_sweep"].items():
        item = report["position_z_metrics"]["GRU + RK4"]
        lines.append(
            f"{int(horizon):4d} samples (~{report['lead_time_seconds_median']:.2f}s)  "
            f"windows={report['windows']:,}  mean={item['mean_window_rmse']:.3f} m  "
            f"p95={item['p95_window_rmse']:.3f} m  "
            f">threshold={item['failures_over_threshold']}/{item['windows']} "
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


def render_plots(
    args: argparse.Namespace,
    summary: dict,
    rows: list[dict],
    worst: list[PlotWindow],
) -> None:
    if args.no_plots:
        return
    main = summary["main_horizon"]
    position = main["position_z_metrics"]
    acceleration = main["acceleration_z_metrics"]
    comparison = ["GRU + RK4", "Polynomial", "RK4 only", "Last acceleration"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(
        comparison,
        [position[method]["point_rmse"] for method in comparison],
        color=[COLORS[method] for method in comparison],
    )
    axes[0].set_title("Real Flight Z Position RMSE")
    axes[0].set_ylabel("Point RMSE (m)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(
        comparison,
        [position[method]["failure_rate_pct"] for method in comparison],
        color=[COLORS[method] for method in comparison],
    )
    axes[1].set_title("Z Position Threshold Failure Rate")
    axes[1].set_ylabel(f"Windows > {args.position_threshold:g} m (%)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "z_position_baseline_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    acc_methods = ACC_METHODS
    ax.bar(
        acc_methods,
        [acceleration[method]["point_rmse"] for method in acc_methods],
        color=[COLORS[method] for method in acc_methods],
    )
    ax.set_title("Real Flight Z Acceleration RMSE")
    ax.set_ylabel("Point RMSE")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "z_acceleration_baseline_comparison.png", dpi=180)
    plt.close(fig)

    horizons = [summary["horizon_sweep"][key] for key in sorted(summary["horizon_sweep"], key=int)]
    leads = [item["lead_time_seconds_median"] for item in horizons]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for method in comparison:
        axes[0].plot(
            leads,
            [item["position_z_metrics"][method]["mean_window_rmse"] for item in horizons],
            marker="o",
            color=COLORS[method],
            label=method,
        )
        axes[1].plot(
            leads,
            [item["position_z_metrics"][method]["p95_window_rmse"] for item in horizons],
            marker="o",
            color=COLORS[method],
            label=method,
        )
    axes[0].set_title("Mean Z Position Window RMSE by Horizon")
    axes[1].set_title("P95 Z Position Window RMSE by Horizon")
    for axis in axes:
        axis.set_xlabel("Forecast lead time (s)")
        axis.set_ylabel("RMSE (m)")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "z_error_by_horizon.png", dpi=180)
    plt.close(fig)

    if rows:
        data = pd.DataFrame(rows)
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        axes[0].plot(
            data["start_time_s"],
            data["gru_plus_rk4_position_z_window_rmse_m"],
            color=COLORS["GRU + RK4"],
            label="GRU + RK4",
        )
        axes[0].plot(
            data["start_time_s"],
            data["last_acceleration_position_z_window_rmse_m"],
            color=COLORS["Last acceleration"],
            alpha=0.75,
            label="Last acceleration",
        )
        axes[0].axhline(args.position_threshold, color="black", linestyle="--", linewidth=1)
        axes[0].set_ylabel("Z position RMSE (m)")
        axes[0].set_title("Window Error Over Real Flight")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(
            data["start_time_s"],
            data["gru_plus_rk4_acceleration_z_window_rmse"],
            color=COLORS["GRU + RK4"],
            label="GRU + RK4",
        )
        axes[1].plot(
            data["start_time_s"],
            data["last_acceleration_acceleration_z_window_rmse"],
            color=COLORS["Last acceleration"],
            alpha=0.75,
            label="Last acceleration",
        )
        axes[1].axhline(args.acc_threshold, color="black", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Prediction start time (s)")
        axes[1].set_ylabel("Z acceleration RMSE")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "z_window_error_timeline.png", dpi=180)
        plt.close(fig)

    render_window_examples(args, worst, "worst_z_windows.png", "Worst GRU Z Position Windows")
    render_acceleration_examples(
        args,
        worst,
        "worst_z_acceleration_windows.png",
        "Z Acceleration in Worst Position Windows",
    )


def render_window_examples(args: argparse.Namespace, windows: list[PlotWindow], filename: str, title: str) -> None:
    if not windows:
        return
    count = min(6, len(windows))
    cols = 2
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows), squeeze=False)
    for axis, item in zip(axes.flat, windows[:count], strict=False):
        axis.plot(item.relative_time, item.actual_position, color="black", linewidth=2, label="Reference Z")
        for method in ["GRU + RK4", "Polynomial", "RK4 only", "Last acceleration"]:
            axis.plot(
                item.relative_time,
                item.predicted_positions[method],
                color=COLORS[method],
                linewidth=1.5,
                label=method,
            )
        axis.set_title(f"t0={item.start_time:.2f}s, GRU RMSE={item.score:.2f}m")
        axis.set_xlabel("Horizon time (s)")
        axis.set_ylabel("Z position / altitude (m)")
        axis.grid(alpha=0.3)
    for axis in axes.flat[count:]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.output_dir / filename, dpi=180)
    plt.close(fig)


def render_acceleration_examples(
    args: argparse.Namespace, windows: list[PlotWindow], filename: str, title: str
) -> None:
    if not windows:
        return
    count = min(6, len(windows))
    cols = 2
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows), squeeze=False)
    for axis, item in zip(axes.flat, windows[:count], strict=False):
        axis.plot(
            item.relative_time,
            item.actual_acceleration,
            color="black",
            linewidth=2,
            label="Reference Z acceleration",
        )
        for method in ACC_METHODS:
            axis.plot(
                item.relative_time,
                item.predicted_accelerations[method],
                color=COLORS[method],
                linewidth=1.5,
                label=method,
            )
        axis.set_title(f"t0={item.start_time:.2f}s")
        axis.set_xlabel("Horizon time (s)")
        axis.set_ylabel("Z acceleration")
        axis.grid(alpha=0.3)
    for axis in axes.flat[count:]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.output_dir / filename, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    resolve_paths(args)
    if not args.flight.exists():
        raise FileNotFoundError(f"Converted real-flight parquet not found: {args.flight}")
    if not args.model.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.model}")

    log(f"Writing real-flight Z replay outputs to {args.output_dir}")
    log(f"Flight: {args.flight}")
    log(f"Model: {args.model}")
    log(f"Downsample: {args.downsample}")
    log(f"Axis maps: acc={args.acc_axis_map}, gyro={args.gyro_axis_map}")
    log(
        "Launch conditions: "
        f"oxidizer_fraction={args.oxidizer_fraction:.6f}, "
        f"pressure_scale={args.pressure_scale:.6f}, "
        f"rocket_mass_scale={args.rocket_mass_scale:.6f}"
    )

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
    log(
        f"Real flight rows after downsample: {len(times):,}; "
        f"time {times[0]:.3f}s..{times[-1]:.3f}s; median dt={np.median(dt):.5f}s"
    )

    device, gpu_ids = select_device(args)
    mean_in, std_in, mean_xs, std_xs, scaler_meta = load_scalers(
        args, calculate_x_b, parameters, thrust_curve, sampling_rate
    )
    if inputs.shape[1] != len(mean_in):
        raise RuntimeError(
            f"Real-flight input has {inputs.shape[1]} columns but the scaler has {len(mean_in)}."
        )
    model = load_network(GRU, args, device, gpu_ids, input_size=len(mean_in))

    start_time = time.time()
    main_summary, rows, worst = evaluate_horizon(
        args,
        model,
        device,
        calculate_x_b,
        parameters,
        thrust_curve,
        sampling_rate,
        mean_in,
        std_in,
        mean_xs,
        std_xs,
        inputs,
        position_z,
        acceleration_z,
        times,
        args.pred_len,
        collect_rows=True,
        collect_plots=True,
    )

    horizon_sweep: dict[str, dict] = {}
    for horizon in parse_horizons(args.horizons):
        horizon_summary, _horizon_rows, _horizon_worst = evaluate_horizon(
            args,
            model,
            device,
            calculate_x_b,
            parameters,
            thrust_curve,
            sampling_rate,
            mean_in,
            std_in,
            mean_xs,
            std_xs,
            inputs,
            position_z,
            acceleration_z,
            times,
            horizon,
            collect_rows=False,
            collect_plots=False,
        )
        horizon_sweep[str(horizon)] = horizon_summary
        gru_item = horizon_summary["position_z_metrics"]["GRU + RK4"]
        log(
            f"Horizon {horizon:4d} samples (~{horizon_summary['lead_time_seconds_median']:.2f}s): "
            f"GRU mean Z RMSE={gru_item['mean_window_rmse']:.3f} m, "
            f"p95={gru_item['p95_window_rmse']:.3f} m"
        )

    summary = {
        "flight": str(args.flight),
        "model": str(args.model),
        "parameters": str(args.parameters),
        "thrust_curve": str(args.thrust_curve),
        "output_dir": str(args.output_dir),
        "rows_after_downsample": int(len(times)),
        "time_start_s": float(times[0]),
        "time_end_s": float(times[-1]),
        "dt_median_s": float(np.median(dt)),
        "downsample": args.downsample,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "position_threshold_m": args.position_threshold,
        "acc_threshold": args.acc_threshold,
        "axis_maps": {"acc": args.acc_axis_map, "gyro": args.gyro_axis_map},
        "launch_conditions": {
            "Scenario_Oxidizer_Fraction": args.oxidizer_fraction,
            "Scenario_Pressure_Scale": args.pressure_scale,
            "Scenario_Rocket_Mass_Scale": args.rocket_mass_scale,
        },
        "device": str(device),
        "scaler_metadata": scaler_meta,
        "runtime_seconds": time.time() - start_time,
        "main_horizon": main_summary,
        "horizon_sweep": horizon_sweep,
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

    write_summary(args, summary, rows, worst)
    render_plots(args, summary, rows, worst)
    log(f"Finished. Open summary.txt and PNGs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
