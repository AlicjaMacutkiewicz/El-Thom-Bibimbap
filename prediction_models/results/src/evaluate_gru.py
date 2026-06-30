#!/usr/bin/env python3
"""Exhaustive held-out evaluation for the hybrid GRU/RK4 rocket model.

This file is intentionally standalone: place it anywhere outside the repository and
point it at the repository, the original training-data directory (for scalers), and
an external held-out directory. It never writes into the repository or data folders.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import random
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
CONDITION_COLUMNS = [
    "Scenario_Oxidizer_Fraction",
    "Scenario_Pressure_Scale",
    "Scenario_Rocket_Mass_Scale",
]
CONDITION_ALIASES = {
    "Scenario_Oxidizer_Fraction": "Scenario_Propellant_Fraction",
}
INPUT_COLUMNS = SENSOR_COLUMNS + CONDITION_COLUMNS
TARGET_COLUMNS = ["Acceleration_X", "Acceleration_Y", "Acceleration_Z"]
POSITION_COLUMNS = ["Position_X", "Position_Y", "Position_Z"]
PLAIN_GRU_METHOD = "Plain GRU"
GRU_RK4_METHOD = "GRU-RK4"
GRU_RK4_PHYS_METHOD = "GRU-RK4 + physics"
GRU_RK4_PHYS_GATE_METHOD = "Gated GRU-RK4 + physics"
GRU_RK4_PHYS_GATE_SMOOTH_METHOD = "Gated GRU-RK4 + physics (smooth)"
LAST_ACC_GRU_METHOD = "Last-acc GRU"
BASELINE_METHODS = ["Polynomial", "RK4 only", "Last acceleration", "Oracle acceleration"]
COLORS = {
    PLAIN_GRU_METHOD: "#17becf",
    GRU_RK4_METHOD: "#d62728",
    GRU_RK4_PHYS_METHOD: "#8c564b",
    GRU_RK4_PHYS_GATE_METHOD: "#e377c2",
    GRU_RK4_PHYS_GATE_SMOOTH_METHOD: "#ff9896",
    LAST_ACC_GRU_METHOD: "#7f7f7f",
    "Polynomial": "#1f77b4",
    "RK4 only": "#9467bd",
    "Last acceleration": "#ff7f0e",
    "Oracle acceleration": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(
        description="GPU-batched exhaustive external test for GRU ablations and baselines."
    )
    parser.add_argument("--repo", type=Path, default=home / "Rocket-trajectory-predictor")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Backward-compatible alias for --gru-res-phys-model. "
            "Prefer the explicit model flags for ablation runs."
        ),
    )
    parser.add_argument(
        "--gru-model",
        type=Path,
        default=None,
        help="Plain GRU direct-acceleration checkpoint. Defaults to prediction_models/GRU/src/gru_model.pth.",
    )
    parser.add_argument(
        "--gru-res-model",
        type=Path,
        default=None,
        help="Residual GRU checkpoint without physics-position loss. Defaults to prediction_models/GRU/src/gru_res_model.pth.",
    )
    parser.add_argument(
        "--gru-res-phys-model",
        type=Path,
        default=None,
        help="Residual GRU checkpoint with physics-position loss. Defaults to prediction_models/GRU/src/gru_res_phys_model.pth.",
    )
    parser.add_argument(
        "--gru-res-phys-gate-model",
        type=Path,
        default=None,
        help="Persistence-gated residual GRU checkpoint with trajectory-consistency loss.",
    )
    parser.add_argument(
        "--gru-res-phys-gate-smooth-model",
        type=Path,
        default=None,
        help="Persistence-gated residual GRU checkpoint trained with gate-smoothing loss.",
    )
    parser.add_argument(
        "--last-acc-gru-model",
        type=Path,
        default=None,
        help="Last-acceleration residual GRU checkpoint with a learned correction gate.",
    )
    parser.add_argument(
        "--scaler-data-dir",
        type=Path,
        default=home / "data",
        help="Original training run data. Used only to recreate input/residual normalizers.",
    )
    parser.add_argument(
        "--eval-data-dir",
        type=Path,
        default=home / "testing-data",
        help="External unseen test flight directory; every readable parquet is evaluated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Results output directory. Defaults to ~/gru_ablation_eval_TIMESTAMP.",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=None,
        help=(
            "Rocket parameters JSON for the RK4 baseline. Defaults to "
            "source_data/R7_SIMLE/R7_ROBUSTNESS/parameters.json."
        ),
    )
    parser.add_argument(
        "--thrust-curve",
        type=Path,
        default=None,
        help=(
            "Thrust curve CSV for the RK4 baseline. Defaults to "
            "source_data/R7_SIMLE/R7_OUTPUT/thrust_source.csv."
        ),
    )
    parser.add_argument("--training-num-flights", type=int, default=1652)
    parser.add_argument("--training-start-flight", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=41)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--downsample", type=int, default=25)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--pred-len", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument(
        "--landing-horizons",
        default="100,200,400,800,1600",
        help=(
            "Comma-separated prediction lengths for terminal landing-point forecasts. "
            "Each forecast ends at the final recorded position of each flight."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="auto uses CUDA if available; CPU is useful only for a smoke test.",
    )
    parser.add_argument(
        "--gpu-ids",
        default="all",
        help="CUDA device indices, e.g. '0,1', or 'all'. Both visible GPUs are used by default.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA mixed precision if an initial FP32 run appears too slow.",
    )
    parser.add_argument(
        "--max-eval-flights",
        type=int,
        default=0,
        help="Only for smoke tests: evaluate the first N external flights; 0 means exhaustive.",
    )
    parser.add_argument(
        "--allow-incomplete-scaler-data",
        action="store_true",
        help="Proceed when scaler-data-dir has fewer than training-num-flights files.",
    )
    parser.add_argument(
        "--strict-models",
        action="store_true",
        help="Fail if any configured neural checkpoint is missing instead of skipping missing checkpoints.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG rendering.")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def resolve_paths(args: argparse.Namespace) -> None:
    args.repo = args.repo.expanduser().resolve()
    args.scaler_data_dir = args.scaler_data_dir.expanduser().resolve()
    args.eval_data_dir = args.eval_data_dir.expanduser().resolve()
    if args.parameters is not None:
        args.parameters = args.parameters.expanduser().resolve()
    if args.thrust_curve is not None:
        args.thrust_curve = args.thrust_curve.expanduser().resolve()
    model_root = args.repo / "prediction_models" / "GRU" / "src"
    if args.gru_model is None:
        args.gru_model = model_root / "gru_model.pth"
    else:
        args.gru_model = args.gru_model.expanduser().resolve()
    if args.gru_res_model is None:
        args.gru_res_model = model_root / "gru_res_model.pth"
    else:
        args.gru_res_model = args.gru_res_model.expanduser().resolve()
    if args.gru_res_phys_model is None:
        args.gru_res_phys_model = args.model or model_root / "gru_res_phys_model.pth"
    args.gru_res_phys_model = args.gru_res_phys_model.expanduser().resolve()
    if args.gru_res_phys_gate_model is not None:
        args.gru_res_phys_gate_model = args.gru_res_phys_gate_model.expanduser().resolve()
    if args.gru_res_phys_gate_smooth_model is not None:
        args.gru_res_phys_gate_smooth_model = (
            args.gru_res_phys_gate_smooth_model.expanduser().resolve()
        )
    if args.last_acc_gru_model is not None:
        args.last_acc_gru_model = args.last_acc_gru_model.expanduser().resolve()
    args.model = args.gru_res_phys_model
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path.home() / f"gru_ablation_eval_{stamp}"
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)


def find_training_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("flight_*.parquet"))
    if files:
        return files
    files = sorted(directory.rglob("flight_*.parquet"))
    if files:
        log("WARNING: scaler flights were found recursively; verify this matches training file order.")
    return files


def find_eval_files(directory: Path) -> list[Path]:
    files = sorted(directory.rglob("flight_*.parquet"))
    if not files:
        files = sorted(directory.rglob("*.parquet"))
    return files


def read_flight(
    path: Path, downsample: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_parquet(path)
    required = set(SENSOR_COLUMNS + TARGET_COLUMNS + POSITION_COLUMNS)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    oxidizer_column = CONDITION_COLUMNS[0]
    legacy_column = CONDITION_ALIASES[oxidizer_column]
    if oxidizer_column not in data.columns:
        if legacy_column in data.columns:
            legacy_values = data[legacy_column].to_numpy(dtype=np.float32)
            if not np.allclose(legacy_values, 1.0):
                raise ValueError(
                    f"legacy variable {legacy_column} cannot be interpreted as "
                    f"{oxidizer_column}; regenerate this dataset with the oxidizer-aware scenario"
                )
            data[oxidizer_column] = legacy_values
        else:
            data[oxidizer_column] = 1.0
    for column in CONDITION_COLUMNS[1:]:
        if column not in data.columns:
            data[column] = 1.0
    inputs = data[INPUT_COLUMNS].to_numpy(dtype=np.float32)[::downsample]
    conditions = data[CONDITION_COLUMNS].to_numpy(dtype=np.float32)[::downsample]
    targets = data[TARGET_COLUMNS].to_numpy(dtype=np.float32)[::downsample]
    positions = data[POSITION_COLUMNS].to_numpy(dtype=np.float32)[::downsample]
    if "Time" in data.columns:
        times = data["Time"].to_numpy(dtype=np.float32)[::downsample]
    else:
        times = data.index.to_numpy(dtype=np.float32)[::downsample]
    return inputs, conditions, targets, positions, times


def to_numpy(value: object) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class Metric:
    squared_sum: float = 0.0
    point_count: int = 0
    window_rmse: list[np.ndarray] = field(default_factory=list)
    failures: int = 0

    def add(self, prediction: np.ndarray, truth: np.ndarray, threshold: float) -> np.ndarray:
        error = prediction - truth
        window = np.sqrt(np.mean(np.square(error), axis=1))
        self.squared_sum += float(np.square(error).sum())
        self.point_count += int(error.size)
        self.window_rmse.append(window.astype(np.float32, copy=False))
        self.failures += int((window > threshold).sum())
        return window

    def add_vector_distance(
        self, prediction: np.ndarray, truth: np.ndarray, threshold: float
    ) -> np.ndarray:
        error = prediction - truth
        squared_distance = np.square(error).sum(axis=-1)
        window = np.sqrt(np.mean(squared_distance, axis=1))
        self.squared_sum += float(squared_distance.sum())
        self.point_count += int(squared_distance.size)
        self.window_rmse.append(window.astype(np.float32, copy=False))
        self.failures += int((window > threshold).sum())
        return window

    def summarize(self) -> dict[str, float | int]:
        windows = np.concatenate(self.window_rmse) if self.window_rmse else np.array([], dtype=float)
        return {
            "point_rmse_m": float(np.sqrt(self.squared_sum / self.point_count)),
            "mean_window_rmse_m": float(windows.mean()),
            "median_window_rmse_m": float(np.median(windows)),
            "p95_window_rmse_m": float(np.quantile(windows, 0.95)),
            "p99_window_rmse_m": float(np.quantile(windows, 0.99)),
            "max_window_rmse_m": float(windows.max()),
            "failures_over_threshold": self.failures,
            "windows": int(windows.size),
            "failure_rate_pct": float(100.0 * self.failures / windows.size),
        }


@dataclass
class AccMetric:
    squared_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    absolute_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    count: int = 0

    def add(self, prediction: np.ndarray, truth: np.ndarray) -> None:
        error = prediction - truth
        self.squared_sum += np.square(error).sum(axis=(0, 1))
        self.absolute_sum += np.abs(error).sum(axis=(0, 1))
        self.count += int(error.shape[0] * error.shape[1])

    def summarize(self) -> dict[str, float]:
        component_rmse = np.sqrt(self.squared_sum / self.count)
        return {
            "x_rmse": float(component_rmse[0]),
            "y_rmse": float(component_rmse[1]),
            "z_rmse": float(component_rmse[2]),
            "vector_rmse": float(np.sqrt(self.squared_sum.sum() / self.count)),
            "x_mae": float(self.absolute_sum[0] / self.count),
            "y_mae": float(self.absolute_sum[1] / self.count),
            "z_mae": float(self.absolute_sum[2] / self.count),
        }


@dataclass
class PlotWindow:
    score: float
    flight: str
    start: int
    relative_time: np.ndarray
    actual: np.ndarray
    predictions: dict[str, np.ndarray]

    def __lt__(self, other: "PlotWindow") -> bool:  # noqa: UP037
        return self.score < other.score


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    output_mode: str  # "direct" predicts total acceleration, "residual" predicts x_total - x_b


class PredictionOnly(torch.nn.Module):
    """Hide the recurrent state from DataParallel's batch-dimension gather."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        prediction, _hidden = self.model(*args, **kwargs)
        return prediction


def method_key(method: str) -> str:
    key = "".join(character.lower() if character.isalnum() else "_" for character in method)
    return "_".join(part for part in key.split("_") if part)


def configured_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    candidates = [
        ModelSpec(PLAIN_GRU_METHOD, args.gru_model, "direct"),
        ModelSpec(GRU_RK4_METHOD, args.gru_res_model, "residual"),
        ModelSpec(GRU_RK4_PHYS_METHOD, args.gru_res_phys_model, "residual"),
    ]
    if args.gru_res_phys_gate_model is not None:
        candidates.append(
            ModelSpec(
                GRU_RK4_PHYS_GATE_METHOD,
                args.gru_res_phys_gate_model,
                "persistence_gated_residual",
            )
        )
    if args.gru_res_phys_gate_smooth_model is not None:
        candidates.append(
            ModelSpec(
                GRU_RK4_PHYS_GATE_SMOOTH_METHOD,
                args.gru_res_phys_gate_smooth_model,
                "persistence_gated_residual",
            )
        )
    if args.last_acc_gru_model is not None:
        candidates.append(
            ModelSpec(
                LAST_ACC_GRU_METHOD,
                args.last_acc_gru_model,
                "last_acc_gated_delta",
            )
        )
    specs: list[ModelSpec] = []
    missing: list[ModelSpec] = []
    for spec in candidates:
        if spec.path.exists():
            specs.append(spec)
        else:
            missing.append(spec)
    if missing:
        message = "; ".join(f"{spec.name}: {spec.path}" for spec in missing)
        if args.strict_models:
            raise FileNotFoundError(f"Configured checkpoint(s) missing: {message}")
        log(f"WARNING: skipping missing neural checkpoint(s): {message}")
    if not specs:
        raise FileNotFoundError(
            "No neural checkpoints were found. Provide at least one of "
            "--gru-model, --gru-res-model, or --gru-res-phys-model."
        )
    return specs


def method_order(model_specs: list[ModelSpec]) -> list[str]:
    return [spec.name for spec in model_specs] + BASELINE_METHODS


def acceleration_method_order(model_specs: list[ModelSpec]) -> list[str]:
    return [spec.name for spec in model_specs] + ["RK4 only", "Last acceleration"]


def primary_method(model_specs: list[ModelSpec]) -> str:
    for preferred in [
        LAST_ACC_GRU_METHOD,
        GRU_RK4_PHYS_GATE_SMOOTH_METHOD,
        GRU_RK4_PHYS_GATE_METHOD,
        GRU_RK4_PHYS_METHOD,
        GRU_RK4_METHOD,
        PLAIN_GRU_METHOD,
    ]:
        if any(spec.name == preferred for spec in model_specs):
            return preferred
    return model_specs[0].name


def configure_imports(repo: Path):
    gru_src = repo / "prediction_models" / "GRU" / "src"
    classical_src = repo / "prediction_models" / "classical_model" / "src"
    sys.path.insert(0, str(classical_src))
    sys.path.insert(0, str(gru_src))
    from GRU_model import GRU, PersistenceGatedGRU  # type: ignore
    from physics import (  # type: ignore
        calculate_x_b_conditioned,
        load_parameters,
        load_thrust_curve,
    )

    return GRU, PersistenceGatedGRU, calculate_x_b_conditioned, load_parameters, load_thrust_curve


def select_device(args: argparse.Namespace) -> tuple[torch.device, list[int]]:
    want_cuda = args.device in {"auto", "cuda"}
    if want_cuda and torch.cuda.is_available():
        if args.gpu_ids == "all":
            ids = list(range(torch.cuda.device_count()))
        else:
            ids = [int(part.strip()) for part in args.gpu_ids.split(",") if part.strip()]
        if not ids:
            raise ValueError("No GPU IDs selected.")
        device = torch.device(f"cuda:{ids[0]}")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        details = [
            f"cuda:{i} {torch.cuda.get_device_name(i)} "
            f"({torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GiB)"
            for i in ids
        ]
        log("CUDA devices: " + "; ".join(details))
        return device, ids
    if args.device == "cuda":
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    log("WARNING: evaluating on CPU. Use --device cuda on the server.")
    return torch.device("cpu"), []


def baseline_acceleration(
    calculate_x_b, times, parameters, thrust_curve, rate: float, conditions=None
) -> np.ndarray:
    output = calculate_x_b(
        torch.from_numpy(times.astype(np.float32)), parameters, thrust_curve, rate, conditions
    )
    return to_numpy(output).astype(np.float32, copy=False)


def compute_scalers(
    args: argparse.Namespace, calculate_x_b, parameters: dict, thrust_curve: np.ndarray, rate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    files = find_training_files(args.scaler_data_dir)
    required_end = args.training_start_flight + args.training_num_flights
    if len(files) < required_end and not args.allow_incomplete_scaler_data:
        raise RuntimeError(
            f"Need at least {required_end} scaler-data flight files for an exact replay, found {len(files)}. "
            "Point --scaler-data-dir at the original training files or pass --allow-incomplete-scaler-data "
            "only for a provisional check."
        )
    selected = files[args.training_start_flight : min(required_end, len(files))]
    if len(selected) != args.training_num_flights:
        log(
            f"WARNING: reconstructing approximate scalers from {len(selected)} instead of "
            f"{args.training_num_flights} original flights."
        )
    indices = list(range(len(selected)))
    random.seed(args.split_seed)
    random.shuffle(indices)
    train_indices = indices[: int(len(selected) * args.train_ratio)]
    input_arrays: list[np.ndarray] = []
    target_arrays: list[np.ndarray] = []
    residual_arrays: list[np.ndarray] = []
    broken: list[dict[str, str]] = []
    scaler_start = time.time()
    for progress, index in enumerate(train_indices, 1):
        path = selected[index]
        try:
            inputs, conditions, targets, _positions, times = read_flight(path, args.downsample)
        except Exception as exc:
            broken.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        base = baseline_acceleration(calculate_x_b, times, parameters, thrust_curve, rate, conditions)
        input_arrays.append(inputs)
        target_arrays.append(targets)
        residual_arrays.append(targets - base)
        if progress % 250 == 0:
            log(f"Scaler reconstruction: {progress:,}/{len(train_indices):,} training flights read")
    if broken and not args.allow_incomplete_scaler_data:
        raise RuntimeError(
            f"{len(broken)} original training files could not be read; exact scalers cannot be recreated. "
            "Inspect scaler_file_errors.json or pass --allow-incomplete-scaler-data for a provisional check."
        )
    if not input_arrays:
        raise RuntimeError("No usable training flights were available for scaler reconstruction.")
    all_inputs = np.concatenate(input_arrays, axis=0)
    all_targets = np.concatenate(target_arrays, axis=0)
    all_residuals = np.concatenate(residual_arrays, axis=0)
    mean_in = all_inputs.mean(axis=0)
    std_in = np.where(all_inputs.std(axis=0) == 0, 1e-6, all_inputs.std(axis=0))
    mean_acc = all_targets.mean(axis=0)
    std_acc = np.where(all_targets.std(axis=0) == 0, 1e-6, all_targets.std(axis=0))
    mean_xs = all_residuals.mean(axis=0)
    std_xs = np.where(all_residuals.std(axis=0) == 0, 1e-6, all_residuals.std(axis=0))
    metadata = {
        "scaler_data_dir": str(args.scaler_data_dir),
        "available_flight_files": len(files),
        "selected_flight_files": len(selected),
        "train_flights_used": len(input_arrays),
        "training_num_flights_expected": args.training_num_flights,
        "train_samples": int(len(all_inputs)),  # noqa: RUF046
        "split_seed": args.split_seed,
        "train_ratio": args.train_ratio,
        "broken_files": broken,
        "runtime_seconds": time.time() - scaler_start,
        "exact_expected_file_count_present": len(selected) == args.training_num_flights and not broken,
        "input_columns": INPUT_COLUMNS,
        "condition_columns": CONDITION_COLUMNS,
    }
    np.savez(
        args.output_dir / "reconstructed_scalers.npz",
        mean_in=mean_in,
        std_in=std_in,
        mean_acc=mean_acc,
        std_acc=std_acc,
        mean_xs=mean_xs,
        std_xs=std_xs,
    )
    with (args.output_dir / "scaler_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    log(
        f"Scalers ready from {len(input_arrays):,} training flights / {len(all_inputs):,} samples "
        f"in {metadata['runtime_seconds']:.1f}s"
    )
    log(f"Total acceleration scaler mean={np.round(mean_acc, 4)} std={np.round(std_acc, 4)}")
    log(f"Residual scaler mean={np.round(mean_xs, 4)} std={np.round(std_xs, 4)}")
    return mean_in, std_in, mean_acc, std_acc, mean_xs, std_xs, metadata


def _checkpoint_input_size(state: dict[str, torch.Tensor]) -> int | None:
    # The project uses a custom GRUCell whose input matrices have shape
    # (input_features + mode_flag, hidden_size). Keep support for a native
    # torch.nn.GRU checkpoint as well.
    weight = state.get("layers.0.W_r")
    if weight is not None:
        return int(weight.shape[0]) - 1
    weight = state.get("gru.weight_ih_l0")
    if weight is None:
        return None
    return int(weight.shape[1]) - 1


def load_network(
    GRU,
    PersistenceGatedGRU,
    path_or_args,
    device: torch.device,
    gpu_ids: list[int],
    input_size: int | None = None,
):
    path = Path(path_or_args) if isinstance(path_or_args, (str, Path)) else Path(path_or_args.model)
    state = torch.load(path, map_location="cpu")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    checkpoint_input_size = _checkpoint_input_size(state)
    if input_size is None:
        input_size = checkpoint_input_size or len(INPUT_COLUMNS)
    if checkpoint_input_size is not None and checkpoint_input_size != input_size:
        raise RuntimeError(
            f"{path} expects {checkpoint_input_size} model input columns, but this evaluation "
            f"was configured with {input_size}. Recreate scalers/evaluation data for the same "
            "input schema as the checkpoint."
        )
    output_size = int(state["fc.weight"].shape[0])
    if output_size == 6:
        model_class = PersistenceGatedGRU
    elif output_size == 3:
        model_class = GRU
    else:
        raise RuntimeError(f"Unsupported checkpoint output size {output_size} in {path}.")
    model = model_class(
        input_size=input_size,
        hidden_size=64,
        output_size=output_size,
        num_layers=2,
        dropout=0.2,
    )
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(PredictionOnly(model), device_ids=gpu_ids)
        log(f"Model inference will use DataParallel across GPUs {gpu_ids}.")
    model.eval()
    return model


def load_networks(
    GRU,
    PersistenceGatedGRU,
    model_specs: list[ModelSpec],
    device: torch.device,
    gpu_ids: list[int],
    input_size: int | None = None,
) -> dict[str, torch.nn.Module]:
    models = {}
    for spec in model_specs:
        log(f"Loading {spec.name} from {spec.path}")
        models[spec.name] = load_network(
            GRU, PersistenceGatedGRU, spec.path, device, gpu_ids, input_size
        )
    return models


def integrate_position(
    acceleration: np.ndarray,
    future_times: np.ndarray,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    initial_time: np.ndarray,
) -> np.ndarray:
    positions = np.empty_like(acceleration, dtype=np.float32)
    position = initial_position.copy()
    velocity = initial_velocity.copy()
    previous_time = initial_time.copy()
    for step in range(acceleration.shape[1]):
        dt = np.maximum(future_times[:, step] - previous_time, 0.0)[:, None]
        current_acc = acceleration[:, step, :]
        position = position + velocity * dt + 0.5 * current_acc * dt * dt
        velocity = velocity + current_acc * dt
        positions[:, step, :] = position
        previous_time = future_times[:, step]
    return positions


def polynomial_prediction(
    lookback_times: np.ndarray,
    lookback_position: np.ndarray,
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
    outputs = []
    for axis in range(3):
        y = lookback_position[:, :, axis].astype(np.float64)
        rhs = np.stack(
            [y.sum(axis=1), (x * y).sum(axis=1), (np.square(x) * y).sum(axis=1)],
            axis=1,
        )[..., None]
        coefficients = np.linalg.solve(matrix, rhs)[..., 0]
        outputs.append(
            coefficients[:, 0, None]
            + coefficients[:, 1, None] * future
            + coefficients[:, 2, None] * np.square(future)
        )
    return np.stack(outputs, axis=-1).astype(np.float32)


def predict_normalized(
    model,
    inputs: np.ndarray,
    mean_in: np.ndarray,
    std_in: np.ndarray,
    pred_len: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    amp_enabled = amp and device.type == "cuda"
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            normalized = ((inputs[start : start + batch_size] - mean_in) / std_in).astype(np.float32)
            tensor = torch.from_numpy(normalized).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                result = model(tensor, pred_len=pred_len)
                prediction = result[0] if isinstance(result, tuple) else result
            outputs.append(prediction.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def predict_model_accelerations(
    model_specs: list[ModelSpec],
    models: dict[str, torch.nn.Module],
    input_windows: np.ndarray,
    base_acc: np.ndarray,
    mean_in: np.ndarray,
    std_in: np.ndarray,
    mean_acc: np.ndarray,
    std_acc: np.ndarray,
    mean_xs: np.ndarray,
    std_xs: np.ndarray,
    pred_len: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
    last_acceleration: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    predictions = {}
    for spec in model_specs:
        normalized = predict_normalized(
            models[spec.name],
            input_windows,
            mean_in,
            std_in,
            pred_len,
            batch_size,
            device,
            amp,
        )
        if spec.output_mode == "direct":
            predictions[spec.name] = normalized * std_acc + mean_acc
        elif spec.output_mode == "residual":
            predictions[spec.name] = base_acc + normalized * std_xs + mean_xs
        elif spec.output_mode == "persistence_gated_residual":
            if normalized.shape[-1] != 6:
                raise ValueError(f"{spec.name} must emit 3 residuals and 3 gate logits.")
            if last_acceleration is None:
                raise ValueError(f"{spec.name} requires the last observed acceleration.")
            residual = normalized[:, :, :3] * std_xs + mean_xs
            learned_candidate = base_acc + residual
            gate_logits = np.clip(normalized[:, :, 3:6], -30.0, 30.0)
            gate = 1.0 / (1.0 + np.exp(-gate_logits))
            anchor = np.repeat(last_acceleration[:, None, :], pred_len, axis=1)
            predictions[spec.name] = anchor + gate * (learned_candidate - anchor)
        elif spec.output_mode == "last_acc_gated_delta":
            if normalized.shape[-1] != 6:
                raise ValueError(f"{spec.name} must emit 3 acceleration deltas and 3 gate logits.")
            if last_acceleration is None:
                raise ValueError(f"{spec.name} requires the last observed acceleration.")
            delta = normalized[:, :, :3] * std_acc
            gate_logits = np.clip(normalized[:, :, 3:6], -30.0, 30.0)
            gate = 1.0 / (1.0 + np.exp(-gate_logits))
            anchor = np.repeat(last_acceleration[:, None, :], pred_len, axis=1)
            predictions[spec.name] = anchor + gate * delta
        else:
            raise ValueError(f"Unknown output mode for {spec.name}: {spec.output_mode}")
    return predictions


def retain_worst(
    heap: list[PlotWindow],
    count: int,
    rmse: np.ndarray,
    flight: Path,
    starts: np.ndarray,
    times: np.ndarray,
    actual_z: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    candidate_indices = np.argpartition(rmse, -min(count, len(rmse)))[-min(count, len(rmse)) :]
    for index in candidate_indices:
        rel_time = times[index] - times[index, 0]
        item = PlotWindow(
            score=float(rmse[index]),
            flight=flight.name,
            start=int(starts[index]),
            relative_time=rel_time.copy(),
            actual=actual_z[index].copy(),
            predictions={name: prediction[index].copy() for name, prediction in predictions.items()},
        )
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item.score > heap[0].score:
            heapq.heapreplace(heap, item)


def retain_illustrative(
    examples: dict[float, PlotWindow],
    targets: list[float],
    rmse: np.ndarray,
    flight: Path,
    starts: np.ndarray,
    times: np.ndarray,
    actual_z: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    for target in targets:
        index = int(np.argmin(np.abs(rmse - target)))
        score = float(rmse[index])
        current = examples.get(target)
        if current is not None and abs(current.score - target) <= abs(score - target):
            continue
        examples[target] = PlotWindow(
            score=score,
            flight=flight.name,
            start=int(starts[index]),
            relative_time=(times[index] - times[index, 0]).copy(),
            actual=actual_z[index].copy(),
            predictions={name: prediction[index].copy() for name, prediction in predictions.items()},
        )


def evaluate(
    args: argparse.Namespace,
    model_specs: list[ModelSpec],
    models: dict[str, torch.nn.Module],
    device: torch.device,
    calculate_x_b,
    parameters: dict,
    thrust_curve: np.ndarray,
    rate: float,
    mean_in: np.ndarray,
    std_in: np.ndarray,
    mean_acc: np.ndarray,
    std_acc: np.ndarray,
    mean_xs: np.ndarray,
    std_xs: np.ndarray,
) -> tuple[dict, list[dict], list[PlotWindow], list[PlotWindow]]:
    files = find_eval_files(args.eval_data_dir)
    if args.max_eval_flights:
        files = files[: args.max_eval_flights]
        log(f"SMOKE TEST MODE: limited to {len(files)} external flight files.")
    if not files:
        raise RuntimeError(f"No parquet test files found under {args.eval_data_dir}.")
    log(f"External evaluation files discovered: {len(files):,} under {args.eval_data_dir}")
    axes = ["X", "Y", "Z", "3D"]
    methods = method_order(model_specs)
    acc_methods = acceleration_method_order(model_specs)
    primary = primary_method(model_specs)
    metrics = {method: {axis: Metric() for axis in axes} for method in methods}
    acceleration = {method: AccMetric() for method in acc_methods}
    rows: list[dict] = []
    skipped: list[dict[str, str]] = []
    worst: list[PlotWindow] = []
    illustrative_targets = [0.25, 0.5, 1.0, args.threshold]
    illustrative: dict[float, PlotWindow] = {}
    total_windows = 0
    dt_medians: list[float] = []
    start_time = time.time()
    for sequence, path in enumerate(files, 1):
        try:
            inputs, conditions, targets, positions, times = read_flight(path, args.downsample)
        except Exception as exc:
            skipped.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            log(f"SKIP unreadable file {path.name}: {type(exc).__name__}: {exc}")
            continue
        starts = np.arange(args.seq_len, len(times) - args.pred_len)
        if not len(starts):
            skipped.append({"file": str(path), "error": "insufficient samples after downsampling"})
            continue
        if len(times) > 1:
            dt_medians.append(float(np.median(np.diff(times))))
        input_windows = np.stack([inputs[i - args.seq_len : i] for i in starts])
        condition_context = conditions[starts - 1]
        future_times = np.stack([times[i : i + args.pred_len] for i in starts])
        actual_acc = np.stack([targets[i : i + args.pred_len] for i in starts])
        actual_position = np.stack([positions[i : i + args.pred_len] for i in starts])
        base_acc = baseline_acceleration(
            calculate_x_b, future_times, parameters, thrust_curve, rate, condition_context
        )
        last_observed_acceleration = targets[starts - 1]
        model_accelerations = predict_model_accelerations(
            model_specs,
            models,
            input_windows,
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
            last_observed_acceleration,
        )
        last_acc = np.repeat(last_observed_acceleration[:, None, :], args.pred_len, axis=1)
        for method, prediction in model_accelerations.items():
            acceleration[method].add(prediction, actual_acc)
        acceleration["RK4 only"].add(base_acc, actual_acc)
        acceleration["Last acceleration"].add(last_acc, actual_acc)
        previous = starts - 1
        before_previous = starts - 2
        historical_dt = np.maximum(times[previous] - times[before_previous], 1e-6)
        initial_position = positions[previous]
        initial_velocity = (positions[previous] - positions[before_previous]) / historical_dt[:, None]
        initial_time = times[previous]
        predictions = {
            method: integrate_position(
                prediction, future_times, initial_position, initial_velocity, initial_time
            )
            for method, prediction in model_accelerations.items()
        }
        predictions.update(
            {
                "RK4 only": integrate_position(
                    base_acc, future_times, initial_position, initial_velocity, initial_time
                ),
                "Last acceleration": integrate_position(
                    last_acc, future_times, initial_position, initial_velocity, initial_time
                ),
                "Oracle acceleration": integrate_position(
                    actual_acc, future_times, initial_position, initial_velocity, initial_time
                ),
                "Polynomial": polynomial_prediction(
                    np.stack([times[i - args.seq_len : i] for i in starts]),
                    np.stack([positions[i - args.seq_len : i] for i in starts]),
                    future_times,
                ),
            }
        )
        flight_row: dict[str, object] = {"file": path.name, "windows": int(len(starts))}  # noqa: RUF046
        flight_window_rmse: dict[str, np.ndarray] = {}
        for method in methods:
            key = method_key(method)
            for axis_index, axis in enumerate(["X", "Y", "Z"]):
                axis_rmse = metrics[method][axis].add(
                    predictions[method][:, :, axis_index],
                    actual_position[:, :, axis_index],
                    args.threshold,
                )
                flight_row[f"{key}_{axis.lower()}_mean_window_rmse_m"] = float(axis_rmse.mean())
                flight_row[f"{key}_{axis.lower()}_max_window_rmse_m"] = float(axis_rmse.max())
                flight_row[f"{key}_{axis.lower()}_failures"] = int((axis_rmse > args.threshold).sum())
            current_rmse = metrics[method]["3D"].add_vector_distance(
                predictions[method], actual_position, args.threshold
            )
            flight_window_rmse[method] = current_rmse
            flight_row[f"{key}_3d_mean_window_rmse_m"] = float(current_rmse.mean())
            flight_row[f"{key}_3d_max_window_rmse_m"] = float(current_rmse.max())
            flight_row[f"{key}_3d_failures"] = int((current_rmse > args.threshold).sum())
        retain_worst(
            worst,
            6,
            flight_window_rmse[primary],
            path,
            starts,
            future_times,
            actual_position,
            predictions,
        )
        retain_illustrative(
            illustrative,
            illustrative_targets,
            flight_window_rmse[primary],
            path,
            starts,
            future_times,
            actual_position,
            predictions,
        )
        rows.append(flight_row)
        total_windows += len(starts)
        if sequence == 1 or sequence % 10 == 0 or sequence == len(files):
            elapsed = time.time() - start_time
            speed = total_windows / max(elapsed, 1e-9)
            estimated_total = total_windows / sequence * len(files)
            eta_minutes = max(estimated_total - total_windows, 0.0) / max(speed, 1e-9) / 60.0
            log(
                f"Evaluation: {sequence:,}/{len(files):,} files, {total_windows:,} windows, "
                f"{speed:,.0f} windows/s, ETA {eta_minutes:.1f} min"
            )
    if not rows:
        raise RuntimeError("No readable external test flights produced evaluation windows.")
    summary = {
        "models": [
            {"name": spec.name, "path": str(spec.path), "output_mode": spec.output_mode}
            for spec in model_specs
        ],
        "primary_method": primary,
        "neural_methods": [spec.name for spec in model_specs],
        "methods": methods,
        "eval_data_dir": str(args.eval_data_dir),
        "parameters": str(args.parameters_path_used),
        "thrust_curve": str(args.thrust_curve_path_used),
        "flight_files_discovered": len(files),
        "flights_evaluated": len(rows),
        "skipped_files": skipped,
        "windows_evaluated": total_windows,
        "threshold_m": args.threshold,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "downsample": args.downsample,
        "dt_seconds_median": float(np.median(dt_medians)) if dt_medians else None,
        "history_window_seconds_median": float(max(args.seq_len - 1, 0) * np.median(dt_medians))
        if dt_medians
        else None,
        "prediction_horizon_seconds_median": float(max(args.pred_len - 1, 0) * np.median(dt_medians))
        if dt_medians
        else None,
        "input_columns": INPUT_COLUMNS,
        "condition_columns": CONDITION_COLUMNS,
        "runtime_seconds": time.time() - start_time,
        "device": str(device),
        "cuda_devices_used": list(range(torch.cuda.device_count()))
        if device.type == "cuda" and args.gpu_ids == "all"
        else args.gpu_ids,
        "amp_enabled": bool(args.amp and device.type == "cuda"),
        "position_metrics": {
            method: {axis: metrics[method][axis].summarize() for axis in axes}
            for method in methods
        },
        "acceleration_metrics": {
            method: acceleration[method].summarize() for method in acceleration
        },
    }
    return summary, rows, sorted(worst, reverse=True), [illustrative[target] for target in illustrative_targets]


def parse_landing_horizons(args: argparse.Namespace) -> list[int]:
    horizons = sorted({int(value.strip()) for value in args.landing_horizons.split(",") if value.strip()})
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("--landing-horizons must contain positive comma-separated integers.")
    return horizons


def endpoint_summary(errors: np.ndarray, threshold: float) -> dict[str, float | int]:
    distance_3d = np.linalg.norm(errors, axis=1)
    distance_xy = np.linalg.norm(errors[:, :2], axis=1)
    rmse_axis = np.sqrt(np.mean(np.square(errors), axis=0))
    return {
        "x_endpoint_rmse_m": float(rmse_axis[0]),
        "y_endpoint_rmse_m": float(rmse_axis[1]),
        "z_endpoint_rmse_m": float(rmse_axis[2]),
        "final_difference_in_landing_rmse_3d_m": float(np.sqrt(np.mean(np.square(distance_3d)))),
        "final_difference_in_landing_median_3d_m": float(np.median(distance_3d)),
        "final_difference_in_landing_p95_3d_m": float(np.quantile(distance_3d, 0.95)),
        "final_difference_in_landing_max_3d_m": float(distance_3d.max()),
        "landing_ground_spot_rmse_xy_m": float(np.sqrt(np.mean(np.square(distance_xy)))),
        "landing_ground_spot_median_xy_m": float(np.median(distance_xy)),
        "landing_ground_spot_p95_xy_m": float(np.quantile(distance_xy, 0.95)),
        "landing_ground_spot_max_xy_m": float(distance_xy.max()),
        "landing_endpoint_failures_over_threshold": int((distance_3d > threshold).sum()),
        "landing_endpoint_failure_rate_pct": float(100.0 * np.mean(distance_3d > threshold)),
        "flights": int(len(errors)),  # noqa: RUF046
    }


def evaluate_landing_horizons(
    args: argparse.Namespace,
    model_specs: list[ModelSpec],
    models: dict[str, torch.nn.Module],
    device: torch.device,
    calculate_x_b,
    parameters: dict,
    thrust_curve: np.ndarray,
    rate: float,
    mean_in: np.ndarray,
    std_in: np.ndarray,
    mean_acc: np.ndarray,
    std_acc: np.ndarray,
    mean_xs: np.ndarray,
    std_xs: np.ndarray,
) -> tuple[dict[str, dict], list[dict]]:
    files = find_eval_files(args.eval_data_dir)
    if args.max_eval_flights:
        files = files[: args.max_eval_flights]
    usable: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for path in files:
        try:
            inputs, conditions, targets, positions, times = read_flight(path, args.downsample)
        except Exception:
            continue
        usable.append((path, inputs, conditions, targets, positions, times))
    reports: dict[str, dict] = {}
    rows: list[dict] = []
    methods = method_order(model_specs)
    primary = primary_method(model_specs)
    log("Running terminal landing-point horizon sweep...")
    for horizon in parse_landing_horizons(args):
        histories = []
        future_times = []
        actual_accelerations = []
        actual_positions = []
        initial_positions = []
        initial_velocities = []
        initial_times = []
        previous_accelerations = []
        condition_contexts = []
        lookback_times = []
        lookback_positions = []
        names = []
        lead_seconds = []
        for path, inputs, conditions, targets, positions, times in usable:
            start = len(times) - horizon
            if start < args.seq_len or start < 2:
                continue
            previous = start - 1
            before_previous = start - 2
            dt = max(float(times[previous] - times[before_previous]), 1e-6)
            histories.append(inputs[start - args.seq_len : start])
            future_times.append(times[start:])
            actual_accelerations.append(targets[start:])
            actual_positions.append(positions[start:])
            initial_positions.append(positions[previous])
            initial_velocities.append((positions[previous] - positions[before_previous]) / dt)
            initial_times.append(times[previous])
            previous_accelerations.append(targets[previous])
            condition_contexts.append(conditions[previous])
            lookback_times.append(times[start - args.seq_len : start])
            lookback_positions.append(positions[start - args.seq_len : start])
            names.append(path.name)
            lead_seconds.append(float(times[-1] - times[start]))
        if not names:
            reports[str(horizon)] = {"flights": 0, "note": "No flights long enough for this horizon."}
            continue
        histories_array = np.stack(histories)
        time_array = np.stack(future_times)
        actual_acc_array = np.stack(actual_accelerations)
        actual_pos_array = np.stack(actual_positions)
        initial_pos_array = np.stack(initial_positions)
        initial_vel_array = np.stack(initial_velocities)
        initial_time_array = np.asarray(initial_times, dtype=np.float32)
        condition_context_array = np.stack(condition_contexts)
        base = baseline_acceleration(
            calculate_x_b, time_array, parameters, thrust_curve, rate, condition_context_array
        )
        model_accelerations = predict_model_accelerations(
            model_specs,
            models,
            histories_array,
            base,
            mean_in,
            std_in,
            mean_acc,
            std_acc,
            mean_xs,
            std_xs,
            horizon,
            args.batch_size,
            device,
            args.amp,
            np.stack(previous_accelerations),
        )
        last_acc = np.repeat(np.stack(previous_accelerations)[:, None, :], horizon, axis=1)
        acceleration_predictions = {
            **model_accelerations,
            "RK4 only": base,
            "Last acceleration": last_acc,
            "Oracle acceleration": actual_acc_array,
        }
        position_predictions = {
            method: integrate_position(
                acceleration, time_array, initial_pos_array, initial_vel_array, initial_time_array
            )
            for method, acceleration in acceleration_predictions.items()
        }
        position_predictions["Polynomial"] = polynomial_prediction(
            np.stack(lookback_times), np.stack(lookback_positions), time_array
        )
        trajectory_metrics = {
            method: {axis: Metric() for axis in ["X", "Y", "Z", "3D"]} for method in methods
        }
        method_reports = {}
        endpoint_errors = {}
        for method in methods:
            prediction = position_predictions[method]
            for axis_index, axis in enumerate(["X", "Y", "Z"]):
                trajectory_metrics[method][axis].add(
                    prediction[:, :, axis_index], actual_pos_array[:, :, axis_index], args.threshold
                )
            trajectory_metrics[method]["3D"].add_vector_distance(
                prediction, actual_pos_array, args.threshold
            )
            endpoint_errors[method] = prediction[:, -1, :] - actual_pos_array[:, -1, :]
            method_reports[method] = {
                "trajectory_position_metrics": {
                    axis: trajectory_metrics[method][axis].summarize()
                    for axis in ["X", "Y", "Z", "3D"]
                },
                "landing_endpoint_metrics": endpoint_summary(endpoint_errors[method], args.threshold),
            }
        actual_terminal = actual_pos_array[:, -1, :]
        reports[str(horizon)] = {
            "horizon_samples": horizon,
            "lead_time_seconds_median": float(np.median(lead_seconds)),
            "lead_time_seconds_min": float(np.min(lead_seconds)),
            "lead_time_seconds_max": float(np.max(lead_seconds)),
            "flights": len(names),
            "actual_terminal_z_abs_median": float(np.median(np.abs(actual_terminal[:, 2]))),
            "actual_terminal_z_abs_max": float(np.max(np.abs(actual_terminal[:, 2]))),
            "methods": method_reports,
        }
        for index, name in enumerate(names):
            row: dict[str, object] = {
                "horizon_samples": horizon,
                "lead_time_seconds": lead_seconds[index],
                "file": name,
                "actual_landing_x": float(actual_terminal[index, 0]),
                "actual_landing_y": float(actual_terminal[index, 1]),
                "actual_landing_z": float(actual_terminal[index, 2]),
            }
            for method in methods:
                key = method_key(method)
                predicted_terminal = position_predictions[method][index, -1, :]
                error = endpoint_errors[method][index]
                row[f"{key}_predicted_landing_x"] = float(predicted_terminal[0])
                row[f"{key}_predicted_landing_y"] = float(predicted_terminal[1])
                row[f"{key}_predicted_landing_z"] = float(predicted_terminal[2])
                row[f"{key}_landing_error_x_m"] = float(error[0])
                row[f"{key}_landing_error_y_m"] = float(error[1])
                row[f"{key}_landing_error_z_m"] = float(error[2])
                row[f"{key}_landing_ground_spot_error_xy_m"] = float(np.linalg.norm(error[:2]))
                row[f"{key}_final_difference_in_landing_3d_m"] = float(np.linalg.norm(error))
            rows.append(row)
        gru_endpoint = method_reports[primary]["landing_endpoint_metrics"]
        log(
            f"Landing horizon {horizon:4d} samples (~{np.median(lead_seconds):.1f}s): "
            f"{len(names)} flights, {primary} final difference RMSE="
            f"{gru_endpoint['final_difference_in_landing_rmse_3d_m']:.3f} m, "
            f"> {args.threshold:g} m="
            f"{gru_endpoint['landing_endpoint_failures_over_threshold']}/{len(names)}"
        )
    return reports, rows


def write_outputs(
    args: argparse.Namespace,
    summary: dict,
    rows: list[dict],
    worst: list[PlotWindow],
    illustrative: list[PlotWindow],
    landing_rows: list[dict],
) -> None:
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (args.output_dir / "per_flight_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "worst_gru_windows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "file", "start_sample_after_downsample", "primary_window_rmse_m"])
        for rank, item in enumerate(worst, 1):
            writer.writerow([rank, item.flight, item.start, item.score])
    with (args.output_dir / "illustrative_gru_windows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_rmse_m", "file", "start_sample_after_downsample", "actual_gru_rmse_m"])
        for target, item in zip([0.25, 0.5, 1.0, args.threshold], illustrative, strict=False):
            writer.writerow([target, item.flight, item.start, item.score])
    if landing_rows:
        with (args.output_dir / "landing_horizon_per_flight.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(landing_rows[0].keys()))
            writer.writeheader()
            writer.writerows(landing_rows)
    lines = [
        "EXHAUSTIVE EXTERNAL TEST SUMMARY - XYZ POSITION AND ACCELERATION",
        "Models:",
        *(
            f"  {item['name']}: {item['path']} ({item['output_mode']})"
            for item in summary["models"]
        ),
        f"Primary neural method for examples: {summary['primary_method']}",
        f"External files evaluated: {summary['flights_evaluated']} "
        f"(skipped: {len(summary['skipped_files'])})",
        f"RK4 parameters: {summary['parameters']}",
        f"RK4 thrust curve: {summary['thrust_curve']}",
        f"Windows evaluated: {summary['windows_evaluated']:,}",
        f"History window: {summary['seq_len']} samples"
        + (
            f" (~{summary['history_window_seconds_median']:.2f}s median)"
            if summary.get("history_window_seconds_median") is not None
            else ""
        ),
        f"Prediction window: {summary['pred_len']} samples"
        + (
            f" (~{summary['prediction_horizon_seconds_median']:.2f}s median)"
            if summary.get("prediction_horizon_seconds_median") is not None
            else ""
        ),
        f"Runtime: {summary['runtime_seconds'] / 60:.2f} minutes",
        f"Failure threshold: >{summary['threshold_m']:.1f} m window RMSE",
        "",
        f"{summary['pred_len']}-STEP POSITION FORECAST METRICS - 3D TRAJECTORY DISTANCE",
    ]
    for method in summary["methods"]:
        item = summary["position_metrics"][method]["3D"]
        lines.append(
            f"{method:20s} point_RMSE={item['point_rmse_m']:.3f} m  "
            f"mean_window={item['mean_window_rmse_m']:.3f} m  "
            f"p99={item['p99_window_rmse_m']:.3f} m  "
            f">threshold={item['failures_over_threshold']:,}/{item['windows']:,} "
            f"({item['failure_rate_pct']:.4f}%)"
        )
    lines.append("")
    primary = summary["primary_method"]
    lines.append(f"{primary} POSITION BY AXIS")
    for axis in ["X", "Y", "Z", "3D"]:
        item = summary["position_metrics"][primary][axis]
        lines.append(
            f"{axis:3s} point_RMSE={item['point_rmse_m']:.3f} m  "
            f"p99_window={item['p99_window_rmse_m']:.3f} m  "
            f">threshold={item['failures_over_threshold']:,}/{item['windows']:,} "
            f"({item['failure_rate_pct']:.4f}%)"
        )
    lines.append("")
    lines.append("ACCELERATION METRICS BY AXIS")
    for method, item in summary["acceleration_metrics"].items():
        lines.append(
            f"{method:20s} X_RMSE={item['x_rmse']:.4f}  "
            f"Y_RMSE={item['y_rmse']:.4f}  Z_RMSE={item['z_rmse']:.4f}  "
            f"vector_RMSE={item['vector_rmse']:.4f}"
        )
    if summary.get("landing_horizon_metrics"):
        lines.append("")
        lines.append("FINAL DIFFERENCE IN LANDING - ENDPOINT 3D DISTANCE")
        lines.append("(Each horizon is a forecast ending at the recorded final position.)")
        for horizon, report_item in summary["landing_horizon_metrics"].items():
            if not report_item.get("flights"):
                lines.append(f"{horizon:>5s} samples: no eligible flights")
                continue
            item = report_item["methods"][primary]["landing_endpoint_metrics"]
            lines.append(
                f"{int(horizon):5d} samples (~{report_item['lead_time_seconds_median']:.1f}s)  "
                f"flights={report_item['flights']:3d}  "
                f"3D_RMSE={item['final_difference_in_landing_rmse_3d_m']:.3f} m  "
                f"XY_spot_RMSE={item['landing_ground_spot_rmse_xy_m']:.3f} m  "
                f"p95_3D={item['final_difference_in_landing_p95_3d_m']:.3f} m  "
                f">threshold={item['landing_endpoint_failures_over_threshold']}/{item['flights']}  "
                f"actual_terminal_|Z|_median={report_item['actual_terminal_z_abs_median']:.3f} m"
            )
    if summary["skipped_files"]:
        lines.append("")
        lines.append("SKIPPED FILES")
        lines.extend(f"{entry['file']}: {entry['error']}" for entry in summary["skipped_files"])
    report = "\n".join(lines) + "\n"
    (args.output_dir / "summary.txt").write_text(report, encoding="utf-8")
    log("\n" + report)


def render_plots(
    args: argparse.Namespace,
    summary: dict,
    rows: list[dict],
    worst: list[PlotWindow],
    illustrative: list[PlotWindow],
    landing_rows: list[dict],
) -> None:
    if args.no_plots:
        return
    primary = summary["primary_method"]
    forecast_label = f"{summary.get('pred_len', args.pred_len)}-step"
    comparison = [
        method
        for method in summary["neural_methods"] + ["Polynomial", "RK4 only", "Last acceleration"]
        if method in summary["position_metrics"]
    ]
    position = summary["position_metrics"]
    metric_3d = {method: position[method]["3D"] for method in comparison}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(
        comparison,
        [metric_3d[m]["point_rmse_m"] for m in comparison],
        color=[COLORS[m] for m in comparison],
    )
    axes[0].set_ylabel("Point RMSE (m)")
    axes[0].set_title("3D Position Distance RMSE")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.3)
    failure_rates = [metric_3d[m]["failure_rate_pct"] for m in comparison]
    nonzero_rates = [rate for rate in failure_rates if rate > 0]
    display_floor = min(nonzero_rates) / 10 if nonzero_rates else 1e-6
    axes[1].bar(
        comparison,
        [max(rate, display_floor) for rate in failure_rates],
        color=[COLORS[m] for m in comparison],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel(f"Windows with RMSE > {args.threshold:g} m (%) - log scale")
    axes[1].set_title("Threshold Failure Rate")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    for index, rate in enumerate(failure_rates):
        axes[1].text(index, max(rate, display_floor), f"{rate:.4g}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(args.output_dir / "baseline_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axis_labels = ["X", "Y", "Z", "3D"]
    bar_width = 0.8 / max(len(comparison), 1)
    x = np.arange(len(axis_labels))
    for index, method in enumerate(comparison):
        axes[0].bar(
            x + (index - (len(comparison) - 1) / 2) * bar_width,
            [position[method][axis]["point_rmse_m"] for axis in axis_labels],
            bar_width,
            color=COLORS[method],
            label=method,
        )
        axes[1].bar(
            x + (index - (len(comparison) - 1) / 2) * bar_width,
            [position[method][axis]["failure_rate_pct"] for axis in axis_labels],
            bar_width,
            color=COLORS[method],
            label=method,
        )
    axes[0].set_xticks(x, axis_labels)
    axes[0].set_ylabel("Position RMSE (m)")
    axes[0].set_title("Position Error by Axis and 3D Distance")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].set_xticks(x, axis_labels)
    axes[1].set_yscale("symlog", linthresh=0.001)
    axes[1].set_ylabel(f"Windows > {args.threshold:g} m (%)")
    axes[1].set_title("Threshold Failures by Axis")
    axes[1].grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "position_axis_comparison.png", dpi=180)
    plt.close(fig)

    acceleration = summary["acceleration_metrics"]
    fig, ax = plt.subplots(figsize=(10, 5))
    acc_methods = [
        method
        for method in summary["neural_methods"] + ["RK4 only", "Last acceleration"]
        if method in acceleration
    ]
    acceleration_labels = ["X", "Y", "Z", "Vector"]
    acc_bar_width = 0.8 / max(len(acc_methods), 1)
    for index, method in enumerate(acc_methods):
        item = acceleration[method]
        ax.bar(
            np.arange(4) + (index - (len(acc_methods) - 1) / 2) * acc_bar_width,
            [item["x_rmse"], item["y_rmse"], item["z_rmse"], item["vector_rmse"]],
            acc_bar_width,
            color=COLORS[method],
            label=method,
        )
    ax.set_xticks(np.arange(4), acceleration_labels)
    ax.set_ylabel("Acceleration RMSE")
    ax.set_title("Acceleration Error by Axis")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "acceleration_axis_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method in comparison:
        key = method_key(method)
        values = np.array([float(row[f"{key}_3d_mean_window_rmse_m"]) for row in rows])
        values.sort()
        cdf = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, cdf, label=method, color=COLORS[method], linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel(f"Per-flight mean {forecast_label} 3D distance RMSE (m) - log scale")
    ax.set_ylabel("Fraction of flights")
    ax.set_title("Per-Flight Error Distribution")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "per_flight_error_cdf.png", dpi=180)
    plt.close(fig)

    key = f"{method_key(primary)}_3d_mean_window_rmse_m"
    worst_rows = sorted(rows, key=lambda row: float(row[key]), reverse=True)[:30]
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [str(row["file"]).removeprefix("flight_").removesuffix(".parquet") for row in worst_rows][::-1]
    values = [float(row[key]) for row in worst_rows][::-1]
    ax.barh(labels, values, color=COLORS[primary])
    ax.set_xlabel(f"Mean {forecast_label} 3D distance RMSE (m)")
    ax.set_title(f"30 Hardest Flights for {primary}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "hardest_flights.png", dpi=180)
    plt.close(fig)

    shown = worst[: min(3, len(worst))]
    trajectory_methods = [
        method
        for method in summary["neural_methods"] + ["Polynomial", "RK4 only", "Last acceleration"]
        if shown and method in shown[0].predictions
    ]
    fig, axes = plt.subplots(len(shown), 3, figsize=(16, 3.8 * len(shown)), squeeze=False)
    for row_index, item in enumerate(shown):
        for axis_index, axis_label in enumerate(["X", "Y", "Z"]):
            axis = axes[row_index, axis_index]
            axis.plot(item.relative_time, item.actual[:, axis_index], color="black", linewidth=2.5, label="Actual")
            for method in trajectory_methods:
                axis.plot(
                    item.relative_time,
                    item.predictions[method][:, axis_index],
                    color=COLORS[method],
                    label=method,
                )
            axis.set_title(f"{axis_label}: {item.flight}, 3D RMSE={item.score:.3f} m")
            axis.set_xlabel("Horizon time (s)")
            axis.set_ylabel(f"{axis_label} position (m)")
            axis.grid(alpha=0.3)
            if axis_index == 0:
                axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.output_dir / "worst_gru_trajectories.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(illustrative), 3, figsize=(16, 3.8 * len(illustrative)), squeeze=False)
    requested = [0.25, 0.5, 1.0, args.threshold]
    illustrative_methods = [
        method
        for method in summary["neural_methods"] + ["Polynomial", "RK4 only", "Last acceleration"]
        if illustrative and method in illustrative[0].predictions
    ]
    for row_index, (target, item) in enumerate(zip(requested, illustrative, strict=False)):
        for axis_index, axis_label in enumerate(["X", "Y", "Z"]):
            axis = axes[row_index, axis_index]
            axis.plot(item.relative_time, item.actual[:, axis_index], color="black", linewidth=2.5, label="Actual")
            for method in illustrative_methods:
                axis.plot(
                    item.relative_time,
                    item.predictions[method][:, axis_index],
                    color=COLORS[method],
                    label=method,
                )
            axis.set_title(
                f"{axis_label}, nearest {target:g}m 3D error: actual={item.score:.3f} m"
            )
            axis.set_xlabel("Horizon time (s)")
            axis.set_ylabel(f"{axis_label} position (m)")
            axis.grid(alpha=0.3)
            if axis_index == 0:
                axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.output_dir / "illustrative_gru_trajectories.png", dpi=180)
    plt.close(fig)

    landing = summary.get("landing_horizon_metrics", {})
    valid_horizons = [item for item in landing.values() if item.get("flights")]
    if valid_horizons:
        valid_horizons.sort(key=lambda item: item["lead_time_seconds_median"])
        leads = [item["lead_time_seconds_median"] for item in valid_horizons]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for method in comparison:
            endpoints = [item["methods"][method]["landing_endpoint_metrics"] for item in valid_horizons]
            axes[0].plot(
                leads,
                [entry["final_difference_in_landing_rmse_3d_m"] for entry in endpoints],
                marker="o",
                color=COLORS[method],
                label=method,
            )
            axes[1].plot(
                leads,
                [entry["landing_ground_spot_rmse_xy_m"] for entry in endpoints],
                marker="o",
                color=COLORS[method],
                label=method,
            )
        axes[0].set_title("Final Difference in Landing (3D)")
        axes[0].set_ylabel("Endpoint 3D RMSE (m)")
        axes[1].set_title("Landing Ground-Spot Difference (XY)")
        axes[1].set_ylabel("Endpoint horizontal RMSE (m)")
        for axis in axes:
            axis.set_xlabel("Prediction lead time before landing (s)")
            axis.set_yscale("log")
            axis.grid(alpha=0.3)
            axis.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "landing_error_by_horizon.png", dpi=180)
        plt.close(fig)

        longest = max(valid_horizons, key=lambda item: item["horizon_samples"])["horizon_samples"]
        longest_rows = [row for row in landing_rows if row["horizon_samples"] == longest]
        primary_key = method_key(primary)
        fig, ax = plt.subplots(figsize=(9, 7))
        for row in longest_rows:
            actual_x, actual_y = row["actual_landing_x"], row["actual_landing_y"]
            pred_x = row[f"{primary_key}_predicted_landing_x"]
            pred_y = row[f"{primary_key}_predicted_landing_y"]
            ax.plot([actual_x, pred_x], [actual_y, pred_y], color="#cccccc", linewidth=0.6)
        ax.scatter(
            [row["actual_landing_x"] for row in longest_rows],
            [row["actual_landing_y"] for row in longest_rows],
            color="black",
            s=18,
            label="Actual landing",
        )
        ax.scatter(
            [row[f"{primary_key}_predicted_landing_x"] for row in longest_rows],
            [row[f"{primary_key}_predicted_landing_y"] for row in longest_rows],
            color=COLORS[primary],
            s=18,
            label=f"{primary} predicted landing",
        )
        ax.set_title(f"Landing Spots at Longest Horizon ({longest} samples)")
        ax.set_xlabel("Landing X (m)")
        ax.set_ylabel("Landing Y (m)")
        ax.axis("equal")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "landing_spots_longest_horizon.png", dpi=180)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    resolve_paths(args)
    log(f"Writing evaluation outputs to {args.output_dir}")
    log(f"Repository: {args.repo}")
    log(f"Scaler data: {args.scaler_data_dir}")
    log(f"External test data: {args.eval_data_dir}")
    model_specs = configured_model_specs(args)
    log("Neural checkpoints selected:")
    for spec in model_specs:
        log(f"  {spec.name}: {spec.path} ({spec.output_mode})")
    GRU, PersistenceGatedGRU, calculate_x_b, load_parameters, load_thrust_curve = (
        configure_imports(args.repo)
    )
    model_root = args.repo / "source_data" / "R7_SIMLE"
    parameters_path = (
        args.parameters or model_root / "R7_ROBUSTNESS" / "parameters.json"
    )
    thrust_curve_path = args.thrust_curve or model_root / "R7_OUTPUT" / "thrust_source.csv"
    args.parameters_path_used = parameters_path
    args.thrust_curve_path_used = thrust_curve_path
    log(f"RK4 parameters: {parameters_path}")
    log(f"RK4 thrust curve: {thrust_curve_path}")
    parameters = load_parameters(parameters_path)
    thrust_curve = load_thrust_curve(thrust_curve_path)
    sampling_rate = 500.0 / args.downsample
    device, gpu_ids = select_device(args)
    mean_in, std_in, mean_acc, std_acc, mean_xs, std_xs, scaler_meta = compute_scalers(
        args, calculate_x_b, parameters, thrust_curve, sampling_rate
    )
    models = load_networks(
        GRU,
        PersistenceGatedGRU,
        model_specs,
        device,
        gpu_ids,
        input_size=len(mean_in),
    )
    summary, rows, worst, illustrative = evaluate(
        args,
        model_specs,
        models,
        device,
        calculate_x_b,
        parameters,
        thrust_curve,
        sampling_rate,
        mean_in,
        std_in,
        mean_acc,
        std_acc,
        mean_xs,
        std_xs,
    )
    summary["scaler_metadata"] = scaler_meta
    landing_metrics, landing_rows = evaluate_landing_horizons(
        args,
        model_specs,
        models,
        device,
        calculate_x_b,
        parameters,
        thrust_curve,
        sampling_rate,
        mean_in,
        std_in,
        mean_acc,
        std_acc,
        mean_xs,
        std_xs,
    )
    summary["landing_horizon_metrics"] = landing_metrics
    write_outputs(args, summary, rows, worst, illustrative, landing_rows)
    render_plots(args, summary, rows, worst, illustrative, landing_rows)
    log(f"Finished. Open the PNG files and summary.txt in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
