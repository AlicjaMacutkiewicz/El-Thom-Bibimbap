import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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


def ensure_condition_columns(flight_data, source="flight data"):
    """Populate nominal conditions and reject ambiguous legacy randomization data."""
    oxidizer_column = CONDITION_COLUMNS[0]
    legacy_column = CONDITION_ALIASES[oxidizer_column]
    if oxidizer_column not in flight_data.columns:
        if legacy_column in flight_data.columns:
            legacy_values = flight_data[legacy_column].to_numpy(dtype=np.float32)
            if not np.allclose(legacy_values, 1.0):
                raise ValueError(
                    f"{source} contains variable {legacy_column} but no {oxidizer_column}. "
                    "This is a legacy generic-propellant dataset and cannot be used as "
                    "oxidizer-aware training data; regenerate it with "
                    "robustness_oxidizer_40_100.json."
                )
            flight_data[oxidizer_column] = legacy_values
        else:
            flight_data[oxidizer_column] = 1.0

    for column in CONDITION_COLUMNS[1:]:
        if column not in flight_data.columns:
            flight_data[column] = 1.0
    return flight_data


def read_flight_data(start_flight, num_of_flights, output_dir="../../../1955-1959", downsample=1):
    """
    Loads telemetry and simulation data from Parquet files.

    Separates input features (sensors plus scenario conditions), target accelerations,
    raw scenario conditions, positions, and times for physics integration.

    Args:
        start_flight (int): Index of the first flight file to load.
        num_of_flights (int): Total number of flight files to process.
        output_dir (str): Relative or absolute path to the directory containing .parquet files.

    Returns:
        tuple: Lists of numpy arrays containing inputs, raw conditions, targets, positions,
            and timestamps.
    """

    flights_inputs = []
    flights_conditions = []
    flights_targets = []
    flight_positions = []
    flight_times = []

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    flight_files = sorted(output_path.glob("flight_*.parquet"))

    if len(flight_files) < start_flight + num_of_flights:
        raise ValueError(
            f"expected at least {start_flight + num_of_flights} flight files, found {len(flight_files)}."
        )

    for file_path in flight_files[start_flight : start_flight + num_of_flights]:
        flight_data = pd.read_parquet(file_path)

        if not set(SENSOR_COLUMNS).issubset(flight_data.columns):
            raise ValueError(f"{file_path} is missing required sensor columns: {SENSOR_COLUMNS}")
        ensure_condition_columns(flight_data, file_path)
        input_data = flight_data[INPUT_COLUMNS].values.astype(np.float32)[::downsample]
        condition_data = flight_data[CONDITION_COLUMNS].values.astype(np.float32)[::downsample]

        target_columns = ["Acceleration_X", "Acceleration_Y", "Acceleration_Z"]
        target_data = flight_data[target_columns].values.astype(np.float32)[::downsample]

        position_columns = ["Position_X", "Position_Y", "Position_Z"]
        position_data = flight_data[position_columns].values.astype(np.float32)[::downsample]

        if "Time" in flight_data.columns:
            time_data = flight_data["Time"].to_numpy(dtype=np.float32)[::downsample]
        else:
            time_data = flight_data.index.to_numpy(dtype=np.float32)[::downsample]

        flights_inputs.append(input_data)
        flights_conditions.append(condition_data)
        flights_targets.append(target_data)
        flight_positions.append(position_data)
        flight_times.append(time_data)

    return flights_inputs, flights_conditions, flights_targets, flight_positions, flight_times


def split_flights(flights, split_ratio=0.8, seed=41):
    """
    Splits a list of flight arrays into training and testing sets.

    Args:
        flights (list): List of numpy arrays representing individual flights.
        split_ratio (float): Fraction of the data to use for training (default: 0.8).

    Returns:
        tuple: (train_flights, test_flights)
    """
    random.seed(seed)

    num_flights = len(flights)
    indices = list(range(num_flights))

    random.shuffle(indices)
    split_idx = int(len(flights) * split_ratio)

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_flights = [flights[i] for i in train_indices]
    test_flights = [flights[i] for i in test_indices]

    return train_flights, test_flights


def normalize_flights(flights, array):
    """
    Applies Z-score normalization to each flight independently.

    Args:
        flights (list): List of flight arrays to normalize.
        array (np.ndarray): The reference array (usually training data) to calculate mean and std.

    Returns:
        list: Normalized flight arrays.
    """
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    std = np.where(std == 0, 1e-6, std)  # prevent division by zero
    return [(flight - mean) / std for flight in flights]


def estimate_velocity(positions, times):
    """
    Estimates velocity from positional data using numerical differentiation (central difference).

    Args:
        positions (np.ndarray): Array of shape (timesteps, 3) representing X, Y, Z positions.
        times (np.ndarray): Array of shape (timesteps,) representing time in seconds.

    Returns:
        np.ndarray: Estimated velocities of shape (timesteps, 3).
    """
    velocities = np.zeros_like(positions, dtype=np.float32)

    if len(positions) < 2:
        return velocities

    dt = np.diff(times).astype(np.float32)
    dt = np.where(dt == 0.0, 1e-6, dt)  # prevent division by zero
    segment_velocity = np.diff(positions, axis=0) / dt[:, None]

    # edge cases
    velocities[0] = segment_velocity[0]
    velocities[-1] = segment_velocity[-1]

    # internal points
    if len(positions) > 2:
        velocities[1:-1] = 0.5 * (segment_velocity[:-1] + segment_velocity[1:])

    return velocities


def make_sequences(
    flights_inputs,
    flights_conditions,
    flights_targets,
    flight_positions,
    flight_times,
    seq_len,
    pred_len,
):
    """
    Converts full-flight time-series data into overlapping training sequences for the GRU model.

    Splits the synchronized telemetry data into a historical observation window (Spin-Up) and a future
    prediction window (Cut-Off).

    Args:
        flights_inputs (list of np.ndarray): Model input data, including sensors and scenario columns.
        flights_conditions (list of np.ndarray): Raw scenario columns for conditioned RK4 calls.
        flights_targets (list of np.ndarray): Target true accelerations from the simulator.
        flight_positions (list of np.ndarray): True physical positions (X, Y, Z).
        flight_times (list of np.ndarray): Timestamps for the telemetry data.
        seq_len (int): Length of the historical input sequence (lookback window).
        pred_len (int): Length of the future sequence to predict (forecast window).

    Returns:
        tuple: PyTorch tensors for X (inputs), y_acc (target accelerations), y_pos (target positions),
            t_y (target times), and initial integration conditions (pos, vel, time).
    """
    X, condition_context, y_hist_acc, y_acc, y_pos, t_y, initial_pos, initial_vel, initial_time = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # iterate simultaneously over inputs and targets
    for f_in, f_cond, f_tar, positions, times in zip(
        flights_inputs, flights_conditions, flights_targets, flight_positions, flight_times, strict=False
    ):
        velocities = estimate_velocity(positions, times)

        for i in range(len(f_in) - seq_len - pred_len):
            start_idx = i + seq_len - 1
            target_start_idx = i + seq_len
            target_end_idx = target_start_idx + pred_len

            # Extract seq_len values from past observations.
            X.append(f_in[i : i + seq_len])
            condition_context.append(f_cond[start_idx])

            # Keep the matching past true accelerations for visualization only.
            y_hist_acc.append(f_tar[i : i + seq_len])

            # Extract pred_len future values to be predicted  (Accelerations - 3 columns)
            y_acc.append(f_tar[target_start_idx:target_end_idx])
            y_pos.append(positions[target_start_idx:target_end_idx])

            # Store exact times for the target part
            t_y.append(times[target_start_idx:target_end_idx])
            initial_pos.append(positions[start_idx])
            initial_vel.append(velocities[start_idx])
            initial_time.append(times[start_idx])

    # convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    condition_context = np.array(condition_context, dtype=np.float32)
    y_hist_acc = np.array(y_hist_acc, dtype=np.float32)
    y_acc = np.array(y_acc, dtype=np.float32)
    y_pos = np.array(y_pos, dtype=np.float32)
    t_y = np.array(t_y, dtype=np.float32)
    initial_pos = np.array(initial_pos, dtype=np.float32)
    initial_vel = np.array(initial_vel, dtype=np.float32)
    initial_time = np.array(initial_time, dtype=np.float32)

    # convert to tensors (required for model training)
    return (
        torch.from_numpy(X),
        torch.from_numpy(condition_context),
        torch.from_numpy(y_hist_acc),
        torch.from_numpy(y_acc),
        torch.from_numpy(y_pos),
        torch.from_numpy(t_y),
        torch.from_numpy(initial_pos),
        torch.from_numpy(initial_vel),
        torch.from_numpy(initial_time),
    )
