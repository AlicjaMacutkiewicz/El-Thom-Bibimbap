import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# adjust path to find the classical simulator module
CLASSICAL_MODEL_SRC = Path(__file__).resolve().parents[2] / "classical_model" / "src"
if str(CLASSICAL_MODEL_SRC) not in sys.path:
    sys.path.append(str(CLASSICAL_MODEL_SRC))

from RK4Sim import rk4_t  # noqa: E402

# ---> physical constants <---
GRAVITY = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)
R_EARTH = 6371000.0
DEFAULT_FUEL_MASS = 13.04
DEFAULT_ISP = 204.26
G0 = 9.80665

# cache for RK4 baseline calculations to prevent redundant numerical integration
_RK4_CACHE = {}
DEFAULT_CONDITIONS = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def load_parameters(parameters_path):
    """
    Loads rocket parameters required to calculate the deterministic
    baseline acceleration (x_b).
    """
    with open(parameters_path, encoding="utf-8") as file:
        return json.load(file)


def load_thrust_curve(thrust_path):
    """
    Loads the engine's thrust curve (Time vs. Thrust).
    The calculate_x_b() function will later interpolate between these known points.
    """
    curve = np.loadtxt(thrust_path, delimiter=",", dtype=np.float32)

    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError("thrust curve must have two columns: time, thrust")
    return curve


def get_launch_direction(parameters):
    """
    Converts launch angles (inclination, heading) from parameters into a 3D direction vector.
    This vector dictates how the engine thrust is distributed across the X, Y, and Z axes.
    """
    flight = parameters.get("flight", {})

    # JSON stores angles in degrees, numpy trigonometric functions expect radians
    inclination = np.deg2rad(float(flight.get("inclination", 90.0)))
    heading = np.deg2rad(float(flight.get("heading", 0.0)))

    # inclination = 90 degrees implies a fully vertical launch
    horizontal = np.cos(inclination)
    return np.array(
        [
            horizontal * np.cos(heading),  # X
            horizontal * np.sin(heading),  # Y
            np.sin(inclination),  # Z
        ],
        dtype=np.float32,
    )


def _thrust_curve_to_dict(thrust_curve):
    return dict(zip(thrust_curve[:, 0], thrust_curve[:, 1], strict=False))


def _rk4_cache_key(parameters, thrust_curve):
    """Generates a unique hash key for the RK4 cache based on initial conditions."""
    rocket = parameters.get("rocket", {})
    motors = parameters.get("motors", {})
    stored_results = parameters.get("stored_results", {})

    return (
        float(rocket.get("mass", 50.876)),
        float(motors.get("fuel_mass", DEFAULT_FUEL_MASS)),
        float(motors.get("isp", DEFAULT_ISP)),
        float(
            stored_results.get(
                "flight_time", np.max(thrust_curve[:, 0]) if len(thrust_curve) else 0.0
            )
        ),
        tuple(get_launch_direction(parameters).tolist()),
        thrust_curve.shape,
        float(thrust_curve[0, 0]),
        float(thrust_curve[-1, 0]),
        float(thrust_curve[:, 1].sum()),
    )


def _get_rk4_tables(parameters, thrust_curve, sampling_rate):
    """Executes the classical RK4 simulator and caches the resulting baseline trajectory."""
    cache_key = _rk4_cache_key(parameters, thrust_curve)
    cached = _RK4_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rocket = parameters.get("rocket", {})
    motors = parameters.get("motors", {})
    stored_results = parameters.get("stored_results", {})

    rocket_mass = float(rocket.get("mass", 50.876))
    fuel_mass = float(motors.get("fuel_mass", DEFAULT_FUEL_MASS))
    isp = float(motors.get("isp", DEFAULT_ISP))
    flight_time = float(stored_results.get("flight_time", np.max(thrust_curve[:, 0])))

    trajectory = rk4_t(
        start_position=np.array([0.0, 0.0, R_EARTH], dtype=np.float32),
        rocket_mass=rocket_mass,
        fuel_mass=fuel_mass,
        angle=get_launch_direction(parameters),
        time=flight_time,
        thrust=_thrust_curve_to_dict(thrust_curve),
        isp=isp,
        sampling_rate=sampling_rate,
    )

    rk4_times = np.array([row[0] for row in trajectory], dtype=np.float32)
    rk4_positions = np.array([row[1] for row in trajectory], dtype=np.float32)
    rk4_velocities = np.array([row[2] for row in trajectory], dtype=np.float32)
    rk4_accelerations = np.array([row[3] for row in trajectory], dtype=np.float32)

    _RK4_CACHE[cache_key] = (rk4_times, rk4_positions, rk4_velocities, rk4_accelerations)
    return _RK4_CACHE[cache_key]


def _conditioned_physics(parameters, thrust_curve, condition):
    """Return parameters and thrust adjusted by launch-condition inputs.

    condition order:
    0. oxidizer fraction,
    1. pressure/thrust scale,
    2. rocket mass scale.
    """
    oxidizer_fraction, pressure_scale, rocket_mass_scale = [
        float(value) for value in condition
    ]
    if not 0.0 < oxidizer_fraction <= 1.0:
        raise ValueError(f"oxidizer fraction must be in (0, 1], got {oxidizer_fraction}")
    if pressure_scale <= 0.0 or rocket_mass_scale <= 0.0:
        raise ValueError("pressure and rocket-mass scales must be positive")

    conditioned_parameters = copy.deepcopy(parameters)
    rocket = conditioned_parameters.setdefault("rocket", {})
    motors = conditioned_parameters.setdefault("motors", {})

    base_rocket_mass = float(rocket.get("mass", 50.876))
    base_fuel_mass = float(motors.get("fuel_mass", DEFAULT_FUEL_MASS))
    nominal_oxidizer_mass = motors.get("nominal_oxidizer_mass_kg")
    fixed_fuel_mass = motors.get("fixed_fuel_mass_kg")

    rocket["mass"] = base_rocket_mass * rocket_mass_scale
    if "inertia" in rocket:
        rocket["inertia"] = [float(value) * rocket_mass_scale for value in rocket["inertia"]]

    conditioned_thrust = np.asarray(thrust_curve, dtype=np.float32).copy()
    if nominal_oxidizer_mass is not None and fixed_fuel_mass is not None:
        nominal_oxidizer_mass = float(nominal_oxidizer_mass)
        fixed_fuel_mass = float(fixed_fuel_mass)
        configured_fuel_mass = nominal_oxidizer_mass + fixed_fuel_mass
        if not np.isclose(configured_fuel_mass, base_fuel_mass, rtol=0.02):
            raise ValueError(
                "conditioned RK4 motor masses do not match: "
                f"configured={configured_fuel_mass}, fuel_mass={base_fuel_mass}"
            )

        motors["fuel_mass"] = base_fuel_mass * oxidizer_fraction
        rocket["mass"] += fixed_fuel_mass * (1.0 - oxidizer_fraction)
        burn_time_exponent = float(motors.get("oxidizer_burn_time_exponent", 1.0))
        burn_time_scale = oxidizer_fraction**burn_time_exponent
        start_time = conditioned_thrust[0, 0]
        conditioned_thrust[:, 0] = start_time + (
            conditioned_thrust[:, 0] - start_time
        ) * burn_time_scale
        motors["isp"] = float(motors.get("isp", DEFAULT_ISP)) * pressure_scale
    else:
        # Compatibility with parameters used by the earlier generic
        # propellant-fraction datasets.
        motors["fuel_mass"] = base_fuel_mass * oxidizer_fraction

    if "grain_initial_height" in motors:
        motors["grain_initial_height"] = (
            float(motors["grain_initial_height"]) * oxidizer_fraction
        )

    conditioned_thrust[:, 1] *= pressure_scale
    return conditioned_parameters, conditioned_thrust


def _condition_groups(condition_array):
    """Yield each condition and its row indices without repeatedly scanning the array."""
    rounded = np.round(condition_array.astype(np.float32), 6)
    unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(np.bincount(inverse, minlength=len(unique)))))
    for index, condition in enumerate(unique):
        yield condition, order[offsets[index] : offsets[index + 1]]


def calculate_x_b_conditioned(times, parameters, thrust_curve, sampling_rate, conditions=None):
    """Return RK4 baseline acceleration using optional per-row scenario conditions."""
    if conditions is None:
        return calculate_x_b(times, parameters, thrust_curve, sampling_rate)

    if not torch.is_tensor(times):
        times_tensor = torch.as_tensor(times, dtype=torch.float32)
    else:
        times_tensor = times
    times_np = times_tensor.detach().cpu().numpy()
    if torch.is_tensor(conditions):
        conditions_np = conditions.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        conditions_np = np.asarray(conditions, dtype=np.float32)

    if conditions_np.ndim == 1:
        conditioned_parameters, conditioned_thrust = _conditioned_physics(
            parameters, thrust_curve, conditions_np
        )
        return calculate_x_b(times_tensor, conditioned_parameters, conditioned_thrust, sampling_rate)

    if conditions_np.shape[:-1] == times_np.shape and conditions_np.shape[-1] == 3:
        flat_times = times_np.reshape(-1)
        flat_conditions = conditions_np.reshape(-1, 3)
        output = np.empty((flat_times.shape[0], 3), dtype=np.float32)
        for condition, indices in _condition_groups(flat_conditions):
            conditioned_parameters, conditioned_thrust = _conditioned_physics(
                parameters, thrust_curve, condition
            )
            output[indices] = calculate_x_b(
                flat_times[indices], conditioned_parameters, conditioned_thrust, sampling_rate
            )
        return output.reshape((*times_np.shape, 3))

    if conditions_np.ndim == 2 and conditions_np.shape[0] == times_np.shape[0]:
        output = np.empty((*times_np.shape, 3), dtype=np.float32)
        for condition, indices in _condition_groups(conditions_np):
            conditioned_parameters, conditioned_thrust = _conditioned_physics(
                parameters, thrust_curve, condition
            )
            output[indices] = calculate_x_b(
                times_np[indices], conditioned_parameters, conditioned_thrust, sampling_rate
            )
        return output

    raise ValueError(
        "conditions must have shape (3,), (N, 3), or times.shape + (3,), "
        f"got conditions={conditions_np.shape}, times={times_np.shape}"
    )


def calculate_x_b(times, parameters, thrust_curve, sampling_rate):
    """Return base acceleration x_b(t) from the RK4 baseline model.

    x_b - base acceleration

    In this project we split total acceleration into two parts:

        x_total = x_b + x_s

    where:
        x_b = known/base physics calculated by classical_model/src/RK4Sim.py

        x_s = unknown /nonlinear /messy part for example wind type shit

    The neural network should not waste capacity learning the deterministic RK4 baseline.
    So this function gets x_b from RK4 and the GRU learns only the residual x_s.
    """

    # We get tensors but for fun i added that you can also get a list or NumPy array it converts to tensor
    if not torch.is_tensor(times):
        times = torch.as_tensor(times, dtype=torch.float32)

    # Keep all tensors on the same device and with the same dtype as times.

    rk4_times, _, _, rk4_accelerations = _get_rk4_tables(parameters, thrust_curve, sampling_rate)
    flat_times = times.detach().cpu().numpy().reshape(-1)

    x_b = np.stack(
        [np.interp(flat_times, rk4_times, rk4_accelerations[:, axis]) for axis in range(3)],
        axis=-1,
    )
   # AFTER
    x_b = x_b.reshape((*times.shape, 3))

    # gravity_correction = torch.zeros(3, dtype=dtype, device=device)
    # gravity_correction[2] = G0
    # x_b = torch.as_tensor(x_b, dtype=dtype, device=device)
    # x_b = x_b + gravity_correction
    
    return x_b


def calculate_position(
    times,
    parameters,
    thrust_curve,
    sampling_rate,
):
    """
    Returns base acceleration (x_b) calculated by the classical RK4 model.

    In this hybrid architecture, total acceleration is split: x_total = x_b + x_s.
    The deterministic physics (gravity, engine thrust) are handled here, allowing
    the neural network to focus entirely on predicting the nonlinear residuals (x_s).
    """
    if not torch.is_tensor(times):
        times = torch.as_tensor(
            times,
            dtype=torch.float32,
        )

    device = times.device
    dtype = times.dtype

    rk4_times, rk4_positions, _, _ = _get_rk4_tables(parameters, thrust_curve, sampling_rate)
    flat_times = times.detach().cpu().numpy().reshape(-1)

    positions = np.stack(
        [np.interp(flat_times, rk4_times, rk4_positions[:, axis]) for axis in range(3)],
        axis=-1,
    )
    positions = positions.reshape((*times.shape, 3))

    return torch.as_tensor(positions, dtype=dtype, device=device)


class BaseAccelerationMSELoss(nn.Module):
    """
    Computes the Mean Squared Error against the nonlinear acceleration residual.

    Since the GRU is tasked with predicting only the residual (x_s), the target is
    derived by subtracting the classical baseline (x_b) from the simulator's total output.
    """

    def __init__(self, parameters, thrust_curve, sampling_rate, model_type):
        super().__init__()
        self.parameters = parameters
        self.thrust_curve = thrust_curve
        self.mse = nn.MSELoss()
        self.sampling_rate = sampling_rate
        self.model_type = model_type

    def forward(self, predicted_x_s, true_x_total, times, conditions=None):
        if self.model_type == "gru":    # just gru
            true_target = true_x_total[:, :, :3]
        else:                           # residual gru
            true_target = calculate_residual_target(
                true_x_total,
                times,
                self.parameters,
                self.thrust_curve,
                self.sampling_rate,
                conditions,
            )
        return self.mse(predicted_x_s, true_target)


def calculate_residual_target(true_x_total, times, parameters, thrust_curve, sampling_rate, conditions=None):
    """Return the residual target learned by GRU: x_s = x_total - x_b."""
    x_b_numpy = calculate_x_b_conditioned(times, parameters, thrust_curve, sampling_rate, conditions)
    x_b = torch.as_tensor(x_b_numpy, dtype=true_x_total.dtype, device=true_x_total.device)
    return true_x_total[:, :, :3] - x_b


def integrate_acceleration(acceleration, times, initial_position, initial_velocity, initial_time):
    """
    Numerically integrates acceleration into velocity and position using Newtonian kinematics.
    """
    if not torch.is_tensor(times):
        times = torch.as_tensor(times, dtype=torch.float32)

    # Keep all tensors on the same device and with the same dtype as times.

    positions = []
    position = initial_position
    velocity = initial_velocity
    previous_time = initial_time

    for step in range(acceleration.shape[1]):
        current_time = times[:, step]
        dt = (current_time - previous_time).clamp_min(0.0).unsqueeze(-1)
        current_acceleration = acceleration[:, step, :]

        # standard kinematic equations
        position = position + (velocity * dt) + (0.5 * current_acceleration * dt * dt)
        velocity = velocity + (current_acceleration * dt)

        positions.append(position)
        previous_time = current_time

    return torch.stack(positions, dim=1), velocity


class PINNPositionMSELoss(nn.Module):
    """
    Physics-Informed Neural Network (PINN) Loss component.

    Reconstructs the total acceleration (GRU output + RK4 baseline) and integrates
    it via Newtonian kinematics. Penalizes the network if the resulting simulated
    trajectory drifts from the true simulator position.
    """

    def __init__(self, parameters, thrust_curve, sampling_rate, model_type):
        super().__init__()
        self.parameters = parameters
        self.thrust_curve = thrust_curve
        self.mse = nn.MSELoss()
        self.sampling_rate = sampling_rate
        self.model_type = model_type

    def forward(
        self,
        predicted_x_s,
        true_position,
        times,
        initial_position,
        initial_velocity,
        initial_time,
        conditions=None,
    ):
        if self.model_type in ["gru_res", "gru_res_phys"]:
            x_b_numpy = calculate_x_b_conditioned(
                times, self.parameters, self.thrust_curve, self.sampling_rate, conditions
            )
            x_b = torch.as_tensor(x_b_numpy, dtype=predicted_x_s.dtype, device=predicted_x_s.device)
            predicted_x_total = predicted_x_s + x_b
        else:
            predicted_x_total = predicted_x_s

        integrated_position, _ = integrate_acceleration(
            predicted_x_total,
            times,
            initial_position,
            initial_velocity,
            initial_time,
        )

        return self.mse(integrated_position, true_position)


class TotalLoss(nn.Module):
    """
    Composite loss function balancing basic residual MSE and PINN kinematic constraints.

    Args:
        lambda_h (float): Hyperparameter controlling the strictness of physical constraints.
    """

    def __init__(
        self,
        parameters,
        thrust_curve,
        target_mean,
        target_std,
        mean_pos,
        std_pos,
        sampling_rate,
        lambda_h=1e-6,
        model_type="gru_res_phys",
        lambda_regret=0.1,
    ):
        super().__init__()
        self.model_type = model_type

        self.acc_loss = BaseAccelerationMSELoss(parameters, thrust_curve, sampling_rate, model_type)
        self.pinn_loss = PINNPositionMSELoss(parameters, thrust_curve, sampling_rate, model_type)

        self.target_mean = torch.tensor(target_mean, dtype=torch.float32)
        self.target_std  = torch.tensor(target_std,  dtype=torch.float32)
        self.mean_pos = torch.tensor(mean_pos, dtype=torch.float32)
        self.std_pos = torch.tensor(std_pos, dtype=torch.float32)

        self.lambda_h = lambda_h        
        self.lambda_regret = lambda_regret
        self.mse = nn.MSELoss()

    def forward(
        self,
        preds,
        acc_batch,
        pos_batch,
        t_batch,
        initial_pos_batch,
        initial_vel_batch,
        initial_time_batch,
        condition_batch=None,
        anchor_acc_batch=None,
    ):
        device = preds.device

        denormalized_acc_target = acc_batch[:, :, :3] * self.target_std.to(device) + self.target_mean.to(device)
        denormalized_pos_target = pos_batch[:, :, :3] * self.std_pos.to(device) + self.mean_pos.to(
            device
        )

        denorm_initial_pos = initial_pos_batch * self.std_pos.to(device) + self.mean_pos.to(device)
        denorm_initial_vel = initial_vel_batch * self.std_pos.to(device)

        if self.model_type == "gru_res_phys_persist_gate":
            if preds.shape[-1] != 6:
                raise ValueError(
                    "gru_res_phys_persist_gate expects 6 outputs: 3 residuals + 3 gate logits."
                )
            if anchor_acc_batch is None:
                raise ValueError(
                    "gru_res_phys_persist_gate requires the last observed acceleration anchor."
                )

            residual = (
                preds[:, :, :3] * self.target_std.to(device) + self.target_mean.to(device)
            )
            gate = torch.sigmoid(preds[:, :, 3:6])
            x_b_numpy = calculate_x_b_conditioned(
                t_batch,
                self.acc_loss.parameters,
                self.acc_loss.thrust_curve,
                self.acc_loss.sampling_rate,
                condition_batch,
            )
            x_b = torch.as_tensor(x_b_numpy, dtype=preds.dtype, device=device)
            learned_candidate = x_b + residual
            anchor = (
                anchor_acc_batch[:, :3] * self.target_std.to(device)
                + self.target_mean.to(device)
            )
            anchor_sequence = anchor[:, None, :].expand_as(learned_candidate)
            predicted_total = anchor_sequence + gate * (learned_candidate - anchor_sequence)

            predicted_position, _ = integrate_acceleration(
                predicted_total,
                t_batch,
                denorm_initial_pos,
                denorm_initial_vel,
                initial_time_batch,
            )
            anchor_position, _ = integrate_acceleration(
                anchor_sequence,
                t_batch,
                denorm_initial_pos,
                denorm_initial_vel,
                initial_time_batch,
            )

            mse_acc = self.mse(predicted_total, denormalized_acc_target)
            pinn = self.mse(predicted_position, denormalized_pos_target)
            model_window_error = (
                (predicted_position - denormalized_pos_target).square().sum(dim=-1).mean(dim=-1)
            )
            anchor_window_error = (
                (anchor_position - denormalized_pos_target).square().sum(dim=-1).mean(dim=-1)
            )
            regret = torch.relu(model_window_error - anchor_window_error).mean()
            return mse_acc + self.lambda_h * pinn + self.lambda_regret * regret

        # Standard variants emit only three normalized acceleration/residual channels.
        denormalized_preds = preds * self.target_std.to(device) + self.target_mean.to(device)

        # calculate isolated losses
        pinn = self.pinn_loss(
            denormalized_preds,
            denormalized_pos_target,
            t_batch,
            denorm_initial_pos,
            denorm_initial_vel,
            initial_time_batch,
            condition_batch,
        )

        mse_acc = self.acc_loss(
            denormalized_preds, denormalized_acc_target, t_batch, condition_batch
        )

        physics_accounted = bool(self.model_type == "gru_res_phys" or self.model_type == "gru_phys")

        return mse_acc + (self.lambda_h * pinn if physics_accounted else 0)

def default_physics_paths():
    """Retrieves absolute paths for default physics configurations."""

    root = Path(__file__).resolve().parents[3]
    model_root = root / "source_model" / "R7_SIMLE"

    return (
        model_root / "R7_ROBUSTNESS" / "parameters.json",
        model_root / "R7_OUTPUT" / "thrust_source.csv",
    )
