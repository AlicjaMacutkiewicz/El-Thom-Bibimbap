import calendar
import copy
import datetime
import json
import os
import random
import sys

# import cProfile
import time
import traceback
from pathlib import Path

import multiprocess as mp
import numpy as np
import tqdm
from custom_sensors.thermometer import Thermometer
from enviroment_api import (
    download_yearly_weather,
    get_enviroment_from_batched_file,
    get_enviroment_from_date,
)
from logger import Log
from pathos.multiprocessing import ProcessPool
from rocketpy import Accelerometer, Barometer, Environment, Function, Gyroscope, Rocket, SolidMotor
from rocketpy.stochastic import StochasticEnvironment, StochasticSolidMotor
from single_simulation import run_single_simulation

NUM_GPUS = 0
TEST_FLAG = False
try:
    import cupy as cp  # type: ignore

    NUM_GPUS = cp.cuda.runtime.getDeviceCount()
    Log.print_info(f"detected {NUM_GPUS} gpus, hardware acceleration enabled")
except Exception:
    cp = np
    print(" No gpus detected, back to cpu mode")

# @BRIEF
# Initializes rocket with data found in give JSON file
# @ARGUMENTS:
#     path_to_file -> STRING representing path to JSON file
#     drag_curve_csv -> STRING representing path to CSV file
#     thrust_source_csv -> STRING representing path to CSV file
# @RETURN
#     rocket -> Initialized rocketpy::Rocket class


def _descending_above_altitude_trigger(minimum_altitude):
    """Trigger an apogee parachute only when it is above the main-deploy altitude."""

    def trigger(_pressure, height, state):
        return state[5] < 0 and height >= minimum_altitude

    return trigger


def init_rocket_from_JSON(data, drag_curve, motor):
    name = data["id"]["rocket_name"]

    Log.print_info(f"Loading model: {name}")

    rocket_data = data["rocket"]
    Log.print_info("loading rocket")
    rocket = Rocket(
        radius=rocket_data["radius"],
        mass=rocket_data["mass"],
        inertia=tuple(rocket_data["inertia"]),
        power_off_drag=drag_curve,
        power_on_drag=drag_curve,
        center_of_mass_without_motor=rocket_data["center_of_mass_without_propellant"],
        coordinate_system_orientation=rocket_data["coordinate_system_orientation"],
    )
    motor_data = data["motors"]
    rocket.add_motor(motor, position=motor_data["position"])
    Log.print_info("loading nose")
    nose_data = data["nosecones"]
    rocket.add_nose(
        length=nose_data["length"],
        kind=nose_data["kind"],
        position=nose_data["position"],
        power=nose_data["power"],
    )
    Log.print_info("loading fins")
    fins = data["trapezoidal_fins"]["0"]
    rocket.add_trapezoidal_fins(
        n=fins["number"],
        root_chord=fins["root_chord"],
        tip_chord=fins["tip_chord"],
        span=fins["span"],
        position=fins["position"],
        sweep_length=fins["sweep_length"],
    )

    parachutes = data["parachutes"]
    altitude_deployments = [
        float(parachute["deploy_altitude"])
        for parachute in parachutes.values()
        if parachute["deploy_event"] != "apogee"
        and parachute.get("deploy_altitude") is not None
    ]
    highest_altitude_deployment = max(altitude_deployments, default=None)

    for p_id in parachutes:
        p_data = parachutes[p_id]
        if p_data["deploy_event"] == "apogee" and highest_altitude_deployment is not None:
            # When apogee is below the main deployment altitude, RocketPy's
            # built-in altitude and apogee triggers otherwise fire together.
            p_trigger = _descending_above_altitude_trigger(highest_altitude_deployment)
        else:
            p_trigger = (
                "apogee"
                if p_data["deploy_event"] == "apogee"
                else p_data["deploy_altitude"]
            )

        rocket.add_parachute(
            name=p_data["name"], cd_s=p_data["cds"], trigger=p_trigger, lag=p_data["deploy_delay"]
        )
    return rocket


def init_base_motor_from_JSON(data, thrust_source_csv):
    motor_data = data["motors"]

    Log.print_info("loading base motor")
    motor = SolidMotor(
        thrust_source=thrust_source_csv,
        dry_mass=motor_data["dry_mass"],
        dry_inertia=motor_data["dry_inertia"],
        nozzle_radius=motor_data["nozzle_radius"],
        grain_number=motor_data["grain_number"],
        grain_density=motor_data["grain_density"],
        grain_outer_radius=motor_data["grain_outer_radius"],
        grain_initial_inner_radius=motor_data["grain_initial_inner_radius"],
        grain_initial_height=motor_data["grain_initial_height"],
        grain_separation=motor_data["grain_separation"],
        grains_center_of_mass_position=motor_data["grains_center_of_mass_position"],
        center_of_dry_mass_position=motor_data["center_of_dry_mass_position"],
        nozzle_position=motor_data["nozzle_position"],
        # burn_time = 10.0, #declared based on in third static fire test
        throat_radius=motor_data["throat_radius"],
        coordinate_system_orientation=motor_data["coordinate_system_orientation"],
    )

    return motor


def init_stochastic_motor(base_motor, stochastic_motor_params):
    (
        grain_density_param,
        grain_outer_radius_param,
        grain_initial_inner_radius_param,
        grain_initial_height_param,
        nozzle_radius_param,
        throat_radius_param,
        total_impulse_param,
    ) = stochastic_motor_params

    stochastic_motor = StochasticSolidMotor(
        solid_motor=base_motor,
        grain_density=(
            base_motor.grain_density,
            1.5 * abs(grain_density_param * base_motor.grain_density),
            "normal",
        ),
        grain_outer_radius=(
            base_motor.grain_outer_radius,
            1.5 * abs(grain_outer_radius_param * base_motor.grain_outer_radius),
            "normal",
        ),
        grain_initial_inner_radius=(
            base_motor.grain_initial_inner_radius,
            1.5 * abs(grain_initial_inner_radius_param * base_motor.grain_initial_inner_radius),
            "normal",
        ),
        grain_initial_height=(
            base_motor.grain_initial_height,
            1.5 * abs(grain_initial_height_param * base_motor.grain_initial_height),
            "normal",
        ),
        nozzle_radius=(
            base_motor.nozzle_radius,
            1.5 * abs(nozzle_radius_param * base_motor.nozzle_radius),
            "normal",
        ),
        throat_radius=(
            base_motor.throat_radius,
            1.5 * abs(throat_radius_param * base_motor.throat_radius),
            "normal",
        ),
        total_impulse=(
            base_motor.total_impulse,
            1.5 * abs(total_impulse_param * base_motor.total_impulse),
            "normal",
        ),
    )
    return stochastic_motor


def get_environment_data_from_JSON(path_to_file):
    with open(path_to_file, encoding="utf-8") as file:
        data = json.load(file)
    Log.print_info(f"Reading from {path_to_file}")
    env_data = data["environment"]
    return env_data


def init_environment_from_JSON(path_to_file):
    with open(path_to_file, encoding="utf-8") as file:
        data = json.load(file)
    Log.print_info(f"Reading from {path_to_file}")
    env_data = data["environment"]

    date1 = datetime.datetime.strptime("19.03.2026", "%d.%m.%Y")
    print(date1)

    env = Environment(
        latitude=env_data["latitude"],
        longitude=env_data["longitude"],
        date=date1,
        elevation=env_data["elevation"],
        timezone="America/New_York",
    )
    return env


def init_stochastic_environment(env_data, date):
    env_base = get_enviroment_from_date(
        env_data, date, "ENV_DATA_" + date.strftime("%Y/%m/%d_%H:%M:%S")
    )
    env = StochasticEnvironment(env_base)
    return env


def init_flight_config_from_JSON(path_to_file):
    with open(path_to_file, encoding="utf-8") as file:
        data = json.load(file)
    Log.print_info(f"Reading from {path_to_file}")
    flight_data = data["flight"]
    heading = flight_data["heading"]
    rail_length = flight_data["rail_length"]
    return heading, rail_length


def init_accelerometer_from_JSON(path_to_file, name):
    with open(path_to_file, encoding="utf-8") as file:
        data = json.load(file)
    accel_data = data[name]
    accelerometer = Accelerometer(
        name=accel_data["name"],
        sampling_rate=accel_data["sampling_rate"],
        measurement_range=accel_data["measurement_range"],
        resolution=accel_data["resolution"],
        noise_density=accel_data["noise_density"],
        noise_variance=accel_data["noise_variance"],
        random_walk_density=accel_data["random_walk_density"],
        random_walk_variance=accel_data["random_walk_variance"],
        constant_bias=accel_data["constant_bias"],
        operating_temperature=accel_data["operating_temperature"],
        temperature_bias=accel_data["temperature_bias"],
        orientation=accel_data["orientation"],
    )
    return accelerometer


def init_gyroscope_from_JSON(path_to_file, name):
    with open(path_to_file, encoding="utf-8") as file:
        data = json.load(file)
    gyro_data = data[name]
    gyroscope = Gyroscope(
        name=gyro_data["name"],
        sampling_rate=gyro_data["sampling_rate"],
        measurement_range=gyro_data["measurement_range"],
        resolution=gyro_data["resolution"],
        noise_density=gyro_data["noise_density"],
        noise_variance=gyro_data["noise_variance"],
        random_walk_density=gyro_data["random_walk_density"],
        random_walk_variance=gyro_data["random_walk_variance"],
        constant_bias=gyro_data["constant_bias"],
        operating_temperature=gyro_data["operating_temperature"],
        temperature_bias=gyro_data["temperature_bias"],
        orientation=gyro_data["orientation"],
    )
    return gyroscope


def init_thermometer_from_JSON(path_to_file, name):
    with open(path_to_file, encoding="utf-8") as file:
        data = json.load(file)
    thermometer_data = data[name]
    thermometer = Thermometer(
        sampling_rate=thermometer_data["sampling_rate"],
        measurement_range=thermometer_data["measurement_range"],
        resolution=thermometer_data["resolution"],
        noise_density=thermometer_data["noise_density"],
        noise_variance=thermometer_data["noise_variance"],
        operating_temperature=thermometer_data["operating_temperature"],
        constant_bias=thermometer_data["constant_bias"],
    )
    return thermometer


def init_barometer_from_JSON(path_to_file, name):
    with open(path_to_file, encoding="utf-8") as file:
        data = json.load(file)
    barometer_data = data[name]
    barometer = Barometer(
        sampling_rate=barometer_data["sampling_rate"],
        measurement_range=barometer_data["measurement_range"],
        resolution=barometer_data["resolution"],
        noise_variance=barometer_data["noise_variance"],
        constant_bias=barometer_data["constant_bias"],
        operating_temperature=barometer_data["operating_temperature"],
    )
    return barometer


def add_sensors_to_rocket(rocket, acc_list):
    # acc_list.sort(key = lambda x: x.measurement_range)
    for a in acc_list:
        # TODO: replace 1 ?
        if TEST_FLAG:
            a.sampling_rate /= 100
        rocket.add_sensor(a, 1)
    return rocket


def init_stochastic_motor_params(path_to_file):
    with open(path_to_file, encoding="utf-8") as file:
        dataset = json.load(file)

    motor_data = dataset["stochastic_motor_params"]
    grain_density_param = motor_data["grain_density_param"]
    grain_outer_radius_param = motor_data["grain_outer_radius_param"]
    grain_initial_inner_radius_param = motor_data["grain_initial_inner_radius_param"]
    grain_initial_height_param = motor_data["grain_initial_height_param"]
    nozzle_radius_param = motor_data["nozzle_radius_param"]
    throat_radius_param = motor_data["throat_radius_param"]
    total_impulse_param = motor_data["total_impulse_param"]
    params = (
        grain_density_param,
        grain_outer_radius_param,
        grain_initial_inner_radius_param,
        grain_initial_height_param,
        nozzle_radius_param,
        throat_radius_param,
        total_impulse_param,
    )
    return params


def init_paths_from_json(main_paths_file):
    with open(main_paths_file, encoding="utf-8") as file:
        dataset = json.load(file)
    return dataset


def select_thrust_path(source_model_paths):
    """Select the canonical thrust source with legacy-key compatibility."""
    thrust_path = source_model_paths.get("thrust_source")
    if thrust_path is not None:
        return thrust_path
    return source_model_paths["thrust_source_100"]


def _resolve_path(path):
    return Path(path).expanduser()


def _normalize_fraction(value, name):
    value = float(value)
    if value > 1.0:
        value /= 100.0
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    if value > 1.0:
        raise ValueError(f"{name} must not exceed 1.0, got {value}")
    return value


def _normalize_scale(value, name):
    value = float(value)
    if value > 10.0:
        value /= 100.0
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _range_spec(min_value, max_value):
    return {
        "mode": "uniform" if min_value != max_value else "constant",
        "min": min_value,
        "max": max_value,
        "value": min_value,
    }


def default_scenario():
    return {
        "name": "nominal",
        "oxidizer_fraction": {"mode": "constant", "value": 1.0},
        "pressure_scale": {"mode": "constant", "value": 1.0},
        "rocket_mass_scale": {"mode": "constant", "value": 1.0},
        "drag_multiplier": {"mode": "constant", "value": 1.0},
        "legacy_propellant_scaling": False,
        "write_metadata": True,
    }


def load_scenario(path):
    scenario = default_scenario()
    if path is None:
        return scenario

    with open(path, encoding="utf-8") as file:
        loaded = json.load(file)

    scenario.update(loaded)
    if "oxidizer_fraction" not in loaded and "propellant_fraction" in loaded:
        scenario["oxidizer_fraction"] = loaded["propellant_fraction"]
        scenario["legacy_propellant_scaling"] = True
    if "rocket_mass_scale" not in loaded and "mass_scale" in loaded:
        scenario["rocket_mass_scale"] = loaded["mass_scale"]
    return scenario


def sample_scenario_value(spec, rng, name, normalize=_normalize_fraction):
    if isinstance(spec, (int, float)):
        return normalize(spec, name)
    if not isinstance(spec, dict):
        raise ValueError(f"{name} scenario entry must be a number or object, got {spec!r}")

    mode = spec.get("mode", "constant")
    if mode == "constant":
        return normalize(spec.get("value", spec.get("min", 1.0)), name)
    if mode == "uniform":
        min_value = normalize(spec["min"], name)
        max_value = normalize(spec["max"], name)
        if min_value > max_value:
            raise ValueError(
                f"{name} uniform range must have min <= max, got {min_value} > {max_value}"
            )
        if "step" in spec:
            step = normalize(spec["step"], f"{name}.step")
            if step <= 0.0:
                raise ValueError(f"{name} step must be positive, got {step}")
            min_index = int(np.ceil((min_value - 1e-12) / step))
            max_index = int(np.floor((max_value + 1e-12) / step))
            if min_index > max_index:
                raise ValueError(
                    f"{name} step {step} leaves no values in range {min_value} - {max_value}"
                )
            return round(float(rng.integers(min_index, max_index + 1) * step), 10)
        return float(rng.uniform(min_value, max_value))
    if mode == "choice":
        choices = [normalize(value, name) for value in spec["values"]]
        return float(rng.choice(choices))
    if mode == "mixture":
        components = spec["components"]
        if not components:
            raise ValueError(f"{name} mixture must contain at least one component")
        weights = np.asarray(
            [float(component.get("weight", 1.0)) for component in components],
            dtype=np.float64,
        )
        if np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise ValueError(f"{name} mixture weights must be non-negative and not all zero")
        selected = int(rng.choice(len(components), p=weights / weights.sum()))
        component = {
            key: value for key, value in components[selected].items() if key != "weight"
        }
        return sample_scenario_value(component, rng, name, normalize)
    raise ValueError(f"unknown scenario mode for {name}: {mode!r}")


def load_scaled_thrust_curve(
    thrust_path,
    pressure_scale,
    burn_time_scale=1.0,
    source_pressure_scale=1.0,
):
    thrust_path = _resolve_path(thrust_path)
    curve = np.loadtxt(thrust_path, delimiter=",", dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError(f"thrust curve must have two columns: {thrust_path}")
    if not np.all(np.isfinite(curve)):
        raise ValueError(f"thrust curve contains non-finite values: {thrust_path}")
    if np.any(np.diff(curve[:, 0]) <= 0):
        raise ValueError(f"thrust curve timestamps must be strictly increasing: {thrust_path}")
    if pressure_scale <= 0 or burn_time_scale <= 0 or source_pressure_scale <= 0:
        raise ValueError("pressure, source-pressure, and burn-time scales must be positive")

    scaled = curve.copy()
    scaled[:, 0] = curve[0, 0] + (curve[:, 0] - curve[0, 0]) * float(burn_time_scale)
    # A source curve may already represent reduced chamber pressure, as does
    # the FAR-OUT 2026 48-bar curve. Scale relative to that source condition.
    scaled[:, 1] *= float(pressure_scale) / float(source_pressure_scale)
    return scaled


def load_scaled_drag_curve(drag_path, drag_multiplier=1.0):
    """Return a RocketPy drag function scaled by a latent aerodynamic-loss factor."""
    drag_path = _resolve_path(drag_path)
    curve = np.loadtxt(drag_path, delimiter=",", dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError(f"drag curve must have two columns: {drag_path}")
    if not np.all(np.isfinite(curve)):
        raise ValueError(f"drag curve contains non-finite values: {drag_path}")
    if drag_multiplier <= 0.0:
        raise ValueError(f"drag_multiplier must be positive, got {drag_multiplier}")

    scaled = curve.copy()
    scaled[:, 1] *= float(drag_multiplier)
    return Function(
        scaled,
        inputs="Mach Number",
        outputs="Drag Coefficient",
        interpolation="linear",
        extrapolation="constant",
    )


def _motor_propellant_mass(motor_data):
    grain_area = np.pi * (
        motor_data["grain_outer_radius"] ** 2
        - motor_data["grain_initial_inner_radius"] ** 2
    )
    return (
        motor_data["grain_number"]
        * grain_area
        * motor_data["grain_initial_height"]
        * motor_data["grain_density"]
    )


def apply_oxidizer_model(model_data, scenario, oxidizer_fraction):
    """Apply a first-order hybrid-motor surrogate without changing motor type.

    The SolidMotor represents only the fraction of fuel and oxidizer expected
    to be consumed. Unburned paraffin stays attached to the dry rocket mass.
    Oxidizer availability controls burn duration, while chamber-pressure scale
    controls thrust magnitude separately.
    """

    motor_data = model_data["motors"]
    base_propellant_mass = _motor_propellant_mass(motor_data)
    config = scenario.get("oxidizer_model")

    if config is None:
        if oxidizer_fraction != 1.0 and not scenario.get("legacy_propellant_scaling"):
            raise ValueError(
                "non-nominal oxidizer_fraction requires an oxidizer_model configuration"
            )
        # Preserve the old scenario behavior for reproducibility.
        motor_data["grain_initial_height"] *= oxidizer_fraction
        return {
            "burn_time_scale": 1.0,
            "nominal_oxidizer_mass_kg": float("nan"),
            "oxidizer_mass_kg": float("nan"),
            "fixed_fuel_mass_kg": float("nan"),
            "unburned_fuel_mass_kg": 0.0,
            "effective_propellant_mass_kg": base_propellant_mass * oxidizer_fraction,
            "legacy_propellant_scaling": True,
        }

    nominal_oxidizer_mass = float(config["nominal_oxidizer_mass_kg"])
    fixed_fuel_mass = float(config["fixed_fuel_mass_kg"])
    burn_time_exponent = float(config.get("burn_time_exponent", 1.0))
    if nominal_oxidizer_mass <= 0 or fixed_fuel_mass < 0:
        raise ValueError("oxidizer and fixed-fuel masses must be physically valid")
    if burn_time_exponent <= 0:
        raise ValueError("burn_time_exponent must be positive")

    configured_propellant_mass = nominal_oxidizer_mass + fixed_fuel_mass
    if not np.isclose(configured_propellant_mass, base_propellant_mass, rtol=0.02):
        raise ValueError(
            "oxidizer_model masses do not match the source motor propellant mass: "
            f"configured={configured_propellant_mass:.6f} kg, "
            f"source={base_propellant_mass:.6f} kg"
        )

    parameter_expectations = {
        "nominal_oxidizer_mass_kg": nominal_oxidizer_mass,
        "fixed_fuel_mass_kg": fixed_fuel_mass,
        "oxidizer_burn_time_exponent": burn_time_exponent,
    }
    for key, expected in parameter_expectations.items():
        configured = motor_data.get(key)
        if configured is not None and not np.isclose(float(configured), expected):
            raise ValueError(
                f"scenario {key}={expected} does not match motor parameters {configured}"
            )

    oxidizer_mass = nominal_oxidizer_mass * oxidizer_fraction
    consumed_fuel_mass = fixed_fuel_mass * oxidizer_fraction
    effective_propellant_mass = oxidizer_mass + consumed_fuel_mass
    unburned_fuel_mass = fixed_fuel_mass - consumed_fuel_mass

    motor_data["grain_initial_height"] *= effective_propellant_mass / base_propellant_mass
    model_data["rocket"]["mass"] += unburned_fuel_mass

    return {
        "burn_time_scale": oxidizer_fraction**burn_time_exponent,
        "nominal_oxidizer_mass_kg": nominal_oxidizer_mass,
        "oxidizer_mass_kg": oxidizer_mass,
        "fixed_fuel_mass_kg": fixed_fuel_mass,
        "unburned_fuel_mass_kg": unburned_fuel_mass,
        "effective_propellant_mass_kg": effective_propellant_mass,
        "legacy_propellant_scaling": False,
    }


def sample_scenario(scenario, rng):
    oxidizer_fraction = sample_scenario_value(
        scenario.get("oxidizer_fraction", 1.0), rng, "oxidizer_fraction"
    )
    pressure_scale = sample_scenario_value(
        scenario.get("pressure_scale", 1.0), rng, "pressure_scale", _normalize_scale
    )
    rocket_mass_scale = sample_scenario_value(
        scenario.get("rocket_mass_scale", 1.0), rng, "rocket_mass_scale", _normalize_scale
    )
    drag_multiplier = sample_scenario_value(
        scenario.get("drag_multiplier", 1.0), rng, "drag_multiplier", _normalize_scale
    )
    return oxidizer_fraction, pressure_scale, rocket_mass_scale, drag_multiplier


def prefetch_weather_environments(date_table, environment_data):
    Log.print_info("pre-fetching weather data from api...")
    prefetched_envs = {}

    for current_date in tqdm.tqdm(date_table, desc="fetching weather data"):
        env_name = f"ENV_DATA_{current_date.strftime('%Y-%m-%d')}"

        env_base = None
        max_retries = 10

        for attempt in range(max_retries):
            try:
                env_base = get_enviroment_from_date(environment_data, current_date, env_name)
                break

            except Exception as e:
                error_msg = str(e)
                if "temporarily limited" in error_msg or "400 Client Error" in error_msg:
                    wait_time = random.uniform(5, 15) * (attempt + 1)
                    print(
                        f"\nAPI queue full. waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}..."
                    )
                    time.sleep(wait_time)
                else:
                    raise

        if env_base is None:
            raise RuntimeError(
                f"failed to fetch weather data for {current_date} after {max_retries} attempts."
            )

        prefetched_envs[current_date] = env_base
        time.sleep(2)

    Log.print_info("all weather data pre-fetched")
    return prefetched_envs

def parallel_generator(
    amount_in_parrallel,
    date_table,
    yearly_files,
    json_path,
    drag_path,
    environment_data,
    heading,
    rail_length,
    sensor_list,  
    thrust_path,
    source_pressure_scale,
    scenario,
    stochastic_motor_params,
    acceleration_thresholds,
    angular_velocity_thresholds,
):
    N = len(date_table)
    indices = range(N)

    with open(json_path, encoding="utf-8") as file:
        model_data = json.load(file)

    def worker(i):
        current_date = date_table[i]
        oxidizer_fraction = pressure_scale = rocket_mass_scale = drag_multiplier = None
        try:
            if hasattr(cp, "cuda"):
                gpu_count = cp.cuda.runtime.getDeviceCount()
                if gpu_count > 0:
                    gpu_id = i % gpu_count
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    cp.cuda.Device(gpu_id).use()

            # profiling
            # profiler = cProfile.Profile()
            # profiler.enable()


            # mała uwaga do wiktora z przyszłości zastanow sie nad nadpisywaniem jsona i hard kopiowaniem go 
            # dorbanoc 
            np.random.seed(i)
            
            rng = np.random.default_rng(i)

            oxidizer_fraction, pressure_scale, rocket_mass_scale, drag_multiplier = (
                sample_scenario(scenario, rng)
            )
            
            #dla osob zastanawiajacych sie czemu kopia
            #każdy proces musi mieć osobny plik json ponieważ go modyfikujemy
            #czy da sie to zrobić jako osobą zmienną? tak
            #czy powinniśmy tak zrobić, tbh nie wiem, ponieważ funkcje konfuguracyjne z pliku json beda wtedy troche nie ładne pozdraiwam
            temp_model_data = copy.deepcopy(model_data)

            temp_model_data["rocket"]["mass"] *= rocket_mass_scale
            temp_model_data["rocket"]["inertia"] = [
                value * rocket_mass_scale for value in temp_model_data["rocket"]["inertia"]
            ]
            oxidizer_metadata = apply_oxidizer_model(
                temp_model_data, scenario, oxidizer_fraction
            )
            scaled_thrust_curve = load_scaled_thrust_curve(
                thrust_path,
                pressure_scale,
                oxidizer_metadata["burn_time_scale"],
                source_pressure_scale,
            )
            scaled_drag_curve = load_scaled_drag_curve(drag_path, drag_multiplier)
            
            base_motor = init_base_motor_from_JSON(temp_model_data, scaled_thrust_curve)
            stochastic_motor = init_stochastic_motor(base_motor, stochastic_motor_params)

            stochastic_motor._set_stochastic(seed=i)
            sampled_motor = stochastic_motor.create_object()

            rocket = init_rocket_from_JSON(temp_model_data, scaled_drag_curve, sampled_motor)
            rocket = add_sensors_to_rocket(rocket, sensor_list)

            sl_file = yearly_files[current_date.year]["single"]
            pl_file = yearly_files[current_date.year]["levels"]
            env_base = get_enviroment_from_batched_file(
                environment_data, current_date, sl_file, pl_file
            )

            st_environment = StochasticEnvironment(environment=env_base)
            st_environment._set_stochastic(seed=i)
            environment = st_environment.create_object()
            rng = np.random.default_rng(i)

            scenario_metadata = {
                "Name": scenario.get("name", "unnamed"),
                "Oxidizer_Fraction": oxidizer_fraction,
                # Compatibility alias for the current model input schema.
                "Propellant_Fraction": oxidizer_fraction,
                "Pressure_Scale": pressure_scale,
                "Rocket_Mass_Scale": rocket_mass_scale,
                "Drag_Multiplier": drag_multiplier,
                "Rocket_Mass_Kg": temp_model_data["rocket"]["mass"],
                "Effective_Propellant_Mass_Kg": oxidizer_metadata[
                    "effective_propellant_mass_kg"
                ],
                "Burn_Time_Scale": oxidizer_metadata["burn_time_scale"],
                "Base_Thrust_Source": str(thrust_path),
                "Base_Thrust_Pressure_Scale": source_pressure_scale,
            }
            if not oxidizer_metadata["legacy_propellant_scaling"]:
                scenario_metadata.update(
                    {
                        "Nominal_Oxidizer_Mass_Kg": oxidizer_metadata[
                            "nominal_oxidizer_mass_kg"
                        ],
                        "Oxidizer_Mass_Kg": oxidizer_metadata["oxidizer_mass_kg"],
                        "Fixed_Fuel_Mass_Kg": oxidizer_metadata[
                            "fixed_fuel_mass_kg"
                        ],
                        "Unburned_Fuel_Mass_Kg": oxidizer_metadata[
                            "unburned_fuel_mass_kg"
                        ],
                    }
                )

            run_single_simulation(
                current_date,
                rocket,
                environment,
                heading,
                rail_length,
                rng,
                acceleration_thresholds,
                angular_velocity_thresholds,
                i,
                scenario_metadata=(
                    scenario_metadata if scenario.get("write_metadata", True) else None
                ),
            )
            return current_date

        except Exception as e:
            scenario_context = (
                f"oxidizer={oxidizer_fraction}, pressure={pressure_scale}, "
                f"mass_scale={rocket_mass_scale}, drag_multiplier={drag_multiplier}"
            )
            Log.print_error(
                f"simulation failed for {current_date} ({scenario_context}): {e}\n"
                f"{traceback.format_exc()}"
            )
            return None
        # profiling
        # profiler.disable()
        # profiler.dump_stats(f"output/worker_b{i}_profile.prof")

    with ProcessPool(nodes=amount_in_parrallel) as pool:
        results = list(
            tqdm.tqdm(pool.uimap(worker, indices), total=N, desc="Siupi dupi Grzesiu dawaj")
        )
    return results


def main():
    start_time = time.perf_counter()
    global TEST_FLAG
    TEST_FLAG = False

    year_start = 1940
    year_end = 1941

    oxidizer_min = 1.0
    oxidizer_max = 1.0
    oxidizer_explicit = False
    paths_file = "paths.json"
    generation_mode = "nominal"
    scenario_file = None

    args = sys.argv[1:]

    if "test" in args:
        TEST_FLAG = True
        Log.print_warning("running in test mode")
        args.remove("test")

    if "--competition-day" in args:
        generation_mode = "far_out_26_competition_day"
        paths_file = "paths_farout_26.json"
        scenario_file = "scenarios/farout_26.json"
        args.remove("--competition-day")

    if "--scenario" in args:
        idx = args.index("--scenario")
        scenario_file = args[idx + 1]
        generation_mode = Path(scenario_file).stem
        args.pop(idx + 1)
        args.remove("--scenario")

    if "--paths" in args:
        idx = args.index("--paths")
        paths_file = args[idx + 1]
        args.pop(idx + 1)
        args.remove("--paths")

    # --fuel remains as a compatibility alias for --oxidizer.
    if "--oxidizer" in args and "--fuel" in args:
        raise ValueError("use either --oxidizer or the legacy --fuel alias, not both")
    oxidizer_option = (
        "--oxidizer"
        if "--oxidizer" in args
        else "--fuel"
        if "--fuel" in args
        else None
    )
    if oxidizer_option:
        oxidizer_explicit = True
        idx = args.index(oxidizer_option)
        oxidizer_value = args[idx + 1]
        if "-" in oxidizer_value:
            minimum, maximum = oxidizer_value.split("-")
            oxidizer_min = float(minimum) / 100.0
            oxidizer_max = float(maximum) / 100.0
        else:
            oxidizer_min = float(oxidizer_value)
            if oxidizer_min > 1.0:
                oxidizer_min /= 100.0
            oxidizer_max = oxidizer_min
        args.pop(idx + 1)
        args.remove(oxidizer_option)
        Log.print_info(
            f"oxidizer level is set to: {oxidizer_min * 100}% - "
            f"{oxidizer_max * 100}%"
        )

    scenario = load_scenario(scenario_file)
    if oxidizer_explicit:
        scenario["oxidizer_fraction"] = _range_spec(oxidizer_min, oxidizer_max)
        if "oxidizer_model" not in scenario:
            # Preserve the historical command-line behavior for nominal and
            # legacy scenarios. Oxidizer-aware generation should use the
            # dedicated robustness scenario instead.
            scenario["legacy_propellant_scaling"] = True
            Log.print_warning(
                f"{oxidizer_option} is using legacy generic propellant scaling; "
                "select an oxidizer-aware scenario for the hybrid-motor surrogate"
            )

    try:
        if len(args) >= 2:
            year_start = int(args[0])
            year_end = int(args[1])
        elif len(args) == 1:
            year_start = int(args[0])
            year_end = int(args[0])
        else:
            print("no specified dates")
    except ValueError:
        print("no year arguments")

    print(f"generating simulations from {year_start} to {year_end}...")
    Log.print_info(f"generation mode: {generation_mode}")
    Log.print_info(f"paths file: {paths_file}")
    Log.print_info(f"scenario: {scenario.get('name', 'unnamed')}")

    paths = init_paths_from_json(paths_file)
    environment_data = get_environment_data_from_JSON(paths["config_path"])

    sensor_list = []
    sensor_list.append(
        init_accelerometer_from_JSON(paths["sensors_path"]["accelerometer"], "LSM6DSOX_acc_2g")
    )
    sensor_list.append(
        init_accelerometer_from_JSON(paths["sensors_path"]["accelerometer"], "LSM6DSOX_acc_4g")
    )
    sensor_list.append(
        init_accelerometer_from_JSON(paths["sensors_path"]["accelerometer"], "LSM6DSOX_acc_8g")
    )
    sensor_list.append(
        init_accelerometer_from_JSON(paths["sensors_path"]["accelerometer"], "LSM6DSOX_acc_16g")
    )
    sensor_list.append(
        init_gyroscope_from_JSON(paths["sensors_path"]["gyroscope"], "LSM6DSOX_gyro_125dps")
    )
    sensor_list.append(
        init_gyroscope_from_JSON(paths["sensors_path"]["gyroscope"], "LSM6DSOX_gyro_250dps")
    )
    sensor_list.append(
        init_gyroscope_from_JSON(paths["sensors_path"]["gyroscope"], "LSM6DSOX_gyro_500dps")
    )
    sensor_list.append(
        init_gyroscope_from_JSON(paths["sensors_path"]["gyroscope"], "LSM6DSOX_gyro_1000dps")
    )
    sensor_list.append(
        init_gyroscope_from_JSON(paths["sensors_path"]["gyroscope"], "LSM6DSOX_gyro_2000dps")
    )
    sensor_list.append(
        init_barometer_from_JSON(paths["sensors_path"]["barometer"], "BME280_barometer")
    )
    sensor_list.append(
        init_thermometer_from_JSON(paths["sensors_path"]["thermometer"], "DS18B20_thermometer")
    )

    heading, rail_length = init_flight_config_from_JSON(paths["config_path"])

    stochastic_motor_params = init_stochastic_motor_params(paths["config_path"])

    target_sampling_rate = 500
    for sensor in sensor_list:
        if hasattr(sensor, "sampling_rate") and sensor.sampling_rate >= target_sampling_rate:
            sensor.sampling_rate = target_sampling_rate

    with open(paths["config_path"]) as file:
        config_data = json.load(file)

    acceleration_thresholds = config_data["thresholds"]["acceleration"]
    angular_velocity_thresholds = config_data["thresholds"]["angular_velocity"]

    date_table = []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            _, num_days = calendar.monthrange(year, month)
            random_days = random.sample(range(1, num_days + 1), 2)

            for day in random_days:
                date_table.append(datetime.datetime(year, month, day, 12, 0, 0))

    date_table.sort()

    yearly_files = {}
    for year in range(year_start, year_end + 1):
        sl_file, pl_file = download_yearly_weather(
            year, environment_data["longitude"], environment_data["latitude"]
        )
        yearly_files[year] = {"single": sl_file, "levels": pl_file}
    amount_in_parrallel = 12

    source_model_paths = paths["source_model_path"]
    thrust_path = select_thrust_path(source_model_paths)

    results = parallel_generator(
        amount_in_parrallel=amount_in_parrallel,
        date_table=date_table,
        yearly_files=yearly_files,
        json_path=paths["source_model_path"]["parameters"],
        drag_path=paths["source_model_path"]["drag_curve"],
        environment_data=environment_data,
        heading=heading,
        rail_length=rail_length,
        sensor_list=sensor_list,
        thrust_path=thrust_path,
        source_pressure_scale=float(
            paths["source_model_path"].get("thrust_source_pressure_scale", 1.0)
        ),
        scenario=scenario,
        stochastic_motor_params=stochastic_motor_params,
        acceleration_thresholds=acceleration_thresholds,
        angular_velocity_thresholds=angular_velocity_thresholds,
    )

    successful_dates = [d for d in results if d is not None]

    log_file = "output/generated_flights.txt"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- batch Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"total attempted: {len(date_table)} | successful: {len(successful_dates)}\n")
        f.write(f"mode: {generation_mode} | scenario: {scenario.get('name', 'unnamed')}\n")
        for date in successful_dates:
            f.write(f"{date.strftime('%Y-%m-%d')}\n")

    print(f"Logged {len(successful_dates)} successful simulation dates to {log_file}")

    end_time = time.perf_counter()
    total_seconds = end_time - start_time
    formatted_time = str(datetime.timedelta(seconds=int(total_seconds)))
    print(f"total processing time: {formatted_time}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
