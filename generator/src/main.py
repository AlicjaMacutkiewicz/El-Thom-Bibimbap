import calendar
import datetime
import json
import os
import copy
import random
import sys

# import cProfile
import time

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
from rocketpy import Accelerometer, Barometer, Environment, Gyroscope, Rocket, SolidMotor
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


def init_rocket_from_JSON(data, drag_curve_csv, motor):
    name = data["id"]["rocket_name"]

    Log.print_info(f"Loading model: {name}")

    rocket_data = data["rocket"]
    Log.print_info("loading rocket")
    rocket = Rocket(
        radius=rocket_data["radius"],
        mass=rocket_data["mass"],
        inertia=tuple(rocket_data["inertia"]),
        power_off_drag=drag_curve_csv,
        power_on_drag=drag_curve_csv,
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

    for p_id in parachutes:
        p_data = parachutes[p_id]
        p_trigger = "apogee" if p_data["deploy_event"] == "apogee" else p_data["deploy_altitude"]

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



#for those who come afer
#te zmiany pozwalaja wytrenowac ten model ale oglnie to powinno byc wykonane znacznie bardziej etycznie i programistycznie
#postaram sie to zrobic jutro ale to jest kwestia zmienienia kilku znaczacych rzeczy w grzesiu


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
    thrust_path_100, #to rozwiazanie jest lopatologiczne ale no mamy tylko dwa pliki csv jeden dla normalnego tankowania drugi dla tego na konkursie
    thrust_path_60,  #todo zmienic to na mape klucz -> stopien zatankowania, wartosc -> plik csv
    prob_full_fuel,  #zmiany ktore prowadziłem wczeniej (fuel_min, fuel_max) sa teraz useless ale przy roziazaniu bardziej "clean", beda uzyteczne dlatego poki co sa w kodzie
    stochastic_motor_params,
    acceleration_thresholds,
    angular_velocity_thresholds,
    fuel_min,
    fuel_max
):
    N = len(date_table)
    indices = range(N)

    with open(json_path, encoding="utf-8") as file:
        model_data = json.load(file)

    def worker(i):
        current_date = date_table[i]
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

            fuel_fraction = rng.uniform(fuel_min, fuel_max)
            thrust_path = thrust_path_100 if fuel_fraction >= 0.999 else thrust_path_60
            
            #dla osob zastanawiajacych sie czemu kopia
            #każdy proces musi mieć osobny plik json ponieważ go modyfikujemy
            #czy da sie to zrobić jako osobą zmienną? tak
            #czy powinniśmy tak zrobić, tbh nie wiem, ponieważ funkcje konfuguracyjne z pliku json beda wtedy troche nie ładne pozdraiwam
            temp_model_data = copy.deepcopy(model_data)

            temp_model_data["motors"]["grain_initial_height"] *= fuel_fraction
            
            base_motor = init_base_motor_from_JSON(temp_model_data, thrust_path)
            stochastic_motor = init_stochastic_motor(base_motor, stochastic_motor_params)

            sampled_motor = stochastic_motor.create_object()
            stochastic_motor._set_stochastic(seed=i)

            rocket = init_rocket_from_JSON(temp_model_data, drag_path, sampled_motor)
            rocket = add_sensors_to_rocket(rocket, sensor_list)

            sl_file = yearly_files[current_date.year]["single"]
            pl_file = yearly_files[current_date.year]["levels"]
            env_base = get_enviroment_from_batched_file(
                environment_data, current_date, sl_file, pl_file
            )

            st_environment = StochasticEnvironment(environment=env_base)
            environment = st_environment.create_object()
            rng = np.random.default_rng(i)

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
            )
            return current_date

        except Exception as e:
            Log.print_error(f"simulation failed for {current_date}: {e}")
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

    fuel_min = 1.0
    fuel_max = 1.0
    fuel_explicit = False
    paths_file = "paths.json"
    generation_mode = "nominal"

    args = sys.argv[1:]

    if "test" in args:
        TEST_FLAG = True
        Log.print_warning("running in test mode")
        args.remove("test")

    if "--competition-day" in args:
        generation_mode = "far_out_26_competition_day"
        paths_file = "paths_farout_26.json"
        args.remove("--competition-day")

    if "--paths" in args:
        idx = args.index("--paths")
        paths_file = args[idx + 1]
        args.pop(idx + 1)
        args.remove("--paths")

    # Fuel option examples :
    # --fuel 60-70 (generates flights with randomly selected fuel percentage in this case beatwean 60% and 70%)
    # --fuel 67 (generates flights with strict 60% fuel val)

    if "--fuel" in args:
        fuel_explicit = True
        idx = args.index("--fuel")
        fuel_val = args[idx + 1]
        if "-" in fuel_val:
            f_min, f_max = fuel_val.split("-")
            fuel_min , fuel_max = float(f_min) / 100.0, float(f_max) / 100.0
        else:
            fuel_min = float(fuel_val)
            if fuel_min > 1.0:
                fuel_min /= 100.0
            fuel_max = fuel_min
        args.pop(idx+1)
        args.remove("--fuel")
        Log.print_info(f"fuel level is set to: {fuel_min*100}% - {fuel_max*100}%")

    if generation_mode == "far_out_26_competition_day" and not fuel_explicit:
        # Proxy for the measured FAR-OUT 26 oxidizer load: 5.5 kg against the
        # nominal 12 kg propellant-equivalent scale used by this generator.
        fuel_min = fuel_max = 5.5 / 12.0
        Log.print_info(
            "competition-day mode: fuel/oxidizer proxy defaults to "
            f"{fuel_min * 100:.3f}%"
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
        thrust_path_100=paths["source_model_path"]["thrust_source_100"],
        thrust_path_60=paths["source_model_path"]["thrust_source_60"],
        prob_full_fuel=0.6,
        stochastic_motor_params=stochastic_motor_params,
        acceleration_thresholds=acceleration_thresholds,
        angular_velocity_thresholds=angular_velocity_thresholds,
        fuel_min = fuel_min,
        fuel_max = fuel_max
    )

    successful_dates = [d for d in results if d is not None]

    log_file = "output/generated_flights.txt"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- batch Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"total attempted: {len(date_table)} | successful: {len(successful_dates)}\n")
        for date in successful_dates:
            f.write(f"{date.strftime('%Y-%m-%d')}\n")

    print(f"Logged {len(successful_dates)} successful simulation dates to {log_file}")

    end_time = time.perf_counter()
    total_seconds = end_time - start_time
    formatted_time = str(datetime.timedelta(seconds=int(total_seconds)))
    print(f"total processing time: {formatted_time}")


#Again moze nawet dziś to zrobie ale chce żeby osoby czytające ten commit rozumialy o co mi chodzi 
#Zalecenia fryderyka fazbera:
#problem jest taki że dla danego poziomu paliwa wypada mieć no csvke z ciągiem 
#aktulanie (z tego co wiem) mamy 2 takie pliki jeden normalny a drugi dla niezatankowanego (60%)
#no i teraz mamy hardcoded dwie sciezki (thrust_path_100,60)
#celem tej zmiany jest just odpalenie grzegora sprawdzenie czy sama idea dziala
#nie zmiania to fakut ze zrobie z tego clean solution ktore bedzie zakladac dynamiczne dodawanie sciezek z csv 
#i generator bedzie je dopieral na podstawie zakresu ktory pwrowadzilem wczenisej
#(fuel_min , fuel_max)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
