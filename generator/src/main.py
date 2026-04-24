import datetime
import json
import sys
from enviroment_api import * 
from custom_sensors.thermometer import *
import numpy as np
import tqdm
from logger import *
from pathos.multiprocessing import ProcessPool
from rocketpy import Environment, SolidMotor, Rocket, Accelerometer, Gyroscope, Barometer
from rocketpy.stochastic import StochasticEnvironment, StochasticSolidMotor
from logger import *
from single_simulation import run_single_simulation
# import cProfile
import time
import datetime


NUM_GPUS = 0
try:
    import cupy as cp
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
    name = data['id']['rocket_name']

    Log.print_info(f"Loading model: {name}")
     
    rocket_data = data["rocket"]
    Log.print_info("loading rocket")
    rocket = Rocket(
                    radius = rocket_data["radius"],
                    mass = rocket_data["mass"],
                    inertia = tuple(rocket_data["inertia"]),
                    power_off_drag = drag_curve_csv,
                    power_on_drag = drag_curve_csv,
                    center_of_mass_without_motor = rocket_data["center_of_mass_without_propellant"],
                    coordinate_system_orientation = rocket_data["coordinate_system_orientation"]
                )
    motor_data = data["motors"]
    rocket.add_motor(motor, position=motor_data["position"])
    Log.print_info("loading nose")
    nose_data = data["nosecones"]
    rocket.add_nose(
                length = nose_data["length"],
                kind = nose_data["kind"],
                position = nose_data["position"],
                power = nose_data["power"]
            )
    Log.print_info("loading fins")
    fins = data["trapezoidal_fins"]["0"]
    rocket.add_trapezoidal_fins(
                n = fins["number"],
                root_chord = fins["root_chord"],
                tip_chord = fins["tip_chord"],
                span = fins["span"],
                position = fins["position"],
                sweep_length = fins["sweep_length"]
            )

    parachutes = data["parachutes"]

    for p_id in parachutes:
        p_data = parachutes[p_id]
        if p_data["deploy_event"] == "apogee":
            p_trigger = "apogee"
        else:
            p_trigger = p_data["deploy_altitude"]

        rocket.add_parachute(
            name=p_data["name"],
            cd_s=p_data["cds"],
            trigger=p_trigger,
            lag=p_data["deploy_delay"]
        )
    return rocket

def init_base_motor_from_JSON(data, thrust_source_csv):
    motor_data = data["motors"]

    Log.print_info("loading base motor")
    motor = SolidMotor(
        thrust_source = thrust_source_csv,
        dry_mass = motor_data["dry_mass"],
        dry_inertia = motor_data["dry_inertia"],
        nozzle_radius = motor_data["nozzle_radius"],
        grain_number = motor_data["grain_number"],
        grain_density = motor_data["grain_density"],
        grain_outer_radius = motor_data["grain_outer_radius"],
        grain_initial_inner_radius = motor_data["grain_initial_inner_radius"],
        grain_initial_height = motor_data["grain_initial_height"],
        grain_separation = motor_data["grain_separation"],
        grains_center_of_mass_position = motor_data["grains_center_of_mass_position"],
        center_of_dry_mass_position = motor_data["center_of_dry_mass_position"],
        nozzle_position = motor_data["nozzle_position"],
        # burn_time = 10.0, #declared based on in third static fire test
        throat_radius = motor_data["throat_radius"],
        coordinate_system_orientation = motor_data["coordinate_system_orientation"]
    )

    return motor

def init_stochastic_motor(base_motor, stochastic_motor_params):
    (grain_density_param, grain_outer_radius_param, 
     grain_initial_inner_radius_param, grain_initial_height_param, 
     nozzle_radius_param, throat_radius_param, total_impulse_param) = stochastic_motor_params

    stochastic_motor = StochasticSolidMotor(
        solid_motor = base_motor,
        grain_density = grain_density_param * base_motor.grain_density,
        grain_outer_radius = grain_outer_radius_param* base_motor.grain_outer_radius,
        grain_initial_inner_radius = grain_initial_inner_radius_param* base_motor.grain_initial_inner_radius,
        grain_initial_height = grain_initial_height_param* base_motor.grain_initial_height,
        nozzle_radius = nozzle_radius_param * base_motor.nozzle_radius,
        throat_radius = throat_radius_param * base_motor.throat_radius,
        total_impulse = total_impulse_param * base_motor.total_impulse
    )
    return stochastic_motor

def get_environment_data_from_JSON(path_to_file):
    with open(path_to_file, 'r', encoding = 'utf-8')as file:
        data = json.load(file)
    Log.print_info(f"Reading from {path_to_file}")
    env_data = data["environment"]
    return env_data

def init_environment_from_JSON(path_to_file):
    with open(path_to_file, 'r', encoding = 'utf-8')as file:
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
                timezone="America/New_York")
    return env

def init_stochastic_environment(env_data, date):
    env_base = get_enviroment_from_date(env_data , date , f"ENV_DATA_"+date.strftime("%Y/%m/%d_%H:%M:%S")) 
    env = StochasticEnvironment(env_base)
    return env

def init_flight_config_from_JSON(path_to_file):
    with open(path_to_file, 'r', encoding = 'utf-8')as file:
        data = json.load(file)
    Log.print_info(f"Reading from {path_to_file}")
    flight_data = data["flight"]
    heading = flight_data["heading"]
    rail_length = flight_data["rail_length"]
    return heading ,  rail_length

def init_accelerometer_from_JSON(path_to_file, name):
    with open(path_to_file, 'r', encoding = 'utf-8')as file:
        data = json.load(file)
    accel_data = data[name]
    accelerometer = Accelerometer(
        name = accel_data["name"],
        sampling_rate = accel_data["sampling_rate"],
        measurement_range = accel_data["measurement_range"],
        resolution = accel_data["resolution"],
        noise_density = accel_data["noise_density"],
        noise_variance = accel_data["noise_variance"],
        random_walk_density = accel_data["random_walk_density"],
        random_walk_variance = accel_data["random_walk_variance"],
        constant_bias = accel_data["constant_bias"],
        operating_temperature = accel_data["operating_temperature"],
        temperature_bias = accel_data["temperature_bias"],
        orientation = accel_data["orientation"]
    )
    return accelerometer

def init_gyroscope_from_JSON(path_to_file, name):
    with open(path_to_file, 'r', encoding = 'utf-8')as file:
        data = json.load(file)
    gyro_data = data[name]
    gyroscope = Gyroscope(
        name = gyro_data["name"],
        sampling_rate = gyro_data["sampling_rate"],
        measurement_range = gyro_data["measurement_range"],
        resolution = gyro_data["resolution"],
        noise_density = gyro_data["noise_density"],
        noise_variance = gyro_data["noise_variance"],
        random_walk_density = gyro_data["random_walk_density"],
        random_walk_variance = gyro_data["random_walk_variance"],
        constant_bias = gyro_data["constant_bias"],
        operating_temperature = gyro_data["operating_temperature"],
        temperature_bias = gyro_data["temperature_bias"],
        orientation = gyro_data["orientation"]
    )
    return gyroscope

def init_thermometer_from_JSON(path_to_file, name):
    with open(path_to_file, 'r', encoding = 'utf-8')as file:
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
    with open(path_to_file, 'r', encoding = 'utf-8')as file:
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

def add_sensors_to_rocket(rocket , acc_list):
    # acc_list.sort(key = lambda x: x.measurement_range)
    for a in acc_list:
        #TODO: replace 1 ?
        if TEST_FLAG:
            a.sampling_rate /= 100
        rocket.add_sensor(a , 1)
    return rocket

def init_stochastic_motor_params(path_to_file):
    with open(path_to_file, 'r', encoding = 'utf-8')as file:
        dataset = json.load(file)

    motor_data = dataset["stochastic_motor_params"]
    grain_density_param = motor_data["grain_density_param"]
    grain_outer_radius_param = motor_data["grain_outer_radius_param"]
    grain_initial_inner_radius_param = motor_data["grain_initial_inner_radius_param"]
    grain_initial_height_param = motor_data["grain_initial_height_param"]
    nozzle_radius_param = motor_data["nozzle_radius_param"]
    throat_radius_param = motor_data["throat_radius_param"]
    total_impulse_param = motor_data["total_impulse_param"]
    params = (grain_density_param, grain_outer_radius_param, grain_initial_inner_radius_param, grain_initial_height_param, nozzle_radius_param, throat_radius_param, total_impulse_param)
    return params

def init_paths_from_json(main_paths_file):
    with open(main_paths_file, 'r', encoding = 'utf-8')as file:
        dataset = json.load(file)
    return dataset

    
def parallel_generator(N, json_path, drag_path, env_base, heading , rail_length,sensor_list,thrust_path,stochastic_motor_params, acceleration_thresholds, angular_velocity_thresholds):
    indices = range(N) 
    with open(json_path, 'r', encoding='utf-8') as file:
        model_data = json.load(file)

    base_motor = init_base_motor_from_JSON(model_data, thrust_path)
    stochastic_motor = init_stochastic_motor(base_motor,stochastic_motor_params)

    def worker(i):
        if hasattr(cp, 'cuda'):
            gpu_count = cp.cuda.runtime.getDeviceCount()
            if gpu_count > 0:
                gpu_id = i % gpu_count
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                cp.cuda.Device(gpu_id).use()

        #profiling
        # profiler = cProfile.Profile()
        # profiler.enable()

        np.random.seed(i)
           
        sampled_motor = stochastic_motor.create_object()
        stochastic_motor._set_stochastic(seed = i)

        rocket = init_rocket_from_JSON(model_data,drag_path,sampled_motor)
        rocket = add_sensors_to_rocket(rocket, sensor_list)

        st_environment = StochasticEnvironment(environment=env_base)
        environment = st_environment.create_object()
        rng = np.random.default_rng(i)

        result =  run_single_simulation(i, rocket, environment, heading, rail_length, rng, acceleration_thresholds, angular_velocity_thresholds)

        #profiling
        # profiler.disable()
        # profiler.dump_stats(f"output/worker_b{i}_profile.prof") 
        
        return result
    
    with ProcessPool() as pool:
        results = list(tqdm.tqdm(pool.uimap(worker, indices), total = N, desc = "Siupi dupi Grzesiu dawaj"))
    return results


def main():
    start_time = time.perf_counter()
    global TEST_FLAG
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            TEST_FLAG = True
            Log.print_warning("RUNNING IN TEST MODE")
    
    paths = init_paths_from_json("paths.json")
    environment_data = get_environment_data_from_JSON(paths["config_path"])
    sensor_list = [] 
    sensor_list.append(init_accelerometer_from_JSON(paths["sensors_path"]["accelerometer"],"LSM6DSOX_acc_2g"))
    sensor_list.append(init_accelerometer_from_JSON(paths["sensors_path"]["accelerometer"],"LSM6DSOX_acc_4g"))
    sensor_list.append(init_accelerometer_from_JSON(paths["sensors_path"]["accelerometer"],"LSM6DSOX_acc_8g"))
    sensor_list.append(init_accelerometer_from_JSON(paths["sensors_path"]["accelerometer"],"LSM6DSOX_acc_16g"))
    sensor_list.append(init_gyroscope_from_JSON(paths["sensors_path"]["gyroscope"],"LSM6DSOX_gyro_125dps"))
    sensor_list.append(init_gyroscope_from_JSON(paths["sensors_path"]["gyroscope"],"LSM6DSOX_gyro_250dps"))
    sensor_list.append(init_gyroscope_from_JSON(paths["sensors_path"]["gyroscope"],"LSM6DSOX_gyro_500dps"))
    sensor_list.append(init_gyroscope_from_JSON(paths["sensors_path"]["gyroscope"],"LSM6DSOX_gyro_1000dps"))
    sensor_list.append(init_gyroscope_from_JSON(paths["sensors_path"]["gyroscope"],"LSM6DSOX_gyro_2000dps"))
    sensor_list.append(init_barometer_from_JSON(paths["sensors_path"]["barometer"],"BME280_barometer"))
    sensor_list.append(init_thermometer_from_JSON(paths["sensors_path"]["thermometer"],"DS18B20_thermometer"))

    heading, rail_length = init_flight_config_from_JSON(paths["config_path"])
    
    stochastic_motor_params = init_stochastic_motor_params(paths["config_path"])

    target_sampling_rate = 500
    for sensor in sensor_list:
        if hasattr(sensor, 'sampling_rate') and sensor.sampling_rate >= target_sampling_rate:
            sensor.sampling_rate = target_sampling_rate

    with open(paths["config_path"], 'r') as file:
        config_data = json.load(file)

    acceleration_thresholds = config_data["thresholds"]["acceleration"]
    angular_velocity_thresholds = config_data["thresholds"]["angular_velocity"]
    
    flight_simulation_amount_for_scenario = 10
    date  = datetime.datetime(2005 , 12 , 10)
    env_base = get_enviroment_from_date(environment_data, date, "ENV_DATA_"+date.strftime("%Y-%m-%d_%H:%M"))
    parallel_generator(flight_simulation_amount_for_scenario,
                       paths["source_model_path"]["parameters"],
                       paths["source_model_path"]["drag_curve"],
                       env_base,heading,
                       rail_length,
                       sensor_list,
                       paths["source_model_path"]["thrust_source"],
                       stochastic_motor_params,
                       acceleration_thresholds,
                       angular_velocity_thresholds
                       )
    end_time = time.perf_counter()
    total_seconds = end_time - start_time
    formatted_time = str(datetime.timedelta(seconds=int(total_seconds)))
    print(formatted_time)

if __name__=="__main__":
    main()
