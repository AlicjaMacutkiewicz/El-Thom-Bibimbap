import numpy as np
import datetime
import json
import os
import sys

import tqdm
from logger import *
from pathos.multiprocessing import ProcessPool
from rocketpy import Environment, SolidMotor, Rocket, Accelerometer, Gyroscope, GnssReceiver
from rocketpy.stochastic import StochasticSolidMotor

from single_simulation import run_single_simulation


# @BRIEF
# Initializes rocket with data found in give JSON file
# @ARGUMENTS: 
#     path_to_file -> STRING representing path to JSON file
#     drag_curve_csv -> STRING representing path to CSV file
#     thrust_source_csv -> STRING representing path to CSV file
# @RETURN
#     rocket -> Initialized rocketpy::Rocket class 

def init_rocket_from_JSON(path_to_file, drag_curve_csv, motor):
    dir = os.path.dirname(__file__)
    file_name = os.path.join(dir, path_to_file)
    with open(file_name, 'r', encoding='utf-8')as file:
        data= json.load(file)
    Log.print_info(f"Reading from {file_name}")

    name = data['id']['rocket_name']

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
        coordinate_system_orientation=rocket_data["coordinate_system_orientation"]
                )
    motor_data = data["motors"]
    rocket.add_motor(motor, position=motor_data["position"])
    Log.print_info("loading nose")
    nose_data = data["nosecones"]
    rocket.add_nose(
            length=nose_data["length"],
            kind=nose_data["kind"],
            position=nose_data["position"]
            )
    Log.print_info("loading fins")
    rocket.add_trapezoidal_fins(
            n=3,
            root_chord=0.12,
            tip_chord=0.06,
            span=0.10,
            position=1.0, 
            sweep_length=0.05
            )

    parachute_data = data["parachutes"]["0"]
    Log.print_info("loading parachute")
    rocket.add_parachute(
            name=parachute_data["name"],
            cd_s=parachute_data["cds"],
            trigger=parachute_data["deploy_event"],
            lag=parachute_data["deploy_delay"]
            )
    return rocket

def init_base_motor_from_JSON(path_to_file, thrust_source_csv):
    dir = os.path.dirname(__file__)
    file_name = os.path.join(dir, path_to_file)
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)

    Log.print_info(f"Reading from {file_name}")
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
       # burn_time=3.0,
        throat_radius=motor_data["throat_radius"],
        coordinate_system_orientation=motor_data["coordinate_system_orientation"]
    )

    return motor

def init_stochastic_motor(base_motor, stochastic_motor_params):
    (grain_density_param, grain_outer_radius_param, 
     grain_initial_inner_radius_param, grain_initial_height_param, 
     nozzle_radius_param, throat_radius_param, total_impulse_param) = stochastic_motor_params
    stochastic_motor = StochasticSolidMotor(
        solid_motor=base_motor,
        grain_density=grain_density_param * base_motor.grain_density,
        grain_outer_radius=grain_outer_radius_param* base_motor.grain_outer_radius,
        grain_initial_inner_radius=grain_initial_inner_radius_param* base_motor.grain_initial_inner_radius,
        grain_initial_height=grain_initial_height_param* base_motor.grain_initial_height,
        nozzle_radius=nozzle_radius_param * base_motor.nozzle_radius,
        throat_radius=throat_radius_param * base_motor.throat_radius,

        # opcjonalnie, ale można dodać szum do impulsu całkowitego, co może być bardziej realistyczne niż szum w poszczególnych parametrach
        total_impulse=total_impulse_param * base_motor.total_impulse
    )
    return stochastic_motor

def get_environment_data_from_JSON(path_to_file):
    dir = os.path.dirname(__file__)
    file_name = os.path.join(dir, path_to_file)
    with open(file_name, 'r', encoding='utf-8')as file:
        data= json.load(file)
    Log.print_info(f"Reading from {file_name}")
    env_data = data["environment"]
    return env_data

def init_environment_from_JSON(path_to_file):
    dir = os.path.dirname(__file__)
    file_name = os.path.join(dir, path_to_file)
    with open(file_name, 'r', encoding='utf-8')as file:
        data= json.load(file)
    Log.print_info(f"Reading from {file_name}")
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

def init_flight_config_from_JSON(path_to_file):
    dir = os.path.dirname(__file__)
    file_name = os.path.join(dir, path_to_file)
    with open(file_name, 'r', encoding='utf-8')as file:
        data= json.load(file)
    Log.print_info(f"Reading from {file_name}")
    flight_data = data["flight"]
    heading=flight_data["heading"]
    rail_length=flight_data["rail_length"]
    return (heading ,  rail_length)

def init_accelerometer_from_JSON(path_to_file, name):
    dir = os.path.dirname(__file__)
    file_name = os.path.join(dir, path_to_file)
    with open(file_name, 'r', encoding='utf-8')as file:
        data= json.load(file)
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
        orientation=accel_data["orientation"]
    )
    return accelerometer

def init_gyroscope_from_JSON(path_to_file, name):
    dir = os.path.dirname(__file__)
    file_name = os.path.join(dir, path_to_file)
    with open(file_name, 'r', encoding='utf-8')as file:
        data= json.load(file)
    gyro_data = data[name]
    gyroscope = Gyroscope(
        name= gyro_data["name"],
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
        orientation=gyro_data["orientation"]
    )
    return gyroscope

def init_gnss_from_JSON(path_to_file, name):
    dir = os.path.dirname(__file__)
    file_name = os.path.join(dir, path_to_file)
    with open(file_name, 'r', encoding='utf-8')as file:
        data= json.load(file)
    gnss_data = data[name]
    gnss = GnssReceiver(
        name="GNSS",
        sampling_rate=gnss_data["sampling_rate"],
        position_accuracy=gnss_data["position_accuracy"],
        altitude_accuracy=gnss_data["altitude_accuracy"]
    )
    return gnss

def add_gyro_to_rocket(rocket , gyro_list):
    gyro_list.sort(key= lambda x: x.measurement_range)
    for g in gyro_list:
        #TODO: replace 1 
        rocket.add_sensor(g , 1)
    return rocket

def add_acc_to_rocket(rocket , acc_list):
    acc_list.sort(key= lambda x: x.measurement_range)
    for a in acc_list:
        #TODO: replace 1 
        if(TEST_FLAG):
            a.sampling_rate /= 10
        rocket.add_sensor(a , 1)
    return rocket

def test_stochastic_motor(stochastic_motor):
        sampled_motor = stochastic_motor.create_object()
        print("grain_density =", sampled_motor.grain_density)

def init_stochastic_motor_params(path):
    dir = os.path.dirname(__file__)
    file_name = os.path.join(dir, path)
    with open(file_name, 'r', encoding='utf-8')as file:
        dataset= json.load(file)

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
    dir = os.path.dirname(__file__)
    file_name = os.path.join(dir, main_paths_file)
    with open(file_name, 'r', encoding='utf-8')as file:
        dataset= json.load(file)
    return dataset
    
def parallel_generator(N, json_path, drag_path, environment, heading , rail_length,acc_list,thrust_path,stochastic_motor_params):
    indices = range(N) 
    def worker(i):
        np.random.seed(i)
        base_motor = init_base_motor_from_JSON(json_path, thrust_path)
        stochastic_motor = init_stochastic_motor(base_motor,stochastic_motor_params)
        sampled_motor = stochastic_motor.create_object()
        stochastic_motor._set_stochastic(seed=i)

        rocket = init_rocket_from_JSON(json_path,drag_path,sampled_motor)
        rocket = add_acc_to_rocket(rocket, acc_list)
        test_stochastic_motor(stochastic_motor)
        return run_single_simulation(i, rocket, environment, heading, rail_length)
    

    with ProcessPool() as pool:
        results = list(tqdm.tqdm(pool.imap(worker, indices), total=N, desc="Siupi dupi Grzesiu dawaj"))

    #for parquet data saving and packing 
    # |
    # print("Pakowanko...")
    # master_df = pd.concat(results)
    # master_df.reset_index(inplace=True)
    # master_df.to_parquet("dataset_packed.parquet", index=True)
    # |
  
def main():
    global TEST_FLAG
    if(len(sys.argv) > 1):
        if sys.argv[1] == 'test':
            TEST_FLAG = True
            Log.print_warning("RUNNING IN TEST MODE")
    
    paths = init_paths_from_json("paths.json")
    environment = get_environment_data_from_JSON(paths["config_path"])
    
    acc_list = [] 
    acc_list.append(init_accelerometer_from_JSON(paths["sensors_path"]["accelerometer"],"LSM9DS1_acc_2g"))
    acc_list.append(init_accelerometer_from_JSON(paths["sensors_path"]["accelerometer"],"LSM9DS1_acc_4g"))
    acc_list.append(init_accelerometer_from_JSON(paths["sensors_path"]["accelerometer"],"LSM9DS1_acc_8g"))
    acc_list.append(init_accelerometer_from_JSON(paths["sensors_path"]["accelerometer"],"LSM9DS1_acc_16g"))

    acc_list.append(init_gyroscope_from_JSON(paths["sensors_path"]["gyroscope"],"LSM9DS1_gyro_245dps"))
    acc_list.append(init_gyroscope_from_JSON(paths["sensors_path"]["gyroscope"],"LSM9DS1_gyro_500dps"))
    acc_list.append(init_gyroscope_from_JSON(paths["sensors_path"]["gyroscope"],"LSM9DS1_gyro_2000dps"))

    acc_list.append(init_gnss_from_JSON(paths["sensors_path"]["gnss_velocity_heading"], "u-blox_MAX-M10S"))

    heading, rail_length = init_flight_config_from_JSON(paths["config_path"])
    
    stochastic_motor_params = init_stochastic_motor_params(paths["config_path"])
    
    parallel_generator(2,
                       paths["source_model_path"]["parameters"],
                       paths["source_model_path"]["drag_curve"],
                       environment,heading,
                       rail_length,
                       acc_list,
                       paths["source_model_path"]["thrust_source"],
                       stochastic_motor_params
                       )
    
if __name__=="__main__":
    main()
