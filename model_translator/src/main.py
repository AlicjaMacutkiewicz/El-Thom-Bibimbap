import matplotlib.pyplot as plt
import json
import pandas as pd
import tqdm
from enum import IntEnum
from rocketpy import Environment, SolidMotor, Rocket, Flight , Barometer , Accelerometer, Gyroscope
from single_simulation import run_single_simulation
from pathos.multiprocessing import ProcessPool

class LogLevel(IntEnum):
    INFO = 1
    WARNING = 2
    ERROR = 3
C_CURRENT_LOG_LEVEL = LogLevel.INFO

def print_error(msg):
    if C_CURRENT_LOG_LEVEL>= LogLevel.ERROR:
       print("ERROR: " + msg + '\n')
def print_info(msg):
    if C_CURRENT_LOG_LEVEL>= LogLevel.INFO:
        print("INFO: " + msg + '\n')
def print_warning(msg):
    if C_CURRENT_LOG_LEVEL>= LogLevel.WARNING:
        print("WARNING: " + msg + '\n')

# @BRIEF
# Initializes rocket with data found in give JSON file
# @ARGUMENTS: 
#     path_to_file -> STRING representing path to JSON file
#     drag_curve_csv -> STRING representing path to CSV file
#     thrust_source_csv -> STRING representing path to CSV file
# @RETURN
#     rocket -> Initialized rocketpy::Rocket class 

def init_rocket_from_JSON(path_to_file, drag_curve_csv, thrust_source_csv):
    with open(path_to_file, 'r', encoding='utf-8')as file:
        data= json.load(file)
    print_info(f"Reading from {path_to_file}")

    name = data['id']['rocket_name']

    print_info(f"Loading model: {name}")

    motor_data = data["motors"] 
     
    print_info("loading motor")
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
        burn_time=3.0, 
        throat_radius=motor_data["throat_radius"],
        coordinate_system_orientation=motor_data["coordinate_system_orientation"]
            ) 
    rocket_data = data["rocket"]
    print_info("loading rocket")
    rocket = Rocket(
        radius=rocket_data["radius"],
        mass=rocket_data["mass"],
        inertia=tuple(rocket_data["inertia"]),
        power_off_drag=drag_curve_csv,
        power_on_drag=drag_curve_csv,
        center_of_mass_without_motor=rocket_data["center_of_mass_without_propellant"],
        coordinate_system_orientation=rocket_data["coordinate_system_orientation"]
                )
    rocket.add_motor(motor, position=motor_data["position"])
    print_info("loading nose")
    nose_data = data["nosecones"]
    rocket.add_nose(
            length=nose_data["length"],
            kind=nose_data["kind"],
            position=nose_data["position"]
            )
    print_info("loading fins")
    rocket.add_trapezoidal_fins(
            n=3,
            root_chord=0.12,
            tip_chord=0.06,
            span=0.10,
            position=1.0, 
            sweep_length=0.05
            )

    parachute_data = data["parachutes"]["0"]
    print_info("loading parachute")
    rocket.add_parachute(
            name=parachute_data["name"],
            cd_s=parachute_data["cds"],
            trigger=parachute_data["deploy_event"],
            lag=parachute_data["deploy_delay"]
            )
    return rocket

def init_environment_from_JSON(path_to_file):
    with open(path_to_file, 'r', encoding='utf-8')as file:
        data= json.load(file)
    print_info(f"Reading from {path_to_file}")
    env_data = data["environment"]
    env = Environment(
                latitude=env_data["latitude"], 
                longitude=env_data["longitude"],
                date=env_data["date"],
                elevation=env_data["elevation"]
            )
    return env

def init_flight_from_JSON(path_to_file, rocket, environment):
    with open(path_to_file, 'r', encoding='utf-8')as file:
        data= json.load(file)
    print_info(f"Reading from {path_to_file}")
    flight_data = data["flight"]
    temp  = Flight(
               heading=flight_data["heading"],
               environment= environment,
               rocket=rocket,
               rail_length=flight_data["rail_length"]
            )
    return temp

def init_accelerometer_from_JSON(path_to_file, name):
    with open(path_to_file, 'r', encoding='utf-8')as file:
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
    with open(path_to_file, 'r', encoding='utf-8')as file:
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
        rocket.add_sensor(a , 1)
    return rocket

def parallel_generator(N, rocket, environment, flight):
    indices = range(N) 
    def worker(i):
        return run_single_simulation(i, rocket, environment, flight)

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
    json_path = "../../source_model/APEX_OUTPUT/parameters.json"
    drag_path= "../../source_model/APEX_OUTPUT/drag_curve.csv"
    thrust_path= "../../source_model/APEX_OUTPUT/thrust_source.csv"
    rocket = init_rocket_from_JSON(json_path, drag_path , thrust_path)
    environment = init_environment_from_JSON("config.json")
    acc_list = [] 
    acc_list.append(init_accelerometer_from_JSON("../sensors/accelerometer.json","LSM9DS1_acc_2g"))
    acc_list.append(init_accelerometer_from_JSON("../sensors/accelerometer.json","LSM9DS1_acc_4g"))
    acc_list.append(init_accelerometer_from_JSON("../sensors/accelerometer.json","LSM9DS1_acc_8g"))
    acc_list.append(init_accelerometer_from_JSON("../sensors/accelerometer.json","LSM9DS1_acc_16g"))
    acc_list.append(init_gyroscope_from_JSON("../sensors/gyroscope.json","LSM9DS1_gyro_245dps"))
    acc_list.append(init_gyroscope_from_JSON("../sensors/gyroscope.json","LSM9DS1_gyro_500dps"))
    acc_list.append(init_gyroscope_from_JSON("../sensors/gyroscope.json","LSM9DS1_gyro_2000dps"))
    flight = init_flight_from_JSON("config.json",rocket,environment)
    add_acc_to_rocket(rocket , acc_list)
    parallel_generator(3,rocket , environment , flight)
    
if __name__=="__main__":
    main()
