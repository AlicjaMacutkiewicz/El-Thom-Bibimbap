import os
import matplotlib.pyplot as plt
import numpy as np
import json
import string
import tqdm
from enum import IntEnum
import pandas as pd
from rocketpy import Environment, SolidMotor, Rocket, Flight , Barometer , Accelerometer, Gyroscope


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
# cretes string based on rp.solution_array, without np.float type signature
# @ARGUMENTS: 
#       data -> rp.solution_array instance
# @RETURN
#       msg -> string representation of given array
def rp_solution_arr_str(data):
    msg = '['
    for itr in data:
        msg += f"{str(itr)},"
    msg += ']'
    return msg

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

def init_sensors_from_JSON(path_to_file) :
    with open(path_to_file, 'r', encoding='utf-8')as file:
        data= json.load(file)
    barometer_data = data["barometer"] 
    barometer = Barometer(
        sampling_rate=barometer_data["sampling_rate"],
        measurement_range=barometer_data["measurement_range"],
        resolution=barometer_data["resolution"],
        noise_density=barometer_data["noise_density"],
        noise_variance=barometer_data["noise_variance"],
        random_walk_density=barometer_data["random_walk_density"],
        random_walk_variance=barometer_data["random_walk_variance"],
        constant_bias=barometer_data["constant_bias"],
        operating_temperature=barometer_data["operating_temperature"],
        temperature_bias=barometer_data["temperature_bias"]
    )
    
    accel_data = data["accelerometer"]
    accelerometer = Accelerometer(
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
    gyro_data = data["gyroscope"]
    gyroscope = Gyroscope(
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
    return (barometer,accelerometer,gyroscope) 

<<<<<<< HEAD
def generator(N, rocket, environment, flight):
=======
def generator(N, flight):
>>>>>>> origin/sensor-parameters
    for i in tqdm.tqdm(range(N), "Siupi duping grzesia"):
        current_flight =flight
        file_name = f"output/flighjt_{i}.out"
        with open(file_name, 'w+') as file:
            for sample in current_flight.solution:
                file.write(rp_solution_arr_str(sample) + '\n')
        sensor_dfs = []
        for sensor_tuple in rocket.sensors:
            sensor = sensor_tuple.component 
            name = sensor.__class__.__name__
            if name == "Barometer":
                cols = ["Time", "Baro_Pressure"] 
            elif name == "Accelerometer":
                cols = ["Time", "Acc_X", "Acc_Y", "Acc_Z"]
            elif name == "Gyroscope":
                cols = ["Time", "Gyro_X", "Gyro_Y", "Gyro_Z"]
            else:
                num_measurements = len(sensor.measured_data[0]) - 1
                cols = ["Time"] + [f"{name}_{j}" for j in range(num_measurements)]
            df = pd.DataFrame(sensor.measured_data, columns=cols)
            df.set_index("Time", inplace=True)
            sensor_dfs.append(df)
        if sensor_dfs:
            final_telemetry_df = pd.concat(sensor_dfs, axis=1)
            final_telemetry_df.sort_index(inplace=True)
            file_name_csv = f"output/flight_{i}_sensors.out"
            final_telemetry_df.to_csv(file_name_csv, index_label="Time")

def main():
    json_path = "../../source_model/APEX_OUTPUT/parameters.json"
    drag_path= "../../source_model/APEX_OUTPUT/drag_curve.csv"
    thrust_path= "../../source_model/APEX_OUTPUT/thrust_source.csv"
    rocket = init_rocket_from_JSON(json_path, drag_path , thrust_path)
    environment = init_environment_from_JSON("config.json")
    (barometer,accelerometer,gyroscope)  = init_sensors_from_JSON("config.json")
    rocket.add_sensor(gyroscope , 1)
    rocket.add_sensor(accelerometer, 1)
    rocket.add_sensor(barometer, 1)
    flight = init_flight_from_JSON("config.json",rocket,environment)
    generator(10,rocket , environment , flight)

if __name__=="__main__":
    main()
