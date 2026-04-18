import datetime
import json
import numpy as np
import pandas as pd
from rocketpy import Flight , Accelerometer, Gyroscope

import os
import xarray as xr
from enviroment_api import * 
from scipy.interpolate import interp1d

from logger import *


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

#helper function for 'run_single_simulation'
def get_best_acceleration(real_vals, suffix, all_accels_df, thresholds): #change when eagle lands
    cond = [np.abs(real_vals) < t for t in thresholds]
    choices = [all_accels_df[f"LSM6DSOX_acc_{g}g_{suffix}"] for g in [2, 4, 8]]
    return np.select(cond, choices, default=all_accels_df[f"LSM6DSOX_acc_16g_{suffix}"])

def get_best_angular_velocity(real_vals, suffix, all_accels_df, thresholds): #change when eagle lands
    real_vals_dps = np.degrees(real_vals)
    cond = [np.abs(real_vals_dps) < t for t in thresholds]
    choices = [all_accels_df[f"LSM6DSOX_gyro_{dps}dps_{suffix}"] for dps in [125, 250, 500, 1000]]
    return np.select(cond, choices, default=all_accels_df[f"LSM6DSOX_gyro_2000dps_{suffix}"])


def apply_sensor_faults(sensor_data, rng, chance_denominator = 100000):
    chance = 1/chance_denominator
    change = 0
    if rng.random() <= chance:
        print("SENSOR FAULT ")
        change = rng.integers(-65536, 65536) # -(2**16), (2**16)
        Log.print_info("SENSOR FAULT " + str(change))
        #TODO: edycja sensor_data
    return sensor_data + change

def apply_sensor_dropout(current_flight, frame, rng):
    times = frame.index.values
    g = 9.80665
    Log.print_info(f"sensor dropout for length {len(times)}")
    for _ in range(len(times)//10):
        i = np.random.randint(0, len(times))
        ax = current_flight.ax(times[i])
        ay = current_flight.ay(times[i])
        az = current_flight.az(times[i])
        g_force = np.sqrt(ax**2 + ay**2 + az**2) / g
    
        alt = current_flight.z(times[i])

        wind_u = current_flight.env.wind_velocity_x(alt)
        wind_v = current_flight.env.wind_velocity_y(alt)

        wind_speed = np.sqrt(wind_u**2 + wind_v**2)

        drop_rate = np.clip(0.001 + 0.002*wind_speed + (g_force-4)*0.01, 0, 1)

        if rng.random() < drop_rate:
            dropout_interval = rng.integers(50, 500)
            frame.iloc[i : i + dropout_interval] = np.nan

    return frame

def run_single_simulation(i, rocket, environment, heading , rail_length, rng, acceleration_thresholds, angular_velocity_thresholds):
    current_flight = Flight(
            heading=heading,
            environment=environment,
            rocket=rocket,
            rail_length=rail_length
            )
    dir = os.path.dirname(__file__)
    accel_data = []

    for sensor_tuple in rocket.sensors:
        sensor = sensor_tuple.component
        cols = ["Time", f"{sensor.name}_X", f"{sensor.name}_Y", f"{sensor.name}_Z"]
        frame = pd.DataFrame(sensor.measured_data, columns=cols)
        frame.set_index("Time", inplace=True)

        apply_sensor_dropout(current_flight, frame, rng)
        if isinstance(sensor , (Accelerometer, Gyroscope)):
            accel_data.append({
                    "df": frame,
                    "name": sensor.name
                })

    if accel_data:
        all_accels_df = pd.concat([item["df"] for item in accel_data], axis=1)
        times_array = all_accels_df.index.values

        real_acc_x = np.array([current_flight.ax(t) for t in times_array])
        real_acc_y = np.array([current_flight.ay(t) for t in times_array])
        real_acc_z = np.array([current_flight.az(t) for t in times_array])

        real_angvel_x = np.array([current_flight.w1(t) for t in times_array])
        real_angvel_y = np.array([current_flight.w2(t) for t in times_array])
        real_angvel_z = np.array([current_flight.w3(t) for t in times_array])

        all_accels_df["Best_Acc_X"] = get_best_acceleration(real_acc_x, "X", all_accels_df, acceleration_thresholds)
        all_accels_df["Best_Acc_Y"] = get_best_acceleration(real_acc_y, "Y", all_accels_df, acceleration_thresholds)
        all_accels_df["Best_Acc_Z"] = get_best_acceleration(real_acc_z, "Z", all_accels_df, acceleration_thresholds) 
        
        all_accels_df["Best_AngVel_X"] = get_best_angular_velocity(real_angvel_x, "X", all_accels_df, angular_velocity_thresholds)
        all_accels_df["Best_AngVel_Y"] = get_best_angular_velocity(real_angvel_y, "Y", all_accels_df, angular_velocity_thresholds)
        all_accels_df["Best_AngVel_Z"] = get_best_angular_velocity(real_angvel_z, "Z", all_accels_df, angular_velocity_thresholds)
      
        final_cols = ["Best_Acc_X", "Best_Acc_Y", "Best_Acc_Z", 
                      "Best_AngVel_X", "Best_AngVel_Y", "Best_AngVel_Z"]
        final_df = all_accels_df[final_cols]
        # final_df.to_csv(os.path.join(dir, f"output/flight_{i}_test_sensors.csv"), index_label="Time")
        final_df['flight_id'] = i 
        Log.print_info(f"Pakowanko... {i}")
        final_df.to_parquet(f"output/flight_{i}.parquet", index=True)
