import numpy as np
import pandas as pd
from rocketpy import Flight , Accelerometer, Gyroscope, Barometer
from rocketpy.sensors import ScalarSensor

import os
from enviroment_api import * 

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
    raw_data = frame.values.astype(float).copy() 
    
    iterations = len(times) // 10
    
    random_indices = rng.integers(0, len(times), size=iterations)
    drop_times = times[random_indices]
    
    ax_vals = fast_extract(current_flight.ax, drop_times)
    ay_vals = fast_extract(current_flight.ay, drop_times)
    az_vals = fast_extract(current_flight.az, drop_times)
    alts = fast_extract(current_flight.z, drop_times)
    
    g_forces = np.sqrt(ax_vals**2 + ay_vals**2 + az_vals**2) / g
    
    wind_u = np.array([current_flight.env.wind_velocity_x(alt) for alt in alts])
    wind_v = np.array([current_flight.env.wind_velocity_y(alt) for alt in alts])
    wind_speeds = np.sqrt(wind_u**2 + wind_v**2)

    drop_rates = np.clip(0.001 + 0.002 * wind_speeds + (g_forces - 4) * 0.01, 0, 1)
    random_rolls = rng.random(size=iterations)
    
    for k in range(iterations):
        if random_rolls[k] < drop_rates[k]:
            i = random_indices[k]
            dropout_interval = rng.integers(50, 500)
            
            end_idx = min(i + dropout_interval, len(times))
            raw_data[i:end_idx, :] = np.nan

    frame.iloc[:, :] = raw_data
    frame.ffill(inplace=True)
    frame.bfill(inplace=True)

    return frame

def fast_extract(rp_func, times):
    return np.interp(times, rp_func.x_array, rp_func.y_array)

def run_single_simulation(i, rocket, environment, heading , rail_length, rng, acceleration_thresholds, angular_velocity_thresholds):
    current_flight = Flight(
            heading=heading,
            environment=environment,
            rocket=rocket,
            rail_length=rail_length
            )
    dir = os.path.dirname(__file__)
    accel_data = []

    # rocket.motor.thrust.source()

    for sensor_tuple in rocket.sensors:
        sensor = sensor_tuple.component
        if len(sensor.measured_data) > 0:
            if len(sensor.measured_data[0]) == 4:
                cols = ["Time", f"{sensor.name}_X", f"{sensor.name}_Y", f"{sensor.name}_Z"]
            else:
                cols = ["Time", f"{sensor.name}_Value"]
        else:
            cols = []

        frame = pd.DataFrame(sensor.measured_data, columns=cols)
        frame.set_index("Time", inplace=True)

        apply_sensor_dropout(current_flight, frame, rng)
        if isinstance(sensor , (Accelerometer, Gyroscope, ScalarSensor, Barometer)):
            accel_data.append({
                    "df": frame,
                    "name": sensor.name
                })

    if accel_data:
        all_accels_df = pd.concat([item["df"] for item in accel_data], axis=1)
        all_accels_df.sort_index(inplace=True)
        times_array = all_accels_df.index.values

        real_acc_x = fast_extract(current_flight.ax, times_array)
        real_acc_y = fast_extract(current_flight.ay, times_array)
        real_acc_z = fast_extract(current_flight.az, times_array)

        real_angvel_x = fast_extract(current_flight.w1, times_array)
        real_angvel_y = fast_extract(current_flight.w2, times_array)
        real_angvel_z = fast_extract(current_flight.w3, times_array)

        all_accels_df["Best_Acc_X"] = get_best_acceleration(real_acc_x, "X", all_accels_df, acceleration_thresholds)
        all_accels_df["Best_Acc_Y"] = get_best_acceleration(real_acc_y, "Y", all_accels_df, acceleration_thresholds)
        all_accels_df["Best_Acc_Z"] = get_best_acceleration(real_acc_z, "Z", all_accels_df, acceleration_thresholds) 
        
        all_accels_df["Best_AngVel_X"] = get_best_angular_velocity(real_angvel_x, "X", all_accels_df, angular_velocity_thresholds)
        all_accels_df["Best_AngVel_Y"] = get_best_angular_velocity(real_angvel_y, "Y", all_accels_df, angular_velocity_thresholds)
        all_accels_df["Best_AngVel_Z"] = get_best_angular_velocity(real_angvel_z, "Z", all_accels_df, angular_velocity_thresholds)

        scalar_cols = [c for c in all_accels_df.columns if c.endswith("_Value")]
        all_accels_df["Thrust"] = [rocket.motor.thrust(t) for t in times_array]

        final_cols = ["Best_Acc_X", "Best_Acc_Y", "Best_Acc_Z", 
                      "Best_AngVel_X", "Best_AngVel_Y", "Best_AngVel_Z", "Thrust"] + scalar_cols
        

        # final_cols = ["Best_Acc_X", "Best_Acc_Y", "Best_Acc_Z", 
        #               "Best_AngVel_X", "Best_AngVel_Y", "Best_AngVel_Z"] + scalar_cols
        
        final_df = all_accels_df[final_cols].copy()
        final_df.ffill(inplace=True)
        final_df.bfill(inplace=True)
        final_df.to_csv(os.path.join(dir, f"output/flight_{i}_test_sensors.csv"), index_label="Time")
        final_df['flight_id'] = i 
        Log.print_info(f"Pakowanko... {i}")
        final_df.to_parquet(f"output/flight_{i}.parquet", index=True)
