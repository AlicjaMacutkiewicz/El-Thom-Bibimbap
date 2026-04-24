import numpy as np
import pandas as pd
from rocketpy import Flight , Accelerometer, Gyroscope, Barometer
from rocketpy.sensors import ScalarSensor

import os
from enviroment_api import * 

from logger import *

from numba import njit, prange

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

@njit
def apply_sensor_faults_numba(sensor_data, chance_denominator):
    chance = 1.0 / chance_denominator
    change = 0.0
    if np.random.random() <= chance:
        change = float(np.random.randint(-65536, 65536))
    
    return sensor_data + change

def apply_sensor_faults(sensor_data, rng, chance_denominator = 100000):
    result = apply_sensor_faults_numba(float(sensor_data), float(chance_denominator))
    if result != sensor_data:
        Log.print_info("SENSOR FAULT " + str(result - sensor_data))

    return result

@njit(fastmath=True) 
def fast_extract_numba(times, x_array, y_array):
    return np.interp(times, x_array, y_array)

def fast_extract(rp_func, times):
    x_arr = np.asarray(rp_func.x_array, dtype=np.float64)
    y_arr = np.asarray(rp_func.y_array, dtype=np.float64)
    times_arr = np.asarray(times, dtype=np.float64)
    
    return fast_extract_numba(times_arr, x_arr, y_arr)

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

        for col in frame.columns:
            frame[col] = frame[col].apply(lambda val: apply_sensor_faults(val, rng))
            
        if isinstance(sensor , (Accelerometer, Gyroscope, ScalarSensor, Barometer)):
            accel_data.append({
                    "df": frame,
                    "name": sensor.name
                })

    if accel_data:
        all_accels_df = pd.concat([item["df"] for item in accel_data], axis=1)
        all_accels_df.sort_index(inplace=True)
        times_array = all_accels_df.index.values

        all_accels_df["Acceleration_X"] = fast_extract(current_flight.ax, times_array)
        all_accels_df["Acceleration_Y"] = fast_extract(current_flight.ay, times_array)
        all_accels_df["Acceleration_Z"] = fast_extract(current_flight.az, times_array)

        all_accels_df["Position_X"] = fast_extract(current_flight.x, times_array)
        all_accels_df["Position_Y"] = fast_extract(current_flight.y, times_array)
        all_accels_df["Position_Z"] = fast_extract(current_flight.z, times_array)

        all_accels_df["Thrust"] = fast_extract(rocket.motor.thrust, times_array)
        all_accels_df["Mass"] = fast_extract(rocket.total_mass, times_array)

        all_accels_df["Best_Acc_X"] = get_best_acceleration(all_accels_df["Acceleration_X"], "X", all_accels_df, acceleration_thresholds)
        all_accels_df["Best_Acc_Y"] = get_best_acceleration(all_accels_df["Acceleration_Y"], "Y", all_accels_df, acceleration_thresholds)
        all_accels_df["Best_Acc_Z"] = get_best_acceleration(all_accels_df["Acceleration_Z"], "Z", all_accels_df, acceleration_thresholds) 
        
        real_w1, real_w2, real_w3 = fast_extract(current_flight.w1, times_array), fast_extract(current_flight.w2, times_array), fast_extract(current_flight.w3, times_array)
        all_accels_df["Best_AngVel_X"] = get_best_angular_velocity(real_w1, "X", all_accels_df, angular_velocity_thresholds)
        all_accels_df["Best_AngVel_Y"] = get_best_angular_velocity(real_w2, "Y", all_accels_df, angular_velocity_thresholds)
        all_accels_df["Best_AngVel_Z"] = get_best_angular_velocity(real_w3, "Z", all_accels_df, angular_velocity_thresholds)

        scalar_cols = [c for c in all_accels_df.columns if c.endswith("_Value")]
        
        final_cols = [
            "Best_Acc_X", "Best_Acc_Y", "Best_Acc_Z", 
            "Best_AngVel_X", "Best_AngVel_Y", "Best_AngVel_Z"
        ] + scalar_cols + [
            "Thrust", "Mass", 
            "Position_X", "Position_Y", "Position_Z",
            "Acceleration_X", "Acceleration_Y", "Acceleration_Z"
        ]
        
        final_df = all_accels_df[final_cols].copy()
        final_df.ffill(inplace=True)
        final_df.bfill(inplace=True)
        # final_df.to_csv(os.path.join(dir, f"output/flight_{i}_test_sensors.csv"), index_label="Time")
        final_df['flight_id'] = i 
        Log.print_info(f"Pakowanko... {i}")
        final_df.to_parquet(f"output/flight_{i}.parquet", index=True)
