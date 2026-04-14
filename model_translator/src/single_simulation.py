import json

import numpy as np
import pandas as pd
import xarray as xr
from rocketpy import Flight, Accelerometer, Gyroscope, Environment, GnssReceiver
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


# helper function for 'run_single_simulation'
def get_best_acceleration(real_vals, suffix, all_accels_df, thresholds):
    cond = [np.abs(real_vals) < t for t in thresholds]
    choices = [all_accels_df[f"LSM9DS1_acc_{g}g_{suffix}"] for g in [2, 4, 8]]
    return np.select(cond, choices, default=all_accels_df[f"LSM9DS1_acc_16g_{suffix}"])

def get_best_angular_velocity(real_vals, suffix, all_accels_df, thresholds):
    real_vals_dps = np.degrees(real_vals)
    cond = [np.abs(real_vals_dps) < t for t in thresholds]
    choices = [all_accels_df[f"LSM9DS1_gyro_{dps}dps_{suffix}"] for dps in [245, 500]]
    return np.select(cond, choices, default=all_accels_df[f"LSM9DS1_gyro_2000dps_{suffix}"])

def create_new_environment(environment_data, rng):
    """
    Potrzebujemy 2 datasety
    1. ERA5 hourly data on single levels from 1940 to present
    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
     a) reanalysis
     b) 2m temperature
     c) Surface pressure
     d) 10m u-component of wind
     e) 10m v-component of wind
     f) data
     g) godzina (Proponuję zawsze ustawiać 12:00)
     h) Sub-region extraction (North: 54.37, South: 54.12, West 18.38, East: 18.63)
     i) NetCDF4
     j) Zip


    2. ERA5 hourly data on pressure levels from 1940 to present
    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview
     a) reanalysis
     b) geopotential
     c) temperature
     d) u-component of wind
     e) v-component of wind
     f) data
     g) godzina (Proponuję zawsze ustawiać 12:00)
     h) Pressure level (tu klikamy select all)
     i) Sub-region extraction (North: 54.37, South: 54.12, West 18.38, East: 18.63)
     j) NetCDF4
     k) Zip
    """

    pl = xr.open_dataset(f"../../source_model/ERA5_weather/levels/{environment_data['path']}")
    sl = xr.open_dataset(f"../../source_model/ERA5_weather/single/{environment_data['path']}")

    g = 9.80665
    geo = pl["z"].data[0].flatten() # geopotential
    H = geo  / g  # height
    T = pl["t"].data[0].flatten() # temperature
    U = pl["u"].data[0].flatten() # u-wind
    V = pl["v"].data[0].flatten() # v-wind

    t2m = sl["t2m"].data[0].item() # 2 meters temperature
    u10 = sl["u10"].data[0].item() # 10 meters u-wind
    v10 = sl["v10"].data[0].item() # 10 meters v-wind

    h = np.insert(H, 0, 2.0)
    T = np.insert(T, 0, t2m)
    U = np.insert(U, 0, u10)
    V = np.insert(V, 0, v10)

    h_new = np.linspace(0, 30000, 200)
    T_new = interp1d(h, T, fill_value="extrapolate")(h_new)
    U_new = interp1d(h, U, fill_value="extrapolate")(h_new)
    V_new = interp1d(h, V, fill_value="extrapolate")(h_new)


    #
    #   HEJ TU WIKTOR WSTAILEM TE DWIE LINI POD TYM KOMENTARZREM ZEBY JUST 
    # WYNIKI BYLO LOSOWE W RAZIE LEPSZEGO ROZWIAZANIA PROSZE JE USUNAC
    #
    U_new += rng.normal(0, 1.5, size=U_new.shape)
    V_new += rng.normal(0, 1.5, size=V_new.shape)
    env = Environment(
        latitude = environment_data["latitude"],
        longitude = environment_data["longitude"],
        elevation = environment_data["elevation"]
    )

    # todo w tym momencie mozna nakladac szumy na atmosfere

    temp_profile = np.column_stack((h_new, T_new))
    u_profile = np.column_stack((h_new, U_new))
    v_profile = np.column_stack((h_new, V_new))

    env.set_atmospheric_model(
        type="custom_atmosphere",
        temperature=temp_profile,
        wind_u=u_profile,
        wind_v=v_profile,
    )

    return env

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
    if len(times) == 0:
        return frame
    for _ in range(len(times)//10):
        i = rng.integers(0, len(times))
        random_times = times[i]

        ax = current_flight.ax(random_times)
        ay = current_flight.ay(random_times)
        az = current_flight.az(random_times)
        g_force = np.sqrt(ax**2 + ay**2 + az**2) / 9.80665
        alt = current_flight.z(random_times)
        wind_speed = np.sqrt(current_flight.env.wind_velocity_x(alt)**2 +
                             current_flight.env.wind_velocity_y(alt)**2)

        drop_rate = np.clip(0.001 + 0.002*wind_speed + (g_force-4)*0.01, 0, 1)

        if rng.random() < drop_rate:
            dropout_interval = rng.integers(50, 500)
            frame.iloc[i : i + dropout_interval] = np.nan

    return frame

def run_single_simulation(i, rocket, environment_data, heading , rail_length, rng):
    current_flight = Flight(
            heading=heading,
            environment=create_new_environment(environment_data, rng),
            rocket=rocket,
            rail_length=rail_length
            )

    file_name = f"output/flight_{i}.out"
    with open(file_name, 'w+') as file:
            for sample in current_flight.solution:
                file.write(rp_solution_arr_str(sample) + '\n')
    accel_data = []
    gnss_data = []
    acceleration_thresholds = [0, 0, 0]         #m/s^2
    angular_velocity_thresholds = [0, 0]        #dps

    for sensor_tuple in rocket.sensors:
        sensor = sensor_tuple.component
        cols = ["Time", f"{sensor.name}_X", f"{sensor.name}_Y", f"{sensor.name}_Z"]
        frame = pd.DataFrame(sensor.measured_data, columns=cols)
        frame.set_index("Time", inplace=True)

        apply_sensor_dropout(current_flight, frame, rng)
        if isinstance(sensor , (Accelerometer, Gyroscope)):
            accel_data.append({
                    "df": frame,
                    # "range": sensor.measurement_range,
                    "name": sensor.name
                })
        if isinstance(sensor , GnssReceiver):
            gnss_data.append({
                "df": frame,
                # "range": sensor.measurement_range,
                "name": sensor.name
            })

    # Log.print_info(f"majster wihajster  {len(accel_data)} {len(gnss_data)}")
    if accel_data and gnss_data:
        all_accels_df = pd.concat([item["df"] for item in accel_data], axis=1)
        all_gnsss_df = pd.concat([item["df"] for item in gnss_data], axis=1)

        with open('config.json', 'r') as file:
            data = json.load(file)
            acceleration_thresholds[0] = data["thresholds"]["acceleration"]["X"]
            acceleration_thresholds[1] = data["thresholds"]["acceleration"]["Y"]
            acceleration_thresholds[2] = data["thresholds"]["acceleration"]["Z"]

            angular_velocity_thresholds[0] = data["thresholds"]["angular_velocity"]["Small"]
            angular_velocity_thresholds[1] = data["thresholds"]["angular_velocity"]["Big"]


        master_index = all_accels_df.index.union(all_gnsss_df.index).sort_values()

        combined_df = pd.concat([all_accels_df, all_gnsss_df], axis=1).reindex(master_index)
        combined_df = combined_df.ffill().bfill()

        times_array = combined_df.index.values

        # todo to jest koszmarnie wolne
        # todo wydaje mi sie ze to moze byc szybsze:
        #  current_flight.ax(times)
        #  Pawel
        real_acc_x = np.array([current_flight.ax(t) for t in times_array])
        real_acc_y = np.array([current_flight.ay(t) for t in times_array])
        real_acc_z = np.array([current_flight.az(t) for t in times_array])

        real_angvel_x = np.array([current_flight.w1(t) for t in times_array])
        real_angvel_y = np.array([current_flight.w2(t) for t in times_array])
        real_angvel_z = np.array([current_flight.w3(t) for t in times_array])

        combined_df["Best_Acc_X"] = get_best_acceleration(real_acc_x, "X", combined_df, acceleration_thresholds)
        combined_df["Best_Acc_Y"] = get_best_acceleration(real_acc_y, "Y", combined_df, acceleration_thresholds)
        combined_df["Best_Acc_Z"] = get_best_acceleration(real_acc_z, "Z", combined_df, acceleration_thresholds)

        combined_df["Best_AngVel_X"] = get_best_angular_velocity(real_angvel_x, "X", combined_df, angular_velocity_thresholds)
        combined_df["Best_AngVel_Y"] = get_best_angular_velocity(real_angvel_y, "Y", combined_df, angular_velocity_thresholds)
        combined_df["Best_AngVel_Z"] = get_best_angular_velocity(real_angvel_z, "Z", combined_df, angular_velocity_thresholds)

        gnss_name = "u-blox_MAX-M10S"
        combined_df["GNSS_X"] = combined_df[f"{gnss_name}_X"]
        combined_df["GNSS_Y"] = combined_df[f"{gnss_name}_Y"]
        combined_df["GNSS_Z"] = combined_df[f"{gnss_name}_Z"]

        final_cols = ["Best_Acc_X", "Best_Acc_Y", "Best_Acc_Z",
                      "Best_AngVel_X", "Best_AngVel_Y", "Best_AngVel_Z", "GNSS_X","GNSS_Y","GNSS_Z"]
        final_df = combined_df[final_cols].copy()
        final_df.to_csv(f"output/flight_{i}_best_sensors.csv", index_label="Time")

        return final_df
    return None
