import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rocketpy import Flight , Accelerometer, Gyroscope, Environment
import os
import xarray as xr
from scipy.interpolate import interp1d

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
def get_best_acceleration(real_vals, suffix, all_accels_df, thresholds):
    cond = [np.abs(real_vals) < t for t in thresholds]
    print(real_vals)
    choices = [all_accels_df[f"LSM9DS1_acc_{g}g_{suffix}"] for g in [2, 4, 8]]
    return np.select(cond, choices, default=all_accels_df[f"LSM9DS1_acc_16g_{suffix}"])

def get_best_angular_velocity(real_vals, suffix, all_accels_df, thresholds):
    real_vals_dps = np.degrees(real_vals)
    cond = [np.abs(real_vals_dps) < t for t in thresholds]
    choices = [all_accels_df[f"LSM9DS1_gyro_{dps}dps_{suffix}"] for dps in [245, 500]]
    return np.select(cond, choices, default=all_accels_df[f"LSM9DS1_gyro_2000dps_{suffix}"])

def create_new_environment(environment_data):
    '''
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
    '''

    dir = os.path.dirname(__file__)
    pl = xr.open_dataset(os.path.join(dir, f"../../source_model/ERA5_weather/levels/{environment_data['path']}"))
    sl = xr.open_dataset(os.path.join(dir, f"../../source_model/ERA5_weather/single/{environment_data['path']}"))

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
    U_new += np.random.normal(0, 1.5, size=U_new.shape)
    V_new += np.random.normal(0, 1.5, size=V_new.shape)
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

def apply_sensor_faults(sensor_data):
    chance = 1/100000
    change = 0
    if (np.random.rand() <= chance):
        print("SENSOR FAULT ")
        change = np.random.randint(-(2**16), (2**16))
        #TODO: edyjca sensor_data
    return (sensor_data + change) 

def apply_sensor_dropout(current_flight, frame):
    times = frame.index.values

    in_dropout = False
    dropout_remaining = 0

    for _ in range(1000):
        i = np.random.randint(0, len(times))
        ax = current_flight.ax(times[i])
        ay = current_flight.ay(times[i])
        az = current_flight.az(times[i])
        g_force = np.sqrt(ax**2 + ay**2 + az**2) / 9.80665
    
        alt = current_flight.z(times[i])

        wind_u = current_flight.env.wind_velocity_x(alt)
        wind_v = current_flight.env.wind_velocity_y(alt)

        wind_speed = np.sqrt(wind_u**2 + wind_v**2)

        drop_rates = np.clip(0.001 + 0.001*wind_speed + (g_force-4)*0.01, 0, 1)
        if in_dropout:
            frame.iloc[i] = np.nan
            dropout_remaining -= 1
            if dropout_remaining <= 0:
                in_dropout = False
        else:
            if np.random.rand() < drop_rates[i]:
                in_dropout = True
                dropout_remaining = np.random.randint(10, 50)
                frame.iloc[i] = np.nan
    return frame

def run_single_simulation(i, rocket, environment_data, heading , rail_length):
    current_flight = Flight(
            heading=heading,
            environment=create_new_environment(environment_data),
            rocket=rocket,
            rail_length=rail_length
            )
    #for parquet data saving and packing (comment out lines below)
    # |
    dir = os.path.dirname(__file__)
    file_name = os.path.join(dir, f"output/flight_{i}.out")
    with open(file_name, 'w+') as file:
            for sample in current_flight.solution:
                file.write(rp_solution_arr_str(sample) + '\n')
    # |
    accel_data = []

    for sensor_tuple in rocket.sensors:
        sensor = sensor_tuple.component
        if isinstance(sensor , (Accelerometer, Gyroscope)):
            cols = ["Time", f"{sensor.name}_X", f"{sensor.name}_Y" , f"{sensor.name}_Z"]
            frame = pd.DataFrame(sensor.measured_data , columns=cols)
            frame.set_index("Time", inplace=True)
            
            #apply_sensor_dropout(current_flight, frame)

            accel_data.append({
                    "df": frame,
                    "range": sensor.measurement_range,
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

        acceleration_thresholds = [19.613, 39.227, 78.453] #m/s^2
        angular_velocity_thresholds = [245, 500]           #dps

        all_accels_df["Best_Acc_X"] = get_best_acceleration(real_acc_x, "X", all_accels_df, acceleration_thresholds)
        all_accels_df["Best_Acc_Y"] = get_best_acceleration(real_acc_y, "Y", all_accels_df, acceleration_thresholds)
        all_accels_df["Best_Acc_Z"] = get_best_acceleration(real_acc_z, "Z", all_accels_df, acceleration_thresholds) 
        
        all_accels_df["Best_AngVel_X"] = get_best_angular_velocity(real_angvel_x, "X", all_accels_df, angular_velocity_thresholds)
        all_accels_df["Best_AngVel_Y"] = get_best_angular_velocity(real_angvel_y, "Y", all_accels_df, angular_velocity_thresholds)
        all_accels_df["Best_AngVel_Z"] = get_best_angular_velocity(real_angvel_z, "Z", all_accels_df, angular_velocity_thresholds)
        
        final_cols = ["Best_Acc_X", "Best_Acc_Y", "Best_Acc_Z", 
                      "Best_AngVel_X", "Best_AngVel_Y", "Best_AngVel_Z"]
        final_df = all_accels_df[final_cols].copy()
        final_df.to_csv(os.path.join(dir, f"output/flight_{i}_best_sensors.csv"), index_label="Time")
        final_df['flight_id'] = i 

        #for parquet data saving and packing 
        # | 
        # return final_df
        # |
