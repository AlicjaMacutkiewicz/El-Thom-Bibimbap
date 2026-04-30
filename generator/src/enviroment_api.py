import json
import cdsapi
import xarray as xr
import numpy as np
from rocketpy import Environment
import os
import xarray as xr
from scipy.interpolate import interp1d
from logger import *


# date must be datetime type
# PROVIDED PATH SHOULD NOT CONTAINT FILE NAME ONLY DIRECTORY
# This method is not thred save, u need to ensure that no other thred is using same file name 
def get_enviroment_from_date(environment_data, date,  longitude, latitude, filename, path="../../source_model/ERA5_weather/"):
    os.makedirs(os.path.join(path, "single"), exist_ok=True)
    os.makedirs(os.path.join(path, "levels"), exist_ok=True)

    target_single = os.path.join(path , "single", f"single_{filename}")
    target_levels= os.path.join(path , "levels", f"levels_{filename}")

    # i dont know why but miltiurl (which is used by cdsapi), leaves 0-byte ghost files
    # so just if they exist we delete them 
    if(os.path.exists(target_single)):
        os.remove(target_single)
    if(os.path.exists(target_levels)):
        os.remove(target_levels)

    longitude_max = longitude + 0.12
    longitude_min = longitude - 0.13
    latitude_max =  latitude + 0.12
    latitude_min =  latitude - 0.13

    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "surface_pressure"
        ],


        "year": [str(date.year)],
        "month": [str(date.month)],
        "day": [str(date.day)],
        "time": [date.strftime("%H:%M")],
        "data_format": "netcdf4",
        "download_format": "unarchived",
        "area": get_enviroment_from_date
        }

    client = cdsapi.Client(quiet=True)
    client.retrieve(dataset, request, target_single)
    sl = xr.open_dataset(target_single)
    
    dataset = "reanalysis-era5-pressure-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "geopotential",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind"
        ],
        "year": [str(date.year)],
        "month": [str(date.month)],
        "day": [str(date.day)],
        "time": [date.strftime("%H:%M")],
        "pressure_level": [
            "1", "2", "3",
            "5", "7", "10",
            "20", "30", "50",
            "70", "100", "125",
            "150", "175", "200",
            "225", "250", "300",
            "350", "400", "450",
            "500", "550", "600",
            "650", "700", "750",
            "775", "800", "825",
            "850", "875", "900",
            "925", "950", "975",
            "1000"
        ],
        "data_format": "netcdf4",
        "download_format": "unarchived",
        "area": [latitude_max, longitude_min, latitude_min, longitude_max ]
    }

    client = cdsapi.Client(quiet=True)
    client.retrieve(dataset, request, target_levels)
    pl = xr.open_dataset(target_levels)
    
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

    h_new = np.linspace(2, 30000, 200)
    T_new = interp1d(h, T, fill_value="extrapolate")(h_new)
    U_new = interp1d(h, U, fill_value="extrapolate")(h_new)
    V_new = interp1d(h, V, fill_value="extrapolate")(h_new)

    env = Environment(
        latitude = environment_data["latitude"],
        longitude = environment_data["longitude"],
        elevation = environment_data["elevation"]
    )

    temp_profile = np.column_stack((h_new, T_new))
    u_profile = np.column_stack((h_new, U_new))
    v_profile = np.column_stack((h_new, V_new))

    env.set_atmospheric_model(
        type="custom_atmosphere",
        temperature=temp_profile,
        wind_u=u_profile,
        wind_v=v_profile,
    )
    env.date = date    
    # stoch_env = StochasticEnvironment(environment= env,
    # wind_velocity_x_factor= u_profile,
    # wind_velocity_y_factor= v_profile)
    # env = stoch_env.create_object()
    return env

def download_yearly_weather(year, longitude, latitude, path="../../source_model/ERA5_weather/"):
    """
    Downloads an entire year of data (at 12:00) in just TWO API requests.
    Saves them locally so workers don't need internet access.
    """
    os.makedirs(os.path.join(path, "yearly"), exist_ok=True)
    target_single = os.path.join(path, "yearly", f"single_{year}.nc")
    target_levels = os.path.join(path, "yearly", f"levels_{year}.nc")

    if os.path.exists(target_single) and os.path.exists(target_levels):
        Log.print_info(f"Weather data for {year} already downloaded locally.")
        return target_single, target_levels

    months = [f"{i:02d}" for i in range(1, 13)]
    days = [f"{i:02d}" for i in range(1, 32)]
    
    client = cdsapi.Client(quiet=True)
    longitude_max = longitude + 0.12
    longitude_min = longitude - 0.13
    latitude_max =  latitude + 0.12
    latitude_min =  latitude - 0.13
    if not os.path.exists(target_single):
        Log.print_info(f"Queueing API request for {year} SINGLE levels (This will take a minute or two...)")
        dataset_sl = "reanalysis-era5-single-levels"
        request_sl = {
            "product_type": ["reanalysis"],
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "surface_pressure"],
            "year": [str(year)],
            "month": months,
            "day": days,
            "time": ["12:00"],
            "data_format": "netcdf4",
            "download_format": "unarchived",
            "area": [latitude_max, longitude_min, latitude_min, longitude_max ]
        }
        client.retrieve(dataset_sl, request_sl, target_single)

    if not os.path.exists(target_levels):
        Log.print_info(f"Queueing API request for {year} PRESSURE levels (This will take a minute or two...)")
        dataset_pl = "reanalysis-era5-pressure-levels"
        request_pl = {
            "product_type": ["reanalysis"],
            "variable": ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind"],
            "year": [str(year)],
            "month": months,
            "day": days,
            "time": ["12:00"],
            "pressure_level": [
                "1", "2", "3", "5", "7", "10", "20", "30", "50", "70", "100", "125",
                "150", "175", "200", "225", "250", "300", "350", "400", "450", "500",
                "550", "600", "650", "700", "750", "775", "800", "825", "850", "875",
                "900", "925", "950", "975", "1000"
            ],
            "data_format": "netcdf4",
            "download_format": "unarchived",
            "area": [latitude_max, longitude_min, latitude_min, longitude_max ]

        }
        client.retrieve(dataset_pl, request_pl, target_levels)

    return target_single, target_levels


def get_enviroment_from_batched_file(environment_data, date, single_file, levels_file):
    """
    Called by the multiprocessing workers. Opens the local file, slices out the specific date,
    and runs the interpolation math. Very fast.
    """
    with xr.open_dataset(single_file) as sl, xr.open_dataset(levels_file) as pl:
        if 'valid_time' in sl.dims:
            sl = sl.rename_dims({'valid_time': 'time'}).rename_vars({'valid_time': 'time'})
        if 'valid_time' in pl.dims:
            pl = pl.rename_dims({'valid_time': 'time'}).rename_vars({'valid_time': 'time'})
        sl_date = sl.sel(time=date, method="nearest")
        pl_date = pl.sel(time=date, method="nearest")
        
        g = 9.80665
        geo = pl_date["z"].data.flatten() # geopotential
        H = geo / g  # height
        T = pl_date["t"].data.flatten() # temperature
        U = pl_date["u"].data.flatten() # u-wind
        V = pl_date["v"].data.flatten() # v-wind

        t2m = sl_date["t2m"].data.item() # 2 meters temperature
        u10 = sl_date["u10"].data.item() # 10 meters u-wind
        v10 = sl_date["v10"].data.item() # 10 meters v-wind

    h = np.insert(H, 0, 2.0)
    T = np.insert(T, 0, t2m)
    U = np.insert(U, 0, u10)
    V = np.insert(V, 0, v10)

    h_new = np.linspace(2, 30000, 200)
    T_new = interp1d(h, T, fill_value="extrapolate")(h_new)
    U_new = interp1d(h, U, fill_value="extrapolate")(h_new)
    V_new = interp1d(h, V, fill_value="extrapolate")(h_new)

    env = Environment(
        latitude = environment_data["latitude"],
        longitude = environment_data["longitude"],
        elevation = environment_data["elevation"]
    )

    temp_profile = np.column_stack((h_new, T_new))
    u_profile = np.column_stack((h_new, U_new))
    v_profile = np.column_stack((h_new, V_new))

    env.set_atmospheric_model(
        type="custom_atmosphere",
        temperature=temp_profile,
        wind_u=u_profile,
        wind_v=v_profile,
    )
    env.date = date    

    return env