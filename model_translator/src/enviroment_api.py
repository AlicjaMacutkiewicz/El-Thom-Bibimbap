import cdsapi
import rocketpy 
import xarray as xr
import numpy as np
import pandas as pd
from rocketpy import Flight , Accelerometer, Gyroscope, Environment
import os
import xarray as xr
from scipy.interpolate import interp1d
from logger import *


# date must be datetime type
# PROVIDED PATH SHOULD NOT CONTAINT FILE NAME ONLY DIRECTORY
def get_enviroment_from_date(environment_data, date,filename, path="../../source_model/ERA5_weather/single/"):
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "surface_pressure"
        ],
         "year": ["2000"],
         "month": ["10"],
         "day": ["10"],
         "time": ["00:00"],

        "data_format": "grib",
        "download_format": "unarchived",
        "area": [54.37, 18.38, 54.12, 18.63]
        # "year": [str(date.year)],
        # "month": [str(date.month)],
        # "day": [str(date.day)],
        # "time": [date.isoformat(timespec='minutes')],
        }
    client = cdsapi.Client()
    client.retrieve(dataset, request).download(path+filename)
    sl = xr.open_dataset(path+"single_"+filename)
    
    dataset = "reanalysis-era5-pressure-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "geopotential",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind"
        ],
        "year": [str(date.date.year)],
        "month": [str(date.date.month)],
        "day": [str(date.date.day)],
        "time": [date.time.isoformat(timespec='minutes')],
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
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [54.37, 18.38, 54.12, 18.63]
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(path+filename)
    pl = xr.open_dataset(path+"levels_"+filename)
    
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
