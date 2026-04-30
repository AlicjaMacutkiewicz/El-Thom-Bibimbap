import json
import math

import numpy as np
from scipy import constants as const
import isacalc as isa
from matplotlib import pyplot as plt

# -------------------------- Ustawienia ---------------------------

plot = False
save = False

# ----------------------------- Stale -----------------------------

burn_time = 8  # s
rocket_mass = 10  # kg
fuel_mass = 6  # kg
fuel_consumption = fuel_mass / burn_time  # kg/s
earth_mass = 5.972e24  # kg
earth_radius = 6371009  # m
latitude = 54.371552353039895
longitude = 18.613586492761833
phi = math.radians(latitude)
lam = math.radians(longitude)
earth_angular_velocity = 7.2921159e-5
area = 0.05  # m^2
air_density_ground = 1.225  # kg/m^3
air_molar_mass = 0.0289644
molar_gas_const = 8.314462618
air_temperature = const.zero_Celsius + 20
pitch = math.radians(0)  # pionowo
angle_rate = 0.0005  # rad/s
thrust = 400  # todo ciag zalezy od cisnienia
dt = 0.01  # s
atm = isa.Atmosphere()

# ---------------------------- Zmienne ----------------------------

flight_time = 0     # s

position_global = np.array([
    earth_radius * math.cos(phi) * math.cos(lam),
    earth_radius * math.cos(phi) * math.sin(lam),
    earth_radius * math.sin(phi)
])
start_position = position_global.copy()

earth_angular_velocity_vector = np.array([0,0,earth_angular_velocity])
velocity_global = np.cross(earth_angular_velocity_vector, position_global)        # predkosc; x, y, z

# ----------------------------------------------------------------------------------------------------------------

def initR7():
    global burn_time
    global rocket_mass
    global fuel_mass
    global fuel_consumption
    global earth_mass
    global earth_radius
    global latitude
    global longitude
    global phi
    global lam
    global earth_angular_velocity
    global area
    global air_density_ground
    global air_molar_mass
    global molar_gas_const
    global air_temperature
    global pitch
    global angle_rate
    global thrust

    with open('example.json') as f:
        data = json.load(f)