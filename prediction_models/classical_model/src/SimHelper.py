import math

import numpy as np
from scipy import constants as const
import isacalc as isa
from matplotlib import pyplot as plt

# ----------------------------- Stale -----------------------------

plot = False
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

# ---------------------------- Statystyka ----------------------------

# todo wiecej
time_history = []
position_x_history = []
position_y_history = []
position_z_history = []
velocity_history = []
acceleration_history = []

fuel_history = []
gravity_history = []
drag_history = []
air_density_history = []
dynamic_pressure_history = []

free_flight_Y = 0
apogeum = 0
max_q = 0

def plot_history(position_z_history, dynamic_pressure_history, time_history, velocity_history, burn_time,
         acceleration_history, fuel_history, gravity_history, drag_history, air_density_history,
         position_x_history, position_y_history):
    apogeum = max(position_z_history)
    print("Apogeum: ", apogeum)

    max_q = max(dynamic_pressure_history)
    print("Max-q wartosc ", max_q)
    max_q_index = dynamic_pressure_history.index(max_q)
    max_q = time_history[max_q_index]
    print("Max-q czas: ", max_q)
    max_q = position_z_history[max_q_index]
    print("Max-q wysokosc: ", max_q)

    fig, ax = plt.subplots(7, 1, figsize=(30, 30))

    ax[0].plot(time_history, velocity_history)
    ax[0].axvline(x=burn_time, color='b')
    ax[0].set_ylabel("Velocity")

    ax[1].plot(time_history, acceleration_history)
    ax[1].axvline(x=burn_time, color='b')
    ax[1].set_ylabel("Acceleration")

    ax[2].plot(time_history, fuel_history)
    ax[2].axvline(x=burn_time, color='b')
    ax[2].set_ylabel("Fuel")

    ax[3].plot(time_history, gravity_history)
    ax[3].axvline(x=burn_time, color='b')
    ax[3].set_ylabel("Gravity")

    ax[4].plot(time_history, drag_history)
    ax[4].axvline(x=burn_time, color='b')
    ax[4].set_ylabel("Drag")

    ax[5].plot(time_history, air_density_history)
    ax[5].axvline(x=burn_time, color='b')
    ax[5].set_ylabel("Air Density")

    ax[6].plot(time_history, position_z_history)
    ax[6].axvline(x=burn_time, color='b')
    ax[6].set_ylabel("Y")

    plt.xlabel("Time")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.plot(position_x_history, position_y_history, position_z_history)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
