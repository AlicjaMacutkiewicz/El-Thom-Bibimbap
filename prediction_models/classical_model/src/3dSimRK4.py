import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import constants as const
import isacalc as isa
import pandas as pd
from SimHelper import *

# ---------------------------- Symulacja ----------------------------

def DragCoefficient(M):
    # todo doczytac wiecej
    # todo predkosci supersoniczne
    if M < 0.8:
        return 0.5
    elif M < 1.2:
        return 0.5 + (M - 0.8) / 0.4 * 0.7
    elif M < 2.0:
        return 1.2 - (M - 1.2) / 0.8 * 0.6
    else:
        return 0.6

def CalculateDrag(atm_params, velocity, position):
    v_air = velocity - np.cross(earth_angular_velocity_vector, position)
    v_mag = np.linalg.norm(v_air)

    air_density_isa = atm_params[3]
    speed_of_sound = atm_params[4]
    Mach = v_mag / max(speed_of_sound, 1e-8)

    drag_coefficient = DragCoefficient(Mach)
    drag_vector = -0.5 * air_density_isa * drag_coefficient * area * v_mag * v_air

    return drag_vector

def CalculateThrust(position):
    # todo ciag zmienia sie wraz z cisnieniem
    up = position / max(np.linalg.norm(position), 1e-8)
    east = np.cross(np.array([0, 0, 1]), up)

    norm = np.linalg.norm(east)
    if norm < 1e-8:
        east = np.array([1,0,0])
    else:
        east /= norm

    north = np.cross(up, east)
    north /= max(np.linalg.norm(north), 1e-8)
    thrust_dir = (
            math.cos(pitch) * up +
            math.sin(pitch) * north
    )
    thrust_vector = thrust * thrust_dir
    return thrust_vector

def CalculateNonInertialForces(velocity, position):
    coriolis = -2 * np.cross(earth_angular_velocity_vector, velocity)
    centrifugal = -np.cross(earth_angular_velocity_vector, np.cross(earth_angular_velocity_vector, position))
    return coriolis, centrifugal

def RK4(dt, fuel_mass, flight_time):
    def acceleration(position, velocity, fuelMass, flightTime):
        height = max(0, np.linalg.norm(position) - earth_radius)

        if height > 100000:
            drag = np.zeros(3)
        else:
            params = atm.calculate(h=height)
            drag = CalculateDrag(params, velocity, position)

        if flightTime < burn_time and fuelMass > 0:
            thrust_vector = CalculateThrust(position)
        else:
            thrust_vector = np.zeros(3)

        gravity = -const.G * earth_mass * position / max(np.linalg.norm(position) ** 3, 1e-8)
        coriolis, centrifugal = CalculateNonInertialForces(velocity, position)

        mass = rocket_mass + fuelMass
        return (thrust_vector + drag) / max(mass, 1e-8) + gravity + coriolis + centrifugal, gravity, drag

    k1_acceleration, k1_gravity, k1_drag = acceleration(position_global, velocity_global, max(fuel_mass - fuel_consumption * dt / 4, 0), flight_time)
    k1_velocity = k1_acceleration * dt
    k1_position = velocity_global * dt

    k2_acceleration, k2_gravity, k2_drag = acceleration(position_global + 0.5 * k1_position, velocity_global + 0.5 * k1_velocity, max(fuel_mass - fuel_consumption * dt / 2, 0), flight_time + 0.5 * dt)
    k2_velocity = k2_acceleration * dt
    k2_position = (velocity_global + 0.5 * k1_velocity) * dt

    k3_acceleration, k3_gravity, k3_drag = acceleration(position_global + 0.5 * k2_position, velocity_global + 0.5 * k2_velocity, max(fuel_mass - fuel_consumption * dt * 3 / 4, 0), flight_time + 0.5 * dt)
    k3_velocity = k3_acceleration * dt
    k3_position = (velocity_global + 0.5 * k2_velocity) * dt

    k4_acceleration, k4_gravity, k4_drag = acceleration(position_global + k3_position, velocity_global + k3_velocity, max(fuel_mass - fuel_consumption * dt, 0), flight_time + dt)
    k4_velocity = k4_acceleration * dt
    k4_position = (velocity_global + k3_velocity) * dt

    new_acceleration = (k1_acceleration + 2 * k2_acceleration + 2 * k3_acceleration + k4_acceleration) / 6
    new_velocity = velocity_global + (k1_velocity + 2 * k2_velocity + 2 * k3_velocity + k4_velocity) / 6
    new_position = position_global + (k1_position + 2 * k2_position + 2 * k3_position + k4_position) / 6
    new_gravity = (k1_gravity + 2 * k2_gravity + 2 * k3_gravity + k4_gravity) / 6
    new_drag = (k1_drag + 2 * k2_drag + 2 * k3_drag + k4_drag) / 6

    return new_position, new_velocity, new_acceleration, new_gravity, new_drag

while True:
    height = np.linalg.norm(position_global) - earth_radius
    if height <= 0 and flight_time > 0:
        break

    pitch += angle_rate * dt

    if flight_time < burn_time:
        fuel_mass -= fuel_consumption * dt
        fuel_mass = max(fuel_mass, 0)
    elif thrust != 0:
        thrust = 0
        free_flight_Y = position_global[2]
        angle_rate = 0

    position_global, velocity_global, acceleration_local, gravity_local, drag_local = RK4(dt, fuel_mass, flight_time)

    dynamic_pressure = atm.calculate(h=height)[2] * np.linalg.norm(velocity_global) ** 2 / 2

    flight_time += dt

    # ----------- Historia ------------

    time_history.append(flight_time)

    position_x_history.append(position_global[0] - start_position[0])
    position_y_history.append(position_global[1] - start_position[1])
    position_z_history.append(position_global[2] - start_position[2])

    velocity_history.append(np.linalg.norm(velocity_global))
    acceleration_history.append(np.linalg.norm(acceleration_local))
    fuel_history.append(fuel_mass)
    gravity_history.append(np.linalg.norm(gravity_local))
    drag_history.append(np.linalg.norm(drag_local))
    air_density_history.append(atm.calculate(h=height)[3])
    dynamic_pressure_history.append(dynamic_pressure)

#--------------------------- Zapisywanie --------------------------

pd.DataFrame({
    'x': position_x_history,
    'y': position_y_history,
    'z': position_z_history
}).to_csv("rk4_position.csv", index=False)

if not plot:
    plot_history(position_z_history, dynamic_pressure_history, time_history, velocity_history, burn_time, acceleration_history, fuel_history, gravity_history, drag_history, air_density_history, position_x_history, position_y_history)