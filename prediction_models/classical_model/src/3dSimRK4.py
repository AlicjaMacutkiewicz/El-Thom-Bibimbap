import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import constants as const
import isacalc as isa
import pandas as pd
from SimHelper import *

# ---------------------------- Symulacja ----------------------------

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
        if flightTime < burn_time and fuelMass > 0:
            thrust_vector = CalculateThrust(position)
        else:
            thrust_vector = np.zeros(3)

        gravity = -const.G * earth_mass * position / max(np.linalg.norm(position) ** 3, 1e-8)
        coriolis, centrifugal = CalculateNonInertialForces(velocity, position)

        mass = rocket_mass + fuelMass
        return (thrust_vector) / max(mass, 1e-8) + gravity + coriolis + centrifugal, gravity

    k1_acceleration, k1_gravity = acceleration(position_global, velocity_global, max(fuel_mass - fuel_consumption * dt / 4, 0), flight_time)
    k1_velocity = k1_acceleration * dt
    k1_position = velocity_global * dt

    k2_acceleration, k2_gravity = acceleration(position_global + 0.5 * k1_position, velocity_global + 0.5 * k1_velocity, max(fuel_mass - fuel_consumption * dt / 2, 0), flight_time + 0.5 * dt)
    k2_velocity = k2_acceleration * dt
    k2_position = (velocity_global + 0.5 * k1_velocity) * dt

    k3_acceleration, k3_gravity = acceleration(position_global + 0.5 * k2_position, velocity_global + 0.5 * k2_velocity, max(fuel_mass - fuel_consumption * dt * 3 / 4, 0), flight_time + 0.5 * dt)
    k3_velocity = k3_acceleration * dt
    k3_position = (velocity_global + 0.5 * k2_velocity) * dt

    k4_acceleration, k4_gravity = acceleration(position_global + k3_position, velocity_global + k3_velocity, max(fuel_mass - fuel_consumption * dt, 0), flight_time + dt)
    k4_velocity = k4_acceleration * dt
    k4_position = (velocity_global + k3_velocity) * dt

    new_acceleration = (k1_acceleration + 2 * k2_acceleration + 2 * k3_acceleration + k4_acceleration) / 6
    new_velocity = velocity_global + (k1_velocity + 2 * k2_velocity + 2 * k3_velocity + k4_velocity) / 6
    new_position = position_global + (k1_position + 2 * k2_position + 2 * k3_position + k4_position) / 6
    new_gravity = (k1_gravity + 2 * k2_gravity + 2 * k3_gravity + k4_gravity) / 6

    return new_position, new_velocity, new_acceleration, new_gravity

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

    position_global, velocity_global, acceleration_local, gravity_local = RK4(dt, fuel_mass, flight_time)

    dynamic_pressure = atm.calculate(h=height)[2] * np.linalg.norm(velocity_global) ** 2 / 2

    flight_time += dt