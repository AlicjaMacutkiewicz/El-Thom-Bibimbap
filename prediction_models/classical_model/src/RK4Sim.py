import numpy as np
from scipy import constants as const

def calculate_non_inertial_forces(velocity, position):
    earth_angular_velocity = 7.2921159e-5
    earth_angular_velocity_vector = np.array([0, 0, earth_angular_velocity])
    coriolis = -2 * np.cross(earth_angular_velocity_vector, velocity)
    centrifugal = -np.cross(earth_angular_velocity_vector, np.cross(earth_angular_velocity_vector, position))
    return coriolis, centrifugal

def rk4(dt, fuel_mass, rocket_mass, thrust, position, velocity):
    def acceleration(position, velocity, fuelMass):
        earth_mass = 5.972e24
        gravity = -const.G * earth_mass * position / max(np.linalg.norm(position) ** 3, 1e-8)
        coriolis, centrifugal = calculate_non_inertial_forces(velocity, position)

        mass = rocket_mass + fuelMass
        acc = thrust / max(mass, 1e-8) + gravity + coriolis + centrifugal
        return acc

    k1_acceleration = acceleration(position, velocity, fuel_mass)
    k1_velocity = k1_acceleration * dt
    k1_position = velocity * dt

    k2_acceleration = acceleration(position + 0.5 * k1_position, velocity + 0.5 * k1_velocity, fuel_mass)
    k2_velocity = k2_acceleration * dt
    k2_position = (velocity + 0.5 * k1_velocity) * dt

    k3_acceleration = acceleration(position + 0.5 * k2_position, velocity + 0.5 * k2_velocity, fuel_mass)
    k3_velocity = k3_acceleration * dt
    k3_position = (velocity + 0.5 * k2_velocity) * dt

    k4_acceleration = acceleration(position + k3_position, velocity + k3_velocity, fuel_mass)
    k4_velocity = k4_acceleration * dt
    k4_position = (velocity + k3_velocity) * dt

    new_acceleration = (k1_acceleration + 2 * k2_acceleration + 2 * k3_acceleration + k4_acceleration) / 6
    new_velocity = velocity + (k1_velocity + 2 * k2_velocity + 2 * k3_velocity + k4_velocity) / 6
    new_position = position + (k1_position + 2 * k2_position + 2 * k3_position + k4_position) / 6

    return new_position, new_velocity, new_acceleration