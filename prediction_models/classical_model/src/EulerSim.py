import numpy as np
from scipy import constants as const

def calculate_non_inertial_forces(velocity, position):
    earth_angular_velocity = 7.2921159e-5
    earth_angular_velocity_vector = np.array([0, 0, earth_angular_velocity])
    coriolis = -2 * np.cross(earth_angular_velocity_vector, velocity)
    centrifugal = -np.cross(earth_angular_velocity_vector, np.cross(earth_angular_velocity_vector, position))
    return coriolis, centrifugal

def euler(dt, fuel_mass, rocket_mass, thrust, position, velocity):
    earth_mass = 5.972e24
    gravity = -const.G * earth_mass * position / max(np.linalg.norm(position) ** 3, 1e-8)
    coriolis, centrifugal = calculate_non_inertial_forces(velocity, position)

    mass = rocket_mass + fuel_mass
    new_acceleration = thrust / max(mass, 1e-8) + gravity + coriolis + centrifugal

    new_velocity = velocity + new_acceleration * dt
    new_position = position + new_velocity * dt

    return new_position, new_velocity, new_acceleration