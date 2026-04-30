import pandas as pd
from SimHelper import *
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

def Newton(dt, fuel_mass, flight_time):
    height = max(0, np.linalg.norm(position_global) - earth_radius)

    if height > 100000:
        new_drag = np.zeros(3)
    else:
        params = atm.calculate(h=height)
        new_drag = CalculateDrag(params, velocity_global, position_global)

    fuelMass = max(fuel_mass - fuel_consumption * dt, 0)

    if flight_time < burn_time and fuelMass > 0:
        thrust_vector = CalculateThrust(position_global)
    else:
        thrust_vector = np.zeros(3)

    new_gravity = -const.G * earth_mass * position_global / max(np.linalg.norm(position_global) ** 3, 1e-8)
    coriolis, centrifugal = CalculateNonInertialForces(velocity_global, position_global)

    mass = rocket_mass + fuelMass
    new_acceleration = (thrust_vector + new_drag) / max(mass, 1e-8) + new_gravity + coriolis + centrifugal

    new_velocity = velocity_global + new_acceleration * dt
    new_position = position_global + new_velocity * dt

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

    position_global, velocity_global, acceleration_local, gravity_local, drag_local = Newton(dt, fuel_mass, flight_time)

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

if not save:
    pd.DataFrame({
    'x': position_x_history,
    'y': position_y_history,
    'z': position_z_history
}).to_csv("newton_position.csv", index=False)

if not plot:
    plot_history(position_z_history, dynamic_pressure_history, time_history, velocity_history, burn_time, acceleration_history, fuel_history, gravity_history, drag_history, air_density_history, position_x_history, position_y_history)