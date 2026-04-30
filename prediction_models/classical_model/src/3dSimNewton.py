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
    fuelMass = max(fuel_mass - fuel_consumption * dt, 0)

    if flight_time < burn_time and fuelMass > 0:
        thrust_vector = CalculateThrust(position_global)
    else:
        thrust_vector = np.zeros(3)

    new_gravity = -const.G * earth_mass * position_global / max(np.linalg.norm(position_global) ** 3, 1e-8)
    coriolis, centrifugal = CalculateNonInertialForces(velocity_global, position_global)

    mass = rocket_mass + fuelMass
    new_acceleration = (thrust_vector) / max(mass, 1e-8) + new_gravity + coriolis + centrifugal

    new_velocity = velocity_global + new_acceleration * dt
    new_position = position_global + new_velocity * dt

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

    position_global, velocity_global, acceleration_local, gravity_local = Newton(dt, fuel_mass, flight_time)

    dynamic_pressure = atm.calculate(h=height)[2] * np.linalg.norm(velocity_global) ** 2 / 2

    flight_time += dt