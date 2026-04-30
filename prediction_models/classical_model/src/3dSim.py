import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import constants as const
import isacalc as isa

# ----------------------------- Stale -----------------------------

burn_time = 8                               # s
rocket_mass = 10                            # kg
fuel_mass = 6                              # kg
fuel_consumption = fuel_mass / burn_time   # kg/s
earth_mass = 5.972e24   # kg
earth_radius = 6371009  # m
latitude = 54.371552353039895
longitude = 18.613586492761833
phi = math.radians(latitude)
lam = math.radians(longitude)
earth_angular_velocity = 7.2921159e-5
area = 0.05                                 # m^2
air_density_ground = 1.225                  # kg/m^3
air_molar_mass = 0.0289644
molar_gas_const = 8.314462618
air_temperature = const.zero_Celsius + 20
pitch = math.radians(0)                  # pionowo
angle_rate = 0.0005 # rad/s
thrust = 400    # todo ciag zalezy od cisnienia
dt = 0.01 # s
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

# ---------------------------- Wykresy ----------------------------

apogeum = max(position_z_history)
print("Apogeum: ", apogeum)

max_q = max(dynamic_pressure_history)
print("Max-q wartosc ", max_q)
max_q_index = dynamic_pressure_history.index(max_q)
max_q = time_history[max_q_index]
print("Max-q czas: ", max_q)
max_q = position_z_history[max_q_index]
print("Max-q wysokosc: ", max_q)


fig, ax = plt.subplots(7, 1,  figsize=(30, 30))

ax[0].plot(time_history, velocity_history)
ax[0].axvline(x = burn_time, color = 'b')
ax[0].set_ylabel("Velocity")

ax[1].plot(time_history, acceleration_history)
ax[1].axvline(x = burn_time, color = 'b')
ax[1].set_ylabel("Acceleration")

ax[2].plot(time_history, fuel_history)
ax[2].axvline(x = burn_time, color = 'b')
ax[2].set_ylabel("Fuel")

ax[3].plot(time_history, gravity_history)
ax[3].axvline(x = burn_time, color = 'b')
ax[3].set_ylabel("Gravity")

ax[4].plot(time_history, drag_history)
ax[4].axvline(x = burn_time, color = 'b')
ax[4].set_ylabel("Drag")

ax[5].plot(time_history, air_density_history)
ax[5].axvline(x = burn_time, color = 'b')
ax[5].set_ylabel("Air Density")

ax[6].plot(time_history, position_z_history)
ax[6].axvline(x = burn_time, color = 'b')
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

# todo rakieta to nie punkt materialny
# todo yaw/roll
# todo scipy.integrate      solve_ivp