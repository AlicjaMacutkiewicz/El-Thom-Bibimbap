import matplotlib.pyplot as plt
import math
from scipy import constants as const

# ----------------------------- Stale -----------------------------

rocket_mass = 10                            # kg
burn_time = 8                               # s
drag_coefficient = 0.5                      # -
area = 0.05                                 # m^2
air_density_ground = 1.225                  # kg/m^3
air_molar_mass = 0.0289644
molar_gas_const = 8.314462618
air_temperature = const.zero_Celsius + 20
thrust = 400                               # N
fuel_mass = 6                              # kg
fuel_consumption = fuel_mass / burn_time   # kg/s
G = const.gravitational_constant                            # stala grawitacyjna
earth_mass = 5.972e24                      # kg
earth_radius = 6371009                     # m
climb_angle = math.pi / 2                  # rad  90 = pionowo

# ---------------------------- Zmienne ----------------------------

dt = 0.005          # s
height = 0          # m
y_position = 0      # m
x_velocity = 0      # m/s
y_velocity = 0      # m/s
acceleration = 0    # m/s^2
y_acceleration = 0  # m/s^2
flight_time = 0     # s
angle_rate = 0.0005 # rad/s

# ---------------------------- Statystyka ----------------------------

height_history = []
y_position_history = []

thrust_history = []

acceleration_history = []
fuel_history = []
gravity_history = []
drag_history = []
air_density_history = []
time_history = []

total_velocity_history = []
y_velocity_history = []
height_velocity_history = []

dynamic_pressure_history = []

free_flight_Y = 0
apogeum = 0
max_q = 0

# ---------------------------- Symulacja ----------------------------

while True:
    if height <= 0 and flight_time > 0:
        break

    climb_angle += angle_rate

    mass = rocket_mass + fuel_mass

    if flight_time < burn_time:
        fuel_mass -= fuel_consumption * dt
        fuel_mass = max(fuel_mass, 0)
    elif thrust != 0:
        thrust = 0
        free_flight_Y = y_position
        angle_rate = 0

    gravity = G * earth_mass / (height + earth_radius) ** 2

    air_density = air_density_ground * math.exp(
        -air_molar_mass * gravity * height / (molar_gas_const * air_temperature)
    )

    v = math.sqrt(x_velocity ** 2 + y_velocity ** 2)

    drag = 0.5 * drag_coefficient * area * air_density * v**2
    dynamic_pressure = air_density * v**2 / 2

    if v > 0:
        drag_y = drag * (y_velocity / v)
        drag_x = drag * (x_velocity / v)
    else:
        drag_x = 0
        drag_y = 0

    thrust_x = thrust * math.cos(climb_angle)
    thrust_y = thrust * math.sin(climb_angle)

    ax = thrust_x / mass - drag_x / mass
    ay = thrust_y / mass - drag_y / mass - gravity

    y_velocity += ax * dt
    x_velocity += ay * dt

    y_position += y_velocity * dt
    height += x_velocity * dt
    height = max(height, 0)





    # ----------- Historia ------------

    height_history.append(height)
    y_position_history.append(y_position)
    thrust_history.append(thrust)
    total_velocity_history.append(v)
    y_velocity_history.append(y_velocity)
    height_velocity_history.append(x_velocity)
    acceleration_history.append(ay)
    fuel_history.append(fuel_mass)
    gravity_history.append(gravity)
    drag_history.append(drag)
    air_density_history.append(air_density)
    time_history.append(flight_time)
    dynamic_pressure_history.append(dynamic_pressure)

    flight_time += dt

# ---------------------------- Wykresy ----------------------------

apogeum = max(height_history)
print(apogeum)

max_q_index = dynamic_pressure_history.index(max(dynamic_pressure_history))
max_q = time_history[max_q_index]
print("Max-q czas: ", max_q)
max_q = height_history[max_q_index]
print("Max-q wysokosc: ", max_q)

fig, ax = plt.subplots(9, 1,  figsize=(30, 30))

ax[0].plot(time_history, height_history)
ax[0].axvline(x = burn_time, color = 'b')
ax[0].set_ylabel("Height")

ax[1].plot(time_history, thrust_history)
ax[1].axvline(x = burn_time, color = 'b')
ax[1].set_ylabel("Thrust")

ax[2].plot(time_history, total_velocity_history)
ax[2].axvline(x = burn_time, color = 'b')
ax[2].set_ylabel("Velocity")

ax[3].plot(time_history, acceleration_history)
ax[3].axvline(x = burn_time, color = 'b')
ax[3].set_ylabel("Acceleration")

ax[4].plot(time_history, fuel_history)
ax[4].axvline(x = burn_time, color = 'b')
ax[4].set_ylabel("Fuel")

ax[5].plot(time_history, gravity_history)
ax[5].axvline(x = burn_time, color = 'b')
ax[5].set_ylabel("Gravity")

ax[6].plot(time_history, drag_history)
ax[6].axvline(x = burn_time, color = 'b')
ax[6].set_ylabel("Drag")

ax[7].plot(time_history, air_density_history)
ax[7].axvline(x = burn_time, color = 'b')
ax[7].set_ylabel("Air Density")

ax[8].plot(time_history, y_position_history)
ax[8].axvline(x = burn_time, color = 'b')
ax[8].set_ylabel("Y")

plt.xlabel("Time")
plt.show()



plt.plot(y_position_history, height_history)
plt.axvline(x = free_flight_Y, color ='b')
plt.xlabel("Y position")
plt.ylabel("Height")
plt.show()