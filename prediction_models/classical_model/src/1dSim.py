import matplotlib.pyplot as plt
import math

dt = 0.1            # s
fuel_mass = 6    # kg
rocket_mass = 10  # kg
burn_time = 8   # s
fuel_consumption = fuel_mass / burn_time # kg/s
height = 0.00001        # m
thrust = 400      # N
velocity_global = 0      # m/s
acceleration = 0  # m/s^2
G = 6.67430e-11
earth_mass = 5.972e24 # kg
earth_radius = 6371009 # m
flight_time = 0   # s
drag_coefficient = 0.5
area = 0.05
air_density_ground = 1.225
air_molar_mass = 0.0289644
molar_gas_const = 8.314462618
air_temperature = 298.15

height_history = []
thrust_history = []
velocity_history = []
acceleration_history = []
fuel_history = []
gravity_history = []
drag_history = []
air_density_history = []
time_history = []

while height > 0:
    if flight_time < burn_time:
        fuel_mass -= fuel_consumption * dt
        fuel_mass = max(fuel_mass, 0)
    else:
        thrust = 0

    gravity = G * earth_mass / (height + earth_radius) ** 2

    air_density = air_density_ground * math.exp(-air_molar_mass * 9.80665 * height / (molar_gas_const * air_temperature))
    drag = drag_coefficient * area * air_density * velocity_global * abs(velocity_global) / 2

    acceleration = (thrust / (rocket_mass + fuel_mass) - gravity - drag / (rocket_mass + fuel_mass))
    velocity_global += acceleration * dt
    height += velocity_global * dt

    height_history.append(height)
    thrust_history.append(thrust)
    velocity_history.append(velocity_global)
    acceleration_history.append(acceleration)
    fuel_history.append(fuel_mass)
    gravity_history.append(gravity)
    drag_history.append(drag)
    air_density_history.append(air_density)
    time_history.append(flight_time)

    flight_time += dt

fig, ax = plt.subplots(8, 1)

ax[0].plot(time_history, height_history)
ax[0].set_ylabel("Height")

ax[1].plot(time_history, thrust_history)
ax[1].set_ylabel("Thrust")

ax[2].plot(time_history, velocity_history)
ax[2].set_ylabel("Velocity")

ax[3].plot(time_history, acceleration_history)
ax[3].set_ylabel("Acceleration")

ax[4].plot(time_history, fuel_history)
ax[4].set_ylabel("Fuel")

ax[5].plot(time_history, gravity_history)
ax[5].set_ylabel("Gravity")

ax[6].plot(time_history, drag_history)
ax[6].set_ylabel("Drag")

ax[7].plot(time_history, air_density_history)
ax[7].set_ylabel("Air Density")

plt.xlabel("Time")
plt.show()