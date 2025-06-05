import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# === INPUT PARAMETERS ===
mass = 0.625  # kg, example
drag_coefficient = 1.5  # lumped k = 0.5 * rho * C_d * A (kg/m)
rho = 1.17
area = 0.033*0.033*3.1415
g = 9.81  # m/s²
numOfMotors = 2

# === SIMULATION PARAMETERS ===
t_max = 7.0  # total simulation time in seconds
dt = 0.001  # time step
n_steps = int(t_max / dt)

# Thrust data (time in s, thrust in N)

KlimaC6 = np.array([    
[0, 0],
[0.046, 0.953],
[0.168, 5.259],
[0.235, 10.023],
[0.291, 15.00],
[0.418, 9.87],
[0.505, 7.546],
[0.582, 6.631],
[0.679, 6.136],
[0.786, 5.716],
[1.26, 5.678],
[1.357, 5.488],
[1.423, 4.992],
[1.469, 4.116],
[1.618, 1.22],
[1.701, 0.0],
])


KlimaD3 = np.array([  
[0, 0],
[0.073, 0.229],
[0.178, 0.686],
[0.251, 1.287],
[0.313, 2.203],
[0.375, 3.633],
[0.425, 5.006],
[0.473, 6.465],
[0.556, 8.181],
[0.603, 9.01],
[0.655, 6.922],
[0.698, 5.463],
[0.782, 4.291],
[0.873, 3.576],
[1.024, 3.146],
[1.176, 2.946],
[5.282, 2.918],
[5.491, 2.832],
[5.59, 2.517],
[5.782, 1.859],
[5.924, 1.287],
[6.061, 0.715],
[6.17, 0.286],
[6.26, 0.0],
])

thrust_data = KlimaD3


# Interpolate thrust curve
times = thrust_data[:, 0]
thrusts = thrust_data[:, 1]
thrust_func = interp1d(times, thrusts, bounds_error=False, fill_value=0.0)



# === STATE VARIABLES ===
velocity = 0.0
altitude = 0.0

time_array = np.zeros(n_steps)
altitude_array = np.zeros(n_steps)
velocity_array = np.zeros(n_steps)
thrust_array = np.zeros(n_steps)


# === SIMULATION LOOP ===
for i in range(n_steps):
    t = i * dt
    thrust = thrust_func(t)*numOfMotors
    weight = mass * g
    drag = 0.5*area*rho*drag_coefficient * velocity**2 * np.sign(velocity)

    
    # Net force = thrust - drag - weight
    net_force = thrust - drag - weight
    acceleration = net_force / mass
    
    if(acceleration < 0 and t < 1):
        acceleration = 0
    
    
    velocity += acceleration * dt
    altitude += velocity * dt
    
    # Store for plotting
    time_array[i] = t
    altitude_array[i] = altitude
    velocity_array[i] = velocity
    thrust_array[i] = thrust

    # Stop if altitude goes negative (rocket hits ground)
    if altitude < 0:
        altitude_array[i:] = 0
        break

# === PLOTTING ===
plt.figure(figsize=(10, 6))
plt.plot(time_array, altitude_array, label='Altitude (m)')
plt.plot(time_array, thrust_array, label='Velocity (m/s)', linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Altitude / Thrust")
plt.title("Rocket Altitude and Thrust Profile")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
