import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(".."))
from model.mav_dynamics import MavDynamics
from tools.rotations import quaternion_to_euler

Ts = 0.01
T_final = 20.0

mav = MavDynamics(Ts)

# constant control: [da, de, dr, dt]
delta = np.array([0.0, -0.2, 0.0, 0.5])
wind = np.zeros((6, 1))
mav._update_velocity_data(wind)
F = mav._forces_moments(delta)
print("M (pitch moment) at t=0:", F[4,0])

# logging
time_hist = []
alt_hist  = []
u_hist    = []
theta_hist = []

t = 0.0
while t < T_final:
    mav.update(delta, wind)

    pn, pe, pd = mav._state[0:3, 0]
    u = mav._state[3, 0]
    quat = mav._state[6:10]
    phi, theta, psi = quaternion_to_euler(quat)
    h = -pd

    time_hist.append(t)
    alt_hist.append(h)
    u_hist.append(u)
    theta_hist.append(theta)

    t += Ts

# convert to arrays
time_hist = np.array(time_hist)

plt.figure()
plt.plot(time_hist, alt_hist)
plt.xlabel("time [s]")
plt.ylabel("altitude h [m]")
plt.grid(True)

plt.figure()
plt.plot(time_hist, u_hist)
plt.xlabel("time [s]")
plt.ylabel("u [m/s]")
plt.grid(True)

plt.figure()
plt.plot(time_hist, np.rad2deg(theta_hist))
plt.xlabel("time [s]")
plt.ylabel("theta [deg]")
plt.grid(True)

plt.show()
