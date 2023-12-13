"""
Example control_approx_linearization.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from mobotpy.models import DiffDrive
from mobotpy.integration import rk_four
from scipy import signal
from icecream import ic
# Set the simulation time [s] and the sample period [s]
SIM_TIME = 15.0
T = 0.01

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# VEHICLE SETUP

# Set the track length of the vehicle [m]
ELL = 0.25

# Create a vehicle object of type DiffDrive
vehicle = DiffDrive(ELL)

# %%
# COMPUTE THE REFERENCE TRAJECTORY

# Angular rate [rad/s] at which to traverse the circle
theta = 0.0
v_d=0.75
a=4
# Pre-compute the desired trajectory
x_d = np.zeros((3, N))
u_d = np.zeros((2, N))
for k in range(0, N):
    x_d[0, k] = v_d*np.cos(theta) * t[k]
    x_d[1, k] = v_d*np.sin(theta)*t[k] + a#m*(x_d[0,k])
    x_d[2, k] = theta
    u_d[0, k] = v_d
    u_d[1, k] = v_d

# %%
# SIMULATE THE CLOSED-LOOP SYSTEM

# Initial conditions
x_init = np.zeros(3)
x_init[0] = 0.0
x_init[1] = -1.0
x_init[2] = -0.1

# Setup some arrays
x = np.zeros((3, N))
u = np.zeros((2, N))
x[:, 0] = x_init

for k in range(1, N):

    # Simulate the differential drive vehicle motion
    x[:, k] = rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T)

    # Compute the approximate linearization
    V = (u_d[0, k - 1] + u_d[1, k - 1])/2
    A = np.array(
        [
            [0, 0, -V * np.sin(x_d[2, k - 1])],
            [0, 0, V * np.cos(x_d[2, k - 1])],
            [0, 0, 0],
        ]
    )
    B = np.array([[0.5*np.cos(x_d[2, k - 1]), 0.5*np.cos(x_d[2, k - 1])], [0.5*np.sin(x_d[2, k - 1]), 0.5*np.sin(x_d[2, k - 1])], [-1/ELL, 1/ELL]])
    O = np.concatenate(([B,(A@B),(A@A@B)]),axis=1)
    rank = np.linalg.matrix_rank(O)
    if rank != 3:
        ic(rank)
    # Compute the gain matrix to place poles of (A-BK) at p
    p = np.array([-2.0, -1.0, -0.5])
    K = signal.place_poles(A, B, p)

    # Compute the controls (v, omega) and convert to wheel speeds (v_L, v_R)
    u[:, k] = -K.gain_matrix @ (x[:, k - 1] - x_d[:, k - 1]) + u_d[:, k - 1]
    #u[:, k] = vehicle.uni2diff(u_unicycle)

# %%
# MAKE PLOTS


# Plot the states as a function of time
fig1 = plt.figure(1)
fig1.set_figheight(6.4)
# ax1a = plt.subplot(311)
# plt.plot(t, x_d[0, :], "C1--")
# plt.plot(t, x[0, :], "C0")
# plt.grid(color="0.95")
# plt.ylabel(r"$x$ [m]")
#plt.setp(ax1a, xticklabels=[])
#plt.legend(["Desired", "Actual"])
ax1b = plt.subplot(211)

plt.plot(t, x_d[1, :] -x_d[1, :] , "C1--",label="Desired")
plt.plot(t, x[1, :]-x_d[1, :], "C0",label="Actual")
plt.grid(color="0.95")
plt.ylabel(r"$e_l$ [m]")
plt.setp(ax1b, xticklabels=[])
plt.legend(["Desired", "Actual"])
ax1c = plt.subplot(212)
plt.plot(t, (x_d[2, :] - x_d[2, :]) * 180.0 / np.pi, "C1--",label="Desired")
plt.plot(t, x[2, :] - x_d[2, :] * 180.0 / np.pi, "C0",label="Actual")
plt.grid(color="0.95")
plt.ylabel(r"$e_h$ [deg]")
plt.setp(ax1c, xticklabels=[])
# ax1d = plt.subplot(313)
# plt.step(t, u[0, :], "C2", where="post", label="$v_L$")
# plt.step(t, u[1, :], "C3", where="post", label="$v_R$")
# plt.grid(color="0.95")
# plt.ylabel(r"$\bm{u}$ [m/s]")
# plt.xlabel(r"$t$ [s]")
# plt.legend()

# Save the plot
#plt.savefig("../agv-book/figs/ch4/control_approx_linearization_fig1.pdf")
plt.show()
# Plot the position of the vehicle in the plane
fig2 = plt.figure(2)
plt.plot(x_d[0, :], x_d[1, :], "C1--", label="Desired")
plt.plot(x[0, :], x[1, :], "C0", label="Actual")
plt.axis("equal")
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(x[0, 0], x[1, 0], x[2, 0])
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_C, Y_C, "k")
plt.fill(X_B, Y_B, "C2", alpha=0.5, label="Start")
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(
    x[0, N - 1], x[1, N - 1], x[2, N - 1]
)
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_C, Y_C, "k")
plt.fill(X_B, Y_B, "C3", alpha=0.5, label="End")
plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$y$ [m]")
plt.legend()

# Save the plot
#plt.savefig("../agv-book/figs/ch4/control_approx_linearization_fig2.pdf")

# Show all the plots to the screen
plt.show()

# %%
# MAKE AN ANIMATION

# Create and save the animation
ani = vehicle.animate_trajectory(
    x, x_d, T, True, "../agv-book/gifs/ch4/control_approx_linearization.gif"
)

# Show the movie to the screen
plt.show()

# # Show animation in HTML output if you are using IPython or Jupyter notebooks
# plt.rc('animation', html='jshtml')
# display(ani)
# plt.close()