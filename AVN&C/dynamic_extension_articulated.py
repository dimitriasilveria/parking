"""
Example dynamic_extension_tracking.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from mobotpy.models import Articulated
from mobotpy.integration import rk_four
from scipy import signal

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 30.0
T = 0.04

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# COMPUTE THE REFERENCE TRAJECTORY

# Radius of the circle [m]
R = 10

# Angular rate [rad/s] at which to traverse the circle
OMEGA = 0.1

# %%
# VEHICLE SETUP

# Set the track length of the vehicle [m]
ELL = 1.0
a = 1.975
b = 3.850
# VEHICLE SETUP
ell_W_r = ell_W_f = a+b #vehicle width

# Set the track length of the vehicle [m]
ell_T_r = b
ell_T_f = a
# Create a vehicle object of type DiffDrive
vehicle = Articulated(ell_T_r, ell_T_f, ell_W_f)

# Pre-compute the desired trajectory
x_d = np.zeros((4, N))
u_d = np.zeros((2, N))
xi_d = np.zeros((4, N))
ddz_d = np.zeros((2, N))
for k in range(0, N):
    x_d[0, k] = R * np.sin(OMEGA * t[k])
    x_d[1, k] = R * (1 - np.cos(OMEGA * t[k]))
    x_d[2, k] = OMEGA * t[k]
    u_d[0, k] = R * OMEGA
    u_d[1, k] = 0

# Pre-compute the extended system reference trajectory
for k in range(0, N):
    xi_d[0, k] = x_d[0, k]
    xi_d[1, k] = x_d[1, k]
    xi_d[2, k] = u_d[0, k] * np.cos(x_d[2, k])
    xi_d[3, k] = u_d[0, k] * np.sin(x_d[2, k])

# Pre-compute the extended system reference acceleration
for k in range(0, N):
    c = u_d[0, k]/(b+a*np.sin(x_d[3, k]))
    ddz_d[0, k] = -b*u_d[1, k]*np.sin(x_d[2, k])*c - (u_d[0, k]*np.sin(x_d[2, k])*np.sin(x_d[3, k]))*c
    ddz_d[1, k] = -b*u_d[1, k]*np.cos(x_d[2, k])*c - (u_d[0, k]*np.cos(x_d[2, k])*np.sin(x_d[3, k]))*c

# %%
# SIMULATE THE CLOSED-LOOP SYSTEM

# Initial conditions
x_init = np.zeros(4)
x_init[0] = 0.0
x_init[1] = 10.0
x_init[2] = 0.0
x_init[3] = 0.0
# Setup some arrays
x = np.zeros((4, N))
xi = np.zeros((4, N))
u = np.zeros((2, N))
x[:, 0] = x_init

# Set the initial speed [m/s] to be non-zero to avoid singularity
w = np.zeros(2)
u_unicycle = np.zeros(2)
u_unicycle[0] = u_d[0, 0]

# Initial extended state
xi[0, 0] = x_init[0]
xi[1, 0] = x_init[1]
xi[2, 0] = u_d[0, 0] * np.cos(x_init[2])
xi[3, 0] = u_d[0, 0] * np.sin(x_init[2])

# Defined feedback linearized state matrices
A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

# Choose pole locations for closed-loop linear system
p = np.array([-1.0, -2.0, -2.5, -1.5])
K = signal.place_poles(A, B, p)

for k in range(1, N):

    # Simulate the vehicle motion
    x[:, k] = rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T)

    # Update the extended system states
    xi[0, k] = x[0, k]
    xi[1, k] = x[1, k]
    xi[2, k] = u_unicycle[0] * np.cos(x[2, k])
    xi[3, k] = u_unicycle[0] * np.sin(x[2, k])

    # Compute the extended linear system input control signals
    eta = K.gain_matrix @ (xi_d[:, k - 1] - xi[:, k - 1]) + ddz_d[:, k - 1]

    # Compute the new (unicycle) vehicle inputs
    aux = 1/(b+a*np.cos(x[3,k-1]))
    A_0 = np.array([
        [u_unicycle[0]**2*np.sin(x[2,k-1])*np.sin(x[3,k-1])*aux],
        [u_unicycle[0]**2*np.cos(x[2,k-1])*np.sin(x[3,k-1])*aux]
    ])
    B = np.array(
        [
            [np.cos(x[2,k-1]), -aux*np.sin(x[2,k-1])*b*u_unicycle[0]],
            [np.sin(x[2,k-1]), aux*np.cos(x[2,k-1])*b*u_unicycle[0]]
        ]
    )
    B_inv = np.linalg.inv(B)
    w = B_inv @ (eta -np.squeeze(A_0))
    u_unicycle[0] = u_unicycle[0] + T * w[0]
    u_unicycle[1] = w[1]

    # Convert unicycle inputs to differential drive wheel speeds
    u[:, k] = u_unicycle[:]

# %%
# MAKE PLOTS

# Plot the states as a function of time
fig1 = plt.figure(1)
fig1.set_figheight(6.4)
ax1a = plt.subplot(411)
plt.plot(t, x_d[0, :], "C1--")
plt.plot(t, x[0, :], "C0")
plt.grid(color="0.95")
plt.ylabel(r"$x$ [m]")
plt.setp(ax1a, xticklabels=[])
plt.legend(["Desired", "Actual"])
ax1b = plt.subplot(412)
plt.plot(t, x_d[1, :], "C1--")
plt.plot(t, x[1, :], "C0")
plt.grid(color="0.95")
plt.ylabel(r"$y$ [m]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(413)
plt.plot(t, x_d[2, :] * 180.0 / np.pi, "C1--")
plt.plot(t, x[2, :] * 180.0 / np.pi, "C0")
plt.grid(color="0.95")
plt.ylabel(r"$\theta$ [deg]")
plt.setp(ax1c, xticklabels=[])
ax1d = plt.subplot(414)
plt.step(t, x_d[3, :], "C2", where="post", label="$v_L$")
plt.step(t, x[3, :], "C3", where="post", label="$v_R$")
plt.grid(color="0.95")
plt.ylabel(r"$\phi$ [m/s]")
plt.xlabel(r"$t$ [s]")
plt.legend()

# Save the plot
#plt.savefig("../agv-book/figs/ch4/dynamic_extension_tracking_fig1.pdf")

# Plot the position of the vehicle in the plane
# Plot the position of the vehicle in the plane
# fig2 = plt.figure(2)
# plt.plot(x_d[0, :], x_d[1, :], "C1--", label="Desired")
# plt.plot(x[0, :], x[1, :], "C0", label="Actual")
# plt.axis("equal")
# X_L, Y_L, X_R, Y_R, X_F, Y_F, X_BD, Y_BD = vehicle.draw(x[0, 0], x[1, 0], x[2, 0],x[3, 0])
# plt.fill(X_L, Y_L, "k")
# plt.fill(X_R, Y_R, "k")
# plt.fill(X_BD, Y_BD, "C2", alpha=0.5, label="Start")
# X_L, Y_L, X_R, Y_R, X_F, Y_F, X_BD, Y_BD= vehicle.draw(
#     x[0, N - 1], x[1, N - 1], x[2, N - 1], x[3, N - 1]
# )
# plt.fill(X_L, Y_L, "k")
# plt.fill(X_R, Y_R, "k")
# plt.fill(X_BD, Y_BD, "C4", alpha=0.5, label="End")
# plt.xlabel(r"$x$ [m]")
# plt.ylabel(r"$y$ [m]")
# plt.legend()


fig2 = plt.figure(2)
plt.plot(x_d[0, :], x_d[1, :], "C1--", label="Desired")
plt.plot(x[0, :], x[1, :], "C0", label="Actual")
plt.axis("equal")
X_L, Y_L, X_R, Y_R, X_F, Y_F, X_B, Y_B = vehicle.draw(
    x[0, 0], x[1, 0], x[2, 0], x[3, 0]
)
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_F, Y_F, "k")
plt.fill(X_B, Y_B, "C2", alpha=0.5, label="Start")
X_L, Y_L, X_R, Y_R, X_F, Y_F, X_B, Y_B = vehicle.draw(
    x[0, N - 1], x[1, N - 1], x[2, N - 1], x[3, N - 1]
)
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_F, Y_F, "k")
plt.fill(X_B, Y_B, "C3", alpha=0.5, label="End")
plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$y$ [m]")
plt.legend()
# Save the plot
#plt.savefig("../agv-book/figs/ch4/dynamic_extension_tracking_fig2.pdf")

# Show all the plots to the screen
plt.show()



# # Show animation in HTML output if you are using IPython or Jupyter notebooks
# plt.rc('animation', html='jshtml')
# display(ani)
# plt.close()