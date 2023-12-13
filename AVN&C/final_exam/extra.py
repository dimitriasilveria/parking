"""
Example diffdrive_GNSS_EKF.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from mobotpy.models import DiffDrive
from mobotpy.integration import rk_four
from scipy.stats import chi2
from numpy.linalg import inv

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 30.0
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
# FUNCTION DEFINITIONS


def unicycle_GNSS_ekf(u_m,u_d, y, Q, R, e,x_d, P, T):

    # Define some matrices for the a priori step
    G = np.array([ [0.5*np.cos(e[2]+x_d[2]), 0.5*np.cos(e[2]+x_d[2])],
                  [0.5*np.sin(e[2]+x_d[2]), 0.5*np.sin(e[2]+x_d[2])], [-1/ELL, 1/ELL]])
    F = np.identity(3) + T * np.array(
        [[0, 0, -(u_m[0]+u_m[1])*0.5 * np.sin(e[2]+x_d[2])],
         [0, 0, (u_m[0]+u_m[1])*0.5 * np.cos(e[2]+x_d[2])], [0,0, 0]]
    )
    L = T * G

    # Compute a priori estimate
    G_d = np.array([[0.5*np.cos(x_d[2]), 0.5*np.cos(x_d[2])],
                    [0.5*np.sin(x_d[2]), 0.5*np.sin(x_d[2])], 
                    [-1/ELL, 1/ELL]])
    e_new = e + T*(G @ u_m - G_d@u_d)
    P_new = F @ P @ np.transpose(F) + L @ Q @ np.transpose(L)

    # Numerically help the covariance matrix stay symmetric
    P_new = (P_new + np.transpose(P_new)) / 2

    # Define some matrices for the a posteriori step
    C = np.array([[1,0,0],[0,1,0]])
    H = C
    M = np.identity(2)

    # Compute the a posteriori estimate
    K = (
        P_new
        @ np.transpose(H)
        @ inv(H @ P_new @ np.transpose(H) + M @ R @ np.transpose(M))
    )
    e_new = e_new + K @ (y - C @ e_new)
    #print(e_new)
    P_new = (np.identity(3) - K @ H) @ P_new

    # Numerically help the covariance matrix stay symmetric
    P_new = (P_new + np.transpose(P_new)) / 2
    
    # Define the function output
    #e_new = np.zeros_like(e_new)
    return e_new, P_new


# %%
# Angular rate [rad/s] at which to traverse the circle
theta = 0.1
v_d=0.75
a=0
# Pre-compute the desired trajectory
x_d = np.zeros((3, N))
u_d = np.zeros((2, N))
for k in range(0, N):
    x_d[0, k] = v_d*np.cos(theta) * t[k]
    x_d[1, k] = v_d*np.sin(theta)*t[k] + a#m*(x_d[0,k])
    x_d[2, k] = theta
    u_d[0, k] = v_d
    u_d[1, k] = v_d

u_unicycle = u_d

# SET UP INITIAL CONDITIONS AND ESTIMATOR PARAMETERS

# Initial conditions
x_init = np.zeros(3)
x_init[0] = x_d[0,0]
x_init[1] = x_d[1,0]
x_init[2] = x_d[2,0]
e_init = np.zeros(3)
# Setup some arrays for the actual vehicle
x = np.zeros((3, N))
x_est = np.zeros_like(x)
e = np.zeros((3,N))
u = np.zeros((2, N))
x[:, 0] = x_init
e[:, 0] = e_init

# Set the initial guess for the estimator
e_guess = e_init + np.array([0., 0.3, 0.])
#x_est[:,0] = e_guess + x_d[:,0]
# Set the initial pose covariance estimate as a diagonal matrix
P_guess = np.diag(np.square([0., 0.3, 0.]))

# Set the true process and measurement noise covariances
Q = np.diag([0.26**2,0.26**2])
R = np.diag([1.3**2,1.3**2])

# Initialized estimator arrays
e_hat = np.zeros((3, N))
e_hat[:, 0] = e_guess

# Measured odometry (speed and angular rate) and GNSS (x, y) signals
u_m = np.zeros((2, N))
y = np.zeros((2, N))

# Covariance of the estimate
P_hat = np.zeros((3, 3, N))
P_hat[:, :, 0] = P_guess

# Compute some inputs to just drive around





# %%
# SIMULATE AND PLOT WITH GNSS + ODOMETRY FUSION

# Find the scaling factor for plotting covariance bounds
alpha = 0.01
s1 = chi2.isf(alpha, 1)
s2 = chi2.isf(alpha, 2)

# Estimate the process and measurement noise covariances
Q_hat = Q
R_hat = R
u[:,0] = u_unicycle[:,0]
for k in range(1, N):

    # Simulate the differential drive vehicle's motion
    u[:, k] = u_unicycle[:,k]
    x[:, k] = rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T)
    e[:, k] = x[:,k] - x_d[:,k]

    # Simulate the vehicle speed estimate
    u_m[0, k] = u_unicycle[0,k] + np.power(Q[0, 0], 0.5) * np.random.randn(1)[0]

    # Simulate the angular rate gyroscope measurement
    u_m[1, k] = u_unicycle[1,k] + np.power(Q[1, 1], 0.5) * np.random.randn(1)[0]

    # Simulate the GNSS measurement
    #print(x[0,k],x_d[0,k])
    y[0, k] = e[0, k] + np.power(R[0,0], 0.5) * np.random.randn(1)[0]
    y[1, k] = e[1, k] + np.power(R[1,1], 0.5) * np.random.randn(1)[0]

    # Run the EKF estimator
    e_hat[:, k], P_hat[:, :, k] = unicycle_GNSS_ekf(
        u_m[:, k],u_d[:,k], y[:, k], Q_hat, R_hat, e_hat[:, k - 1],x_d[:,k-1], P_hat[:, :, k - 1], T
    )
    x_est[:,k] = e_hat[:,k]+x_d[:,k]

# Set the ranges for error uncertainty axes
x_range = 2.0
y_range = 2.0
theta_range = 2

plt.plot(x[:,0],x[:,1],color='blue')
plt.plot(x_est[:,0],x_est[:,1],color='red')
#plt.show()
# Plot the estimation error with covariance bounds
sigma = np.zeros((3, N))
fig4 = plt.figure(4)
ax1 = plt.subplot(311)
sigma[0, :] = np.sqrt(s1 * P_hat[0, 0, :])
plt.fill_between(
    t,
    -sigma[0, :],
    sigma[0, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - alpha)) + " \% Confidence",
)
plt.plot(t,  e_hat[0, :], "C0", label="Error")
plt.plot(t, e[0, :], "C1")
plt.ylabel(r"$e_x$ [m]")
plt.setp(ax1, xticklabels=[])
ax1.set_ylim([-x_range, x_range])
plt.grid(color="0.95")
plt.legend()
ax2 = plt.subplot(312)
sigma[1, :] = np.sqrt(s1 * P_hat[1, 1, :])
plt.fill_between(t, -sigma[1, :], sigma[1, :], color="C0", alpha=0.2)
plt.plot(t, e_hat[1, :], "C0")
plt.plot(t, e[1, :], "C1")
plt.ylabel(r"$e_L$ [m]")
plt.setp(ax2, xticklabels=[])
ax2.set_ylim([-y_range, y_range])
plt.grid(color="0.95")

ax3 = plt.subplot(313)
sigma[2, :] = np.sqrt(s1 * P_hat[2, 2, :])
plt.fill_between(t, -sigma[2, :], sigma[2, :], color="C0", alpha=0.2)
plt.plot(t, e_hat[2, :], "C0")
plt.plot(t, e[2, :], "C1")
plt.ylabel(r"$e_H$ [m]")
plt.setp(ax2, xticklabels=[])
ax3.set_ylim([-theta_range, theta_range])
plt.grid(color="0.95")



# Show the movie to the screen
plt.show()
print('animating')
# Create and save the animation
ani = vehicle.animate_estimation(
    x, x_est, P_hat, alpha, T, True, "diffdrive_GNSS_EKF.gif"
)
print('showing')
# # Show animation in HTML output if you are using IPython or Jupyter notebooks
#plt.rc('animation', html='jshtml')
plt.show()
#plt.close()