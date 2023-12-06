"""
Example PF_range.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from mobotpy.integration import rk_four
from mobotpy.models import DiffDrive, Articulated
from scipy.stats import chi2

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 30.0
T = 0.04

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# VEHICLE SETUP

a = 1.975
b = 3.850
# VEHICLE SETUP
ell_W_r = ell_W_f = 1.0 #vehicle width

# Set the track length of the vehicle [m]
ell_T_r = b
ell_T_f = a
# Create a vehicle object of type DiffDrive
vehicle = Articulated(ell_T_r, ell_T_f, ell_W_f)

# SENSOR MODELS

# Uncertainty in wheel speed measurements [m/s]
SIGMA_SPEED = 0.025
SIGMA_PHI = 0.018
# Set the range [m] sensor uncertainty
SIGMA_TRANSM = 0.23

# %%
# DEAD RECKONING EXAMPLE

# Set the number of particles to use
M = 1000

# Create an array of particles for each time index
x_pf = np.zeros((4, M, N))

# Set the covariance matrices
Q = np.diag([SIGMA_SPEED**2, 2*SIGMA_PHI**2/T**2])
# Initialize the vehicle's true state
x = np.zeros((4, N))

# Initialize a estimated pose estimate
x_hat = np.zeros((4, N))

# Initialize a covariance matrix
P_hat = np.zeros((4, 4, N))

# Set the initial process covariance
P_hat[:, :, 0] = np.diag(np.square([5.0, 5.0, 3.5, 3.5]))

# Initialize the first particles on the basis of the initial uncertainty
for i in range(1, M):
    x_pf[:, i, 0] = x_hat[:, 0] + np.sqrt(P_hat[:, :, 0]) @ np.random.standard_normal(4)

#Initialize the first particles on a uniform distribution over the space
for i in range(1, M):
    x_pf[:, i, 0] = 100 * np.random.uniform(-1, 1, 4)

for i in range(1, N):

    # Compute some inputs (i.e., drive around)
    v = np.array([0.1, -0.01])

    # Run the vehicle motion model
    x[:, i] = rk_four(vehicle.f, x[:, i - 1], v, T)

    # Propagate each particle through the motion model
    for j in range(0, M):

        # Model the proprioceptive sensors (i.e., speed and turning rate)
        v_m = v + np.sqrt(Q) @ np.random.standard_normal(2)

        # Propagate each particle
        x_pf[:, j, i] = rk_four(vehicle.f, x_pf[:, j, i - 1], v_m, T)

# Plot the results of the dead reckoning example
plt.figure(1)
plt.plot(x_pf[0, :, 0], x_pf[1, :, 0], ".", label="Particles", alpha=0.2)
for k in range(1, N, 1):
    plt.plot(x_pf[0, :, k], x_pf[1, :, k], ".", alpha=0.2)
plt.plot(x[0, :], x[1, :], "C0", label="Actual path")
plt.axis("equal")
plt.xlabel("$x$ [m]")
plt.ylabel("$y$ [m]")
plt.legend()
plt.show()

# %%
# BUILD A MAP OF TRANSMITTERS IN THE VEHICLE'S ENVIRONMENT

# Number of transmitters
M = 20

# Map size [m]
D_MAP = 5

# Randomly place features in the map
f_map = np.zeros((2, M))
for i in range(0, M):
    f_map[:, i] = D_MAP * (2.0 * np.random.rand(2)-1)

#plt.figure(2)
#plt.plot(f_map[0, :], f_map[1, :], "C2*", label="Feature")
#plt.axis("equal")
#plt.xlabel("$x$ [m]")
#plt.ylabel("$y$ [m]")
#plt.legend()
#plt.show()

# Create a transmitter sensor model
def transmitter_sensor(x, transm, R):
    #x is the vector with the vehicle coordinates
    #transm is the vector with each transmitter's location
    # Define how many total features are available
    #m = np.shape(transm)[1]
    alpha=12
    beta=0.03

    m_k = np.shape(transm)[1]
    y = np.zeros(m_k)

        # Compute the range and bearing to all features (including sensor noise)
    for i in range(0, m_k):
        # Range measurement [m]
        r = np.sqrt((transm[0, i] - x[0]) ** 2 + (transm[1, i] - x[1]) ** 2)
        y[i] = alpha*np.exp(-beta*r) + np.sqrt(R[0, 0]) * np.random.randn(1)
        
    # Return the range and bearing to features in y with indices in a
    return y
def h(x, transm):
    #x is the vector with the vehicle coordinates
    #transm is the vector with each transmitter's location
    # Define how many total features are available
    #m = np.shape(transm)[1]

    alpha=12
    beta=0.03
    m_k = np.shape(transm)[1]
    y = np.zeros(m_k)

        # Compute the range and bearing to all features (including sensor noise)
    for i in range(0, m_k):
        # Range measurement [m]
        r = np.sqrt((transm[0, i] - x[0]) ** 2 + (transm[1, i] - x[1]) ** 2)
        y[i] = alpha*np.exp(-beta*r)

    # Return the range and bearing to features in y with indices in a
    return y

# %%
# PARTICLE FILTER FUNCTIONS


def pf_resample(x_pf, x_likelihood):
    """Function to resample particles."""

    # Initialize a set of output particles
    x_pf_resampled = np.zeros((4, M))

    # Do the resampling (one way)
    indices = np.searchsorted(np.cumsum(x_likelihood), np.random.random_sample(M))
    for j in range(0, M):
        x_pf_resampled[:, j] = x_pf[:, indices[j]]

    # Return the resampled particles
    return x_pf_resampled


def articulated_pf(x_pf, v, y, f_map, Q, R, T):
    """Particle filter for articulated vehicle function."""

    # Find the number of transmitters
    m_k = y.shape[0]

    # Initialize the output
    x_pf_new = np.zeros((4, M))

    # Propagate the particles through the vehicle model (i.e., a priori step)
    for j in range(0, M):

        # Model the wheel speed measurements
        v_m = v + np.sqrt(Q) @ np.random.standard_normal(2)

        # Propagate each particle
        x_pf_new[:, j] = rk_four(vehicle.f, x_pf[:, j], v_m, T)

    # Set likelihoods all equal to start the a posteriori step
    x_likelihood = 1.0 / M * np.ones(M)

    # Compute the relative likelihood
    if m_k > 1:

        # Set up some arrays
        y_hat = np.zeros((m_k, M))
        y_dif = np.zeros(m_k)

        # Compute some needed matrices
        #R_inv = np.linalg.inv(np.kron(np.identity(m_k), R))
        #R_det = np.linalg.det(np.kron(np.identity(m_k), R))
        R_inv = np.linalg.inv(R)
        R_det = np.linalg.det(R)
        for j in range(0, M):

            # For each visible beacon find the expected measurement
            y_hat[:,j] = h(x_pf_new[:, j],f_map)

            # Compute the relative likelihoods
            y_dif = y - y_hat[:, j]
            x_likelihood[j] = (
                1.0
                / ((2.0 * np.pi) ** (m_k / 2) * np.sqrt(R_det))
                * np.exp(-0.5 * y_dif.T @ R_inv @ y_dif)
            )

        # Normalize the likelihoods
        x_likelihood /= np.sum(x_likelihood)

        # Generate a set of a posteriori particles by re-sampling on the basis of the likelihoods
        x_pf_new = pf_resample(x_pf_new, x_likelihood)

    return x_pf_new


# %%
# RUN THE PARTICLE FILTER SIMULATION

# Initialize some arrays
x_pf = np.zeros((4, M, N))
x_hat = np.zeros((4, N))
P_hat = np.zeros((4, 4, N))

# Initialize the initial guess to a location different from the actual location
x_hat[:, 0] = x[:, 0] + np.array([0, 5.0, 0.1, 0.1])

# Set some initial conditions
P_hat[:, :, 0] = np.diag(np.square([5.0, 5.0, 3.5, 3.5]))

# Set the covariance matrices
Q = np.diag([SIGMA_SPEED**2, 2*SIGMA_PHI**2/T**2])

# Set sensor range
R_MAX = 25.0
R_MIN = 1.0

# Set the range and bearing covariance
R = np.eye(M)
R = SIGMA_TRANSM ** 2*R
#R = np.diag([SIGMA_TRANSM**2])

# Initialize the first particles on the basis of the initial uncertainty
for i in range(1, M):
    x_pf[:, i, 0] = x_hat[:, 0] + np.sqrt(P_hat[:, :, 0]) @ np.random.standard_normal(4)

# Initialize the first particles on the basis of the initial uncertainty
# for i in range(1, M):
#     x_pf[:, i, 0] = 100 * np.random.uniform(-1, 1, 3)

# Simulate for each time
for i in range(1, N):

    # Compute some inputs (i.e., drive around)
    v = np.array([0.1, -0.01])

    # Run the vehicle motion model
    x[:, i] = rk_four(vehicle.f, x[:, i - 1], v, T)

    # Run the range and bearing sensor model
    y_m = transmitter_sensor(x[:, i], f_map, R)

    # Run the particle filter
    x_pf[:, :, i] = articulated_pf(x_pf[:, :, i - 1], v, y_m, f_map, Q, R, T)

# %%
# PLOT THE SIMULATION OUTPUTS

# Plot the results of the particle filter simulation
plt.figure(3)
plt.plot(x_pf[0, :, 0], x_pf[1, :, 0], ".", label="Particles", alpha=0.2)
for k in range(1, N, 1):
    plt.plot(x_pf[0, :, k], x_pf[1, :, k], ".", alpha=0.2)
plt.plot(x[0, :], x[1, :], "C0", label="Actual path")
plt.plot(f_map[0, :], f_map[1, :], "C2*", label="Feature")
plt.axis("equal")
plt.xlabel("$x$ [m]")
plt.ylabel("$y$ [m]")
plt.legend()
plt.show()

# Compute the mean errors and estimated covariance bounds
for i in range(0, N):
    x_hat[0, i] = np.mean(x_pf[0, :, i])
    x_hat[1, i] = np.mean(x_pf[1, :, i])
    x_hat[2, i] = np.mean(x_pf[2, :, i])

for i in range(0, N):
    P_hat[:, :, i] = np.cov(x_pf[:, :, i])

# Find the scaling factors for plotting covariance bounds
ALPHA = 0.01
s1 = chi2.isf(ALPHA, 1)
s2 = chi2.isf(ALPHA, 2)

fig5 = plt.figure(4)
sigma = np.zeros((4, N))
ax1 = plt.subplot(411)
sigma[0, :] = np.sqrt(s1 * P_hat[0, 0, :])
plt.fill_between(t, -sigma[0, :], sigma[0, :], color="C1", alpha=0.2)
plt.plot(t, x[0, :] - x_hat[0, :], "C0")
plt.ylabel(r"$e_1$ [m]")
plt.setp(ax1, xticklabels=[])
ax2 = plt.subplot(412)
sigma[1, :] = np.sqrt(s1 * P_hat[1, 1, :])
plt.fill_between(t, -sigma[1, :], sigma[1, :], color="C1", alpha=0.2)
plt.plot(t, x[1, :] - x_hat[1, :], "C0")
plt.ylabel(r"$e_2$ [m]")
plt.setp(ax2, xticklabels=[])
ax3 = plt.subplot(413)
sigma[2, :] = np.sqrt(s1 * P_hat[2, 2, :])
plt.fill_between(t, -sigma[2, :], sigma[2, :], color="C1", alpha=0.2)
plt.plot(t, x[2, :] - x_hat[2, :], "C0")
plt.ylabel(r"$e_3$ [rad]")
plt.xlabel(r"$t$ [s]")
ax4 = plt.subplot(414)
sigma[3, :] = np.sqrt(s1 * P_hat[3, 3, :])
plt.fill_between(t, -sigma[3, :], sigma[3, :], color="C1", alpha=0.2)
plt.plot(t, x[3, :] - x_hat[3, :], "C0")
plt.ylabel(r"$e_3$ [rad]")
plt.xlabel(r"$t$ [s]")
plt.show()

# %%