"""
Example control_approx_linearization.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from mobotpy.models import Ackermann
from mobotpy.integration import rk_four
from scipy import signal
from path_planning import Path_Generator
import pandas as pd
# Set the simulation time [s] and the sample period [s]
SIM_TIME = 15.0
T = 1/100

# Create an array of time values [s]

#N = np.size(t)

# %%
# COMPUTE THE REFERENCE TRAJECTORY

# Radius of the circle [m]
R = 10
L = 0.26
W = 0.17


# Angular rate [rad/s] at which to traverse the circle
OMEGA = 0.1

L = 0.26
W = 0.17
spot_length = 0.8
path_gen = Path_Generator(car_length=L,obst_tol=0,target_tol=0.2,radius=1)
obs_1 = [[-0.05,0],[0.05,0.25]]
obs_2 = [[-0.05,-spot_length-0.25],[0.05,-spot_length]]
obs_3 = [[obs_1[1][0]+W/2,obs_1[0][1]-L],[obs_1[1][0]+3/2*W,obs_1[0][1]]]
path_gen.set_obstacles(obs_1)
path_gen.set_obstacles(obs_2)
path_gen.set_obstacles(obs_3)
q_init = np.array([0.3,-1,np.pi/2,0])
q_target = np.array([0,0.1,np.pi/2,0])
#angles_i = np.array([]) #psi, phi
#angles_f = np.array([])
# X1,Y1, Psi1, Phi1 = path_gen.generate_path(q_init,q_target,50,v=0.07,w=W)

# M = np.zeros((len(X1),3))
# M[:,0] = X1
# M[:,1] = Y1
# M[:,2] = Psi1
# df = pd.DataFrame(M, columns = ['x','y','psi'])
# df.to_csv('path.csv',index=False)

#q_init2 = [X1[-1], Y1[-1],Psi1[-1],Phi1[-1]]
#path,X2,Y2, Psi2, Phi2 = path_gen.generate_path(q_init2,q_target2,25,v=-5,w=0)
path = pd.read_csv('/home/dimitria/demo/notebookenv/path_parallel_2.csv')

X1 = path.loc[:,'x'].tolist()
Y1 = path.loc[:,'y'].tolist()
Psi1 = path.loc[:,'psi'].tolist()
#path,X1,Y1, Psi1, Phi1 = path_gen.generate_path([X1[-1],Y1[-1],Psi1[-1],Phi1[-1]]
#,[0.1,0.1,np.pi/2,0],0.1,v=0.11,w=0)



X = np.asarray(X1) #-L*np.cos(np.asarray(Psi1))
Y = np.asarray(Y1) #-L*np.sin(np.asarray(Psi1))
Psi = np.array(Psi1)
#Phi = Phi1
print('path generated', X[-1],Y[-1])
N = np.size(X)
t = np.arange(0.0, SIM_TIME, SIM_TIME/N)
x_d = np.zeros((4, N))
u_d = np.zeros((2, N))
xi_d = np.zeros((6, N))
ddz_d = np.zeros((3, N))
# for k in range(0, N):
x_d[0, :] = X
x_d[1, :] = Y#R * (1 - np.cos(OMEGA * t[k]))
x_d[2, :] = Psi+np.pi 
x_d[3, :] = 0
u_d[0, :] = -0.07#R * OMEGA
u_d[1, :] = 0#OMEGA

# Pre-compute the extended system reference trajectory
for k in range(0, N):
    xi_d[0, k] = x_d[0, k]
    xi_d[1, k] = x_d[1, k]
    xi_d[2, k] = x_d[2, k]
    xi_d[3, k] = u_d[0, k] * np.cos(x_d[2, k])
    xi_d[4, k] = u_d[0, k] * np.sin(x_d[2, k])
    xi_d[5, k] = u_d[0, k]/L * np.tan(x_d[3, k])

# Pre-compute the extended system reference acceleration
for k in range(0, N):
    ddz_d[0, k] = 0#-u_d[0, k] * u_d[1, k] * np.sin(x_d[2, k])
    ddz_d[1, k] = 0#u_d[0, k] * u_d[1, k] * np.cos(x_d[2, k])
    ddz_d[2, k] = 0#u_d[0, k] * u_d[1, k] * np.cos(x_d[2, k])
# VEHICLE SETUP

# Set the track length of the vehicle [m]

# Create a vehicle object of type DiffDrive


# %%
vehicle = Ackermann(L,W)
# SIMULATE THE CLOSED-LOOP SYSTEM

# Initial conditions
x_init = np.array([0.3,-1,-np.pi/2,0])

# Setup some arrays
x = np.zeros((4, N))
xi = np.zeros((6, N))
u = np.zeros((2, N))
x[:, 0] = x_init

# Set the initial speed [m/s] to be non-zero to avoid singularity
w = np.zeros(2)
u[0, 0] = u_d[0, 0]

# Initial extended state
xi[0, 0] = x_init[0]
xi[1, 0] = x_init[1]
xi[2, 0] = x_init[2]
xi[3, 0] = u_d[0, 0] * np.cos(x_init[2])
xi[4, 0] = u_d[0, 0] * np.sin(x_init[2])
xi[5, 0] = u_d[0, 0]/L *np.tan(x_init[3])

# Defined feedback linearized state matrices
A = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]
              , [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
B = np.array([[0, 0, 0], [0, 0, 0],[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Choose pole locations for closed-loop linear system
p = np.array([-1.0, -2.0, -2.5, -1.5, -0.9, -2])
K = signal.place_poles(A, B, p)
for k in range(1, N):

    # Simulate the differential drive vehicle motion
    x[:, k] = rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T)
    if x[3,k] < -0.6:
        x[3,k] = -0.6
    elif x[3,k] > 0.6:
        x[3,k] = 0.6

    # Update the extended system states
    xi[0, k] = x[0, k]
    xi[1, k] = x[1, k]
    xi[2, k] = x[2, k]
    print(k, u_d[0, 0])
    xi[3, k] = u[0,k-1] * np.cos(x[2, k])
    xi[4, k] = u[0,k-1] * np.sin(x[2, k])
    xi[5, 0] = u[0,k-1]/L * np.tan(x[3, k])

    # Compute the extended linear system input control signals
    eta = K.gain_matrix @ (xi_d[:, k - 1] - xi[:, k - 1]) + ddz_d[:, k - 1]

    # Compute the new (unicycle) vehicle inputs
    psi_d = u[0, k -1]/L * np.tan(x[3, k - 1])
    B_n = np.array(
        [
            [np.cos(x[2, k - 1]), -u[0, k-1]*psi_d*np.sin(x[2, k - 1]) , 1],
            [np.sin(x[2, k - 1]), u[0, k-1]*psi_d*np.cos(x[2, k - 1]), 1],
            [1/L * np.tan(x[3, k - 1]), u[0,k-1]/(L*np.cos(x[3, k - 1])**2), 1]
        ]
    )
    print(np.linalg.det(B_n))
    B_inv = np.linalg.inv(B_n)
    w = B_inv @ eta
    u[0, k] = u[0, k] + T * w[0]
    u[1, k] = w[1]

    # Convert unicycle inputs to differential drive wheel speeds
    #print(x.shape)
print('controller', x[:2,-1])
phi = vehicle.ackermann(x)

# Plot the states as a function of time
fig1 = plt.figure(1)
fig1.set_figheight(6.4)
ax1a = plt.subplot(411)
plt.plot(t, x[0, :])
plt.grid(color="0.95")
plt.ylabel(r"$x$ [m]")
plt.setp(ax1a, xticklabels=[])
ax1b = plt.subplot(412)
plt.plot(t, x[1, :])
plt.grid(color="0.95")
plt.ylabel(r"$y$ [m]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(413)
plt.plot(t, x[2, :] * 180.0 / np.pi)
plt.grid(color="0.95")
plt.ylabel(r"$\theta$ [deg]")
plt.setp(ax1c, xticklabels=[])
ax1c = plt.subplot(414)
plt.plot(t, phi[0,:] * 180.0 / np.pi, "C1", label=r"$\phi_L$")
plt.plot(t, phi[1,:] * 180.0 / np.pi, "C2", label=r"$\phi_R$")
plt.grid(color="0.95")
plt.ylabel(r"$\phi_L,\phi_R$ [deg]")
plt.xlabel(r"$t$ [s]")
plt.legend()


# Save the plot
#plt.savefig("../agv-book/figs/ch3/ackermann_kinematic_fig1.pdf")

# Plot the position of the vehicle in the plane
fig2 = plt.figure(2)
path_gen.plot_path(X,Y)
plt.plot(X,Y)
plt.plot(x[0, :], x[1, :])
plt.axis("equal")
X_BL, Y_BL, X_BR, Y_BR, X_FL, Y_FL, X_FR, Y_FR, X_BD, Y_BD = vehicle.draw(
    x[0, 0], x[1, 0], x[2, 0], phi[0,0], phi[1,0]
)
plt.fill(X_BL, Y_BL, "k")
plt.fill(X_BR, Y_BR, "k")
plt.fill(X_FR, Y_FR, "k")
plt.fill(X_FL, Y_FL, "k")
plt.fill(X_BD, Y_BD, "C2", alpha=0.5, label="Start")
X_BL, Y_BL, X_BR, Y_BR, X_FL, Y_FL, X_FR, Y_FR, X_BD, Y_BD = vehicle.draw(
    x[0, N - 1], x[1, N - 1], x[2, N - 1], phi[0,N - 1], phi[1,N - 1]
)
plt.fill(X_BL, Y_BL, "k")
plt.fill(X_BR, Y_BR, "k")
plt.fill(X_FR, Y_FR, "k")
plt.fill(X_FL, Y_FL, "k")
plt.fill(X_BD, Y_BD, "C3", alpha=0.5, label="End")
plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$y$ [m]")
plt.legend()

# Save the plot
#plt.savefig("../agv-book/figs/ch3/ackermann_kinematic_fig2.pdf")

# Show all the plots to the screen
plt.show()
fig3 = plt.figure(3)
plt.plot(np.asarray(Psi)*180/np.pi)
plt.plot(np.asarray(Phi)*180/np.pi)
plt.show()
# MAKE AN ANIMATION

