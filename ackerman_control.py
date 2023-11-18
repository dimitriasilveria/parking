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
T = 1/50

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
X1,Y1, Psi1, Phi1 = path_gen.generate_path(q_init,q_target,50,v=0.07,w=W)

M = np.zeros((len(X1),3))
M[:,0] = X1
M[:,1] = Y1
M[:,2] = Psi1
df = pd.DataFrame(M, columns = ['x','y','psi'])
df.to_csv('path.csv',index=False)

#q_init2 = [X1[-1], Y1[-1],Psi1[-1],Phi1[-1]]
#path,X2,Y2, Psi2, Phi2 = path_gen.generate_path(q_init2,q_target2,25,v=-5,w=0)
# path = pd.read_csv('/home/dimitria/demo/notebookenv/path_parallel_2.csv')

# X1 = path.loc[:,'x'].tolist()
# Y1 = path.loc[:,'y'].tolist()
# Psi1 = path.loc[:,'psi'].tolist()
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
# for k in range(0, N):
x_d[0, :] = X
x_d[1, :] = Y#R * (1 - np.cos(OMEGA * t[k]))
x_d[2, :] = Psi+np.pi 
x_d[3, :] = 0#Phi#0
u_d[0, :] = -0.07#R * OMEGA
u_d[1, :] = 0#OMEGA

# %%
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
u = np.zeros((2, N))
phi = np.zeros((2, N)) #left and right

x[:, 0] = x_init

for k in range(1, N):

    # Simulate the differential drive vehicle motion
    x[:, k] = rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T)
    if x[3,k] < -0.6:
        x[3,k] = -0.6
    elif x[3,k] > 0.6:
        x[3,k] = 0.6

    # Compute the approximate linearization
    A = np.array(
        [
            [0, 0, -u_d[0, k - 1] * np.sin(x_d[2, k - 1]),0],
            [0, 0, u_d[0, k - 1] * np.cos(x_d[2, k - 1]),0],
            [0, 0, 0,u_d[0, k - 1]/(L*np.cos(x_d[3,k-1])**2) ],
            [0,0,0,0]
        ]
    )
    B = np.array([[np.cos(x_d[2, k - 1]), 0], [np.sin(x_d[2, k - 1]), 0]
                  ,[np.tan(x_d[3,k-1])/L, 0], [0, 1]])

    # Compute the gain matrix to place poles of (A-BK) at p
    p = np.array([-1.2, -2., -6, -5])
    K = signal.place_poles(A, B, p)
    if abs(u[1,k]*T) > 0.6:
        u[1,k] = np.sign(u[1,k])*0.6
    # Compute the controls (v, omega) and convert to wheel speeds (v_L, v_R)
    u[:,k] = -K.gain_matrix @ (x[:, k - 1] - x_d[:, k - 1]) + u_d[:, k - 1]
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

