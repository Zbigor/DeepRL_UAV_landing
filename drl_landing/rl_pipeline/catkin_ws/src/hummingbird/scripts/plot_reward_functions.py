# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np



def approach_angle_reward(roll,pitch):
    if np.abs(roll) + np.abs(pitch) < 0.174:
        return 100*np.exp((7.0*(0.174-np.abs(roll) - np.abs(pitch)))**1)
    if (np.abs(roll) + np.abs(pitch)<=1.55)and(np.abs(roll) + np.abs(pitch) >=0.174):
        return -6.0*(np.exp((3.2*(np.abs(roll) + np.abs(pitch)-0.174))**1))
    if (np.abs(roll) + np.abs(pitch)>1.55):
        return -500.0
    
    
def flip_reward(angle,prev_angle):
    if np.abs(angle) < 0.26:
        return 0.05*np.exp(20*(0.26-np.abs(angle)))
    if (np.abs(angle)>=0.26):
        return -7.0*np.exp((2.1*(np.abs(angle)-0.26))**1)

    
    
def approach_velocity_reward(velocity):
    if velocity>1.6:
        return -20.0*np.exp((0.45*(np.abs(velocity)))**1)
    if (velocity<=1.6) and (velocity >=0.1):
        return - 12.5 * np.exp(2.1*(velocity-0.1))
    if velocity < 0.1:
        return +55.0 * np.exp(20*(0.1-velocity))    
 
    
# approach angle    
#roll_space = np.linspace(-1.57,1.57,300)
#pitch_space = np.linspace(-1.57,1.57,300)


#X,Y = np.meshgrid(roll_space,pitch_space)
#
#Z = np.zeros(shape = (len(roll_space),len(pitch_space)))

#for it_r,r in enumerate(roll_space):
#    for it_p,p in enumerate(pitch_space):
#        Z[it_r,it_p] = approach_angle_reward(r,p)


# calculate angle_space for flipping
#angle_space = np.linspace(-3.14,3.14,500)
#dummy_space = np.linspace(-3.14,3.14,500)
#
#
#X,Y = np.meshgrid(angle_space,dummy_space)
#Z = np.zeros(shape = (len(angle_space),len(dummy_space)))
#
#for it_a1,a1 in enumerate(angle_space):
#    for it_a2,a2 in enumerate(dummy_space):
#        Z[it_a1,it_a2] = flip_reward(a1,a2)


# approach velocity

vel_space = np.linspace(0.0,10,500)
dummy_space = np.linspace(0.0,10,500)        

X,Y = np.meshgrid(vel_space,dummy_space)
Z = np.zeros(shape = (len(vel_space),len(dummy_space)))

for it_a1,a1 in enumerate(vel_space):
    for it_a2,a2 in enumerate(dummy_space):
        Z[it_a1,it_a2] = approach_velocity_reward(a1)





fig, ax = plt.subplots(figsize=(7, 7), dpi=100)


# for positive values
p = ax.pcolor(X, Y, Z, cmap=plt.cm.RdBu, vmin=(Z).min(), vmax=(Z).max())

#p = ax.pcolor(X, Y, Z, cmap=plt.cm.RdBu, vmin=Z.min(), vmax=Z.max())

cb = fig.colorbar(p)  

#cnt = plt.contour(Z, cmap=plt.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0, 1, 0, 1])  

