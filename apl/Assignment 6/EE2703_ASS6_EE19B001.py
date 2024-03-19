"""
                             EE2703 Applied Programming Lab - 2021
                                 Assignment 6: Solution
                             Arun Krishna A M S - EE19B001
                                  12th April 2021
"""

import numpy as np
from pylab import *
from sys import argv, exit

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Default values initialization
n    = 100     #spatial grid size.
M    = 5       #number of electrons injected per turn.
nk   = 500     #number of turns to simulate.
u0   = 5       #threshold velocity.
p    = 0.25    #probability that ionization will occur
Msig = 2       #deviation of elctrons injected per turn

#Initializing input values
if len(sys.argv) > 7:
    print("Maximum 6 arguments can only be passed.")
    exit(0)

else:
    try:
        for i in range(len(sys.argv)):
            if i == 1:
                n = int(sys.argv[1])
            elif i == 2:
                M = int(sys.argv[2])
            elif i == 3:
                nk = int(sys.argv[3])        
            elif i == 4:
                u0 = int(sys.argv[4])
            elif i == 5:
                p = float(sys.argv[5])
            elif i == 6:
                Msig = float(sys.argv[6])
        if n < 0 or M < 0 or nk < 0 or u0 < 0 or p < 0 or p > 1 or Msig < 0:
            print("Parameters values invalid")
            exit(0) 
    except:
        print("Parameter Initialization invalid")

print("n    = ",n)
print("M    = ",M)
print("nk   = ",nk)
print("u0   = ",u0)
print("p    = ",p)
print("Msig = ",Msig)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


#initialize the position, speed and change in position arrays
xx = np.zeros(n*M)   #Electron position
u  = np.zeros(n*M)   #Electron velocity
dx = np.zeros(n*M)   #Displacement in current turn

#initialize the intensity, position and speed arrays
I = []    #Intensity of emitted light
X = []    #Electron position
V = []    #Electron velocity


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


for k in range(1,nk+1):

    #If an electron is in the chamber then 'xx' must be 0 < xx < L=n
    #Find all indices where electrons are present by getting the indices of positions greater than zero 
    ii = np.where(xx > 0)[0]   
    
    #Updating the postions, velocities and displacement change
    dx[ii] = u[ii] + 0.5
    xx[ii] += dx[ii]
    u[ii] += + 1

    #If the electrons have reached the anode, their position and velocities are reset to zero
    anodeReached = np.where(xx > n)[0]
    xx[anodeReached] = 0
    u[anodeReached] = 0
    dx[anodeReached] = 0

    #Finding electrons that would have collided 
    kk = np.where(u >= u0)[0]                         #Electrons with velocities greater than threshold 
    ll = np.where(np.random.rand(len(kk)) <= p)[0]    #Electrons collide with a probability '1-p'
    kl = kk[ll]                                       #Contains the indices of energetic electrons that collide
    
    #The collision takes place and the positions and velocities of electrons are affected. The collision can occur
    #at any time in (k − 1)∆t < t < k∆t and the collisions are uniformly distributed in time.
    
    #Collision happens at time 't'
    t = np.random.rand(len(kl)) 
    
    #If the collision happens at time 't', the distance covered till then would be Xprev + u*t + 1/2*a*t^2. After
    #the collision occurs, the electron comes to rest. After which it has (1-t) time for acceleration from 0 intial
    #velocity. This means that X = Xprev + u*t + 1/2*a*t^2 + 1/2*a*(1-t)^2. While the velocity is u = 0 + a*(1-t)
    xx[kl] = xx[kl] - dx[kl] + (u[kl]-1)*t + 0.5*t**2 + 0.5*(1-t)**2
    u[kl] = 0 + (1-t)
    
    #Inaccurate version:
    #xx[kl] = xx[kl] - dx[kl]*np.random.rand()
    #u[kl] = 0


    #The excited atoms at this location resulted in emission from that point. So we have to add a photon at that point. 
    I.extend(xx[kl].tolist())

    #New electrons to be injected
    m = int(randn()*Msig + M)
    
    #get empty spaces where electrons can be injected
    emptySlots = np.where(xx==0)[0]
        
    temp = min(m, len(emptySlots))
    xx[emptySlots[:temp]] = 1 #Injecting new electrons
    u[emptySlots[:temp]]  = 0 #with velocity zero
    dx[emptySlots[:temp]] = 0 #and displacement zero
    
    #Finding positions and speed of electrons and add to X and V
    temp = np.where(xx > 0)[0]
    X.extend(xx[temp].tolist())
    V.extend(u[temp].tolist())
  
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#histogram for light intensity
figure(0)
pops, bins, temp= hist(I, bins = np.arange(0,n+1,1), edgecolor='white', rwidth=1, color='black') #draw histogram
xpos = 0.5*(bins[0:-1] + bins[1:])
title("Light Intensity")
xlabel(r'Position$\rightarrow$')
ylabel(r'Intensity$\rightarrow$')
show()


#Tabulate results
print("Intensity data:")
print("xpos    count")
for i in range(len(pops)):
    print(str(bins[i]) + "     " + str(pops[i]))

#histogram for electron density
figure(1)
hist(X, bins=np.arange(0, n + 1, 1), edgecolor='white', rwidth=1, color='black')
title("Electron Density")
xlabel(r'Position$\rightarrow$')
ylabel(r'Number of Electrons$\rightarrow$')
show()


#phase space diagram
figure(2)
plot(X,V,'o', color='red')
title("Electron Phase Space")
xlabel(r'Position$\rightarrow$')
ylabel(r'Velocity$\rightarrow$')
show()


