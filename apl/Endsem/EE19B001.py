"""
                             EE2703 Applied Programming Lab - 2021
                     Endsemester Assignment - Magnetic Field of Loop Antenna
                             Arun Krishna A M S - EE19B001
                                     30th May 2021
"""

import numpy as np
from pylab import *
from sys import argv, exit
import mpl_toolkits.mplot3d.axes3d as p3
import scipy
import scipy.linalg as sp

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
These are the parameters that control the execution of the code. They have the following defaults, which 
can be corrected by the user via commandline arguments
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Nx = 3                  #                                        
Ny = 3

#Default Initial Values
Nz = 1000               #Length along z-axis for which variation in B is to be observed
N = 100                 #The loop is broken into 'N' parts. Increasing 'N' increases resolution
radius = 10             #Radius of the loop
k = 0.1                 #AC Current Wavenumber
#k = 0                 #DC Current

#Initializing input values
if len(sys.argv) > 5:
    print("Maximum 4 arguments can only be passed.")
    exit(0)

else:
    try:
        for i in range(len(sys.argv)):
            if i == 1:
                Nz = int(sys.argv[1])
            elif i == 2:
                N = int(sys.argv[2])
            elif i == 3:
                radius = float(sys.argv[3])        
            elif i == 4:
                k = float(sys.argv[4])
        if N <= 0 or k < 0 or radius <= 0 or Nz <= 0: 
            print("Parameters values invalid")
            exit(0) 
    except:
        print("Parameter Initialization invalid")

print("Parameters:", "\nNz = ",Nz, "\nN = ",N, "\nradius = ",radius, "\nk = ",k);


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
This part of the code is used to create a cartesian system and generate a current element wire loop of 
'radius' on the X-Y plane and plot the current elements
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Creating a cartesian system with Nx elements along X-axis, Ny elements along Y-axis and Nz-elements along
#Z axis through the 'meshgrid' function. X,Y,Z matrices of shape (Nx,Ny,Nz)

x = np.linspace(-1,1,Nx,dtype = float);
y = np.linspace(-1,1,Ny,dtype = float);
z = np.linspace(1,Nz,Nz,dtype = float);
Y,Z,X = np.meshgrid(y,z,x);

#Defining an array 'theta' of N elements with values 0, 2pi/N, 2pi*2/N, .... 2pi*(N-1)/N denoting the 
#angular/polar location of the current elements in the wire loop antenna

theta = np.linspace(0,2*np.pi,N+1,dtype = float)[:-1]

mag_dL = 2*np.pi*radius/N           #Length of each current element i.e., dl after the wire loop is broke into N pieces
I = np.cos(theta)*1e7               #Current 'I' in the current element. Variation: 'cos(theta)' spatially
#I = np.ones(len(theta))*1e7        #Current 'I' in the current element. Spatially Invariant


#The direction of each current element is tangential to the wire loop. This direction is given by:
dL = mag_dL* np.concatenate([[-np.sin(theta)], [np.cos(theta)]], axis = 0)

#The location of each current element on the X-Y plane is given by:
rL = radius*np.concatenate([[np.cos(theta)], [np.sin(theta)], [np.zeros(N)]], axis = 0)

IdL = np.multiply(I, dL)            #Current Element 'Idl'

#Plotting the current elements using the 'quiver' function in the desired direction.
figure(num=1,figsize=(7,6));
xlabel("X")
ylabel("Y")
title("Current Element")
quiver(rL[0,:], rL[1,:], IdL[0,:], IdL[1,:], label = "Current")
legend(loc = 'upper right');
show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Function to calculate Vector Potential due to each current element by calculating R = |r - r'| where 
#r is the point at which vector potential is to be found while r' correspond to the points on the ring. 
#Ax = Mu/4pi*e^(-jkR)/R*IdLx   
#Ay = Mu/4pi*e^(-jkR)/R*IdLy   

def calc(X,Y,Z, r, theta, IdL):
   R = ((X - r[0])**2 + (Y - r[1])**2 + (Z - r[2])**2)**(0.5)
   Ax = 1e-7*np.multiply(np.exp(-1j*R*k)/R, IdL[0])
   Ay = 1e-7*np.multiply(np.exp(-1j*R*k)/R, IdL[1])
   return Ax, Ay


#Declaration of variables for vector potentials Ax and Ay at each point in the volume 
Ax = np.zeros((Nz,Ny,Nx))   
Ay = np.zeros((Nz,Ny,Nx))   

#Calculating dAx and dAy due to each element by calling the function 'calc' and integrated/added/summed to find Ax and Ay
for i in range(theta.size):
   AxTemp, AyTemp = calc(X,Y,Z, rL[:,i], theta[i], IdL[:,i])
   Ax = Ax + AxTemp
   Ay = Ay + AyTemp 

#Calculating B on the z axis- by B = curl(A) = dAy/dx - dAx/dy
#Xmid and Ymid correspond to values of X and Y axis since 'x' and 'y' arrays span from on both negative and positive axes equally
delX = x[1] - x[0]
delY = y[1] - y[0]
Xmid = int(Nx/2)
Ymid = int(Ny/2)

#dAy/dx = (Ay[:,Ymid,Xmid+1] - Ay[:,Ymid,Xmid-1])/(2*delX)
#dAx/dy = (Ax[:,Ymid-1,Xmid] - Ax[:,Ymid+1,Xmid])/(2*delY)
B = (Ay[:,Ymid,Xmid+1] - Ay[:,Ymid,Xmid-1])/(2*delX) - (Ax[:,Ymid+1,Xmid] - Ax[:,Ymid-1,Xmid])/(2*delY) 
B = np.abs(B)

#Plotting Magnetic Field along Z-axis
figure(num=2);
grid();
xlabel(r'Z axis$\rightarrow$',fontsize=10);
ylabel(r'$|B(z)|$ $\rightarrow$',fontsize=10);
title(r'Magnitude of Magnetic field $|B(z)|$: loglog plot',size = 13);
loglog(z,B,label="Calculated");
show();

#Calculating the fit using least sum square method - by 'lstsq' function. The magnetic field is fitted 
#in the form B = a*z^b 
X = np.vstack([log(z), np.ones(len(z))]).T
M, N = scipy.linalg.lstsq(X, log(B))[0]
N = np.exp(N)

print("When the graph is fitted in the form of B = Nz^M, we find:")
print("N = ",N)
print("M = ",M)

#Plotting the magnetic field and the corresponding fit
figure(num=3);
grid();
xlabel(r'Z axis$\rightarrow$',fontsize=10);
ylabel(r'$|B(z)|$ $\rightarrow$',fontsize=10);
title(r'Fitting of Magnetic field $|B(z)|$ on to $Nz^M$ form: loglog plot',size = 13);
loglog(z,B,label="Calculated");
loglog(z,N*z**M, linewidth = 2, label="Fitting");
legend();
show();



