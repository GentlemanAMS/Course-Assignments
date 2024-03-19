"""
                             EE2703 Applied Programming Lab - 2021
                                 Assignment 5: Solution
                             Arun Krishna A M S - EE19B001
                                  25th March 2021
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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

#Default arguments
Nx = 25                 
Ny = 25
radius = 8
Niter = 1500

if(len(argv) == 5):
    Nx, Ny, radius, Niter = [int(i) for i in sys.argv[1:]]
elif(len(argv) == 1):
    pass
else:
    print("Arguments mismatch. Maximum 4 arguments can only be passed.")
    exit(0);

print("Parameters:", "\nNx = ",Nx, "\nNy = ",Ny, "\nradius = ",radius, "\nNiter = ",Niter);

#The wire must lie within the plate i.e., radius must be less than Nx/2 and Ny<2
if radius > Nx/2 or radius > Ny/2:
    print("The radius can't be greater than the coordinate values");
    exit(0);


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Initializing array 'phi' to zero. The array should have Ny rows and Nx columns. Two arrays, one with x-coordinates
and the other with y-coordinates with the same shape as the array. This is done with 'meshgrid' function.

                       Y,X=meshgrid(y,x)

where x and y are vectors containing the x and y positions. The middle of the region should be set with x = 0 and 
y = 0. The region within a radius of 8 of the centre should have potential of 1V. Function “where” is used to 
locate the points within this region. The coordinates are stored in an index array “ii”.
 
                       ii = np.where(np.multiply(X,X) + np.multiply(Y,Y) < radius^2);
                       phi[ii] = 1.0;
                       
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

x = np.linspace(0.5,-0.5,Ny,dtype = float);
y = np.linspace(0.5,-0.5,Nx,dtype = float);
Y,X = np.meshgrid(y,x);

#The radius input is given in input as an index not as a coordinate. The following line is used to find 'radius'
#in coordinate form. Generally radius = radius/Nx or radius = radius/Ny. But if Nx and Ny are different, we do: 
radius = (1/Nx + 1/Ny)*(1/2)*radius;

ii = np.where(np.multiply(X,X) + np.multiply(Y,Y) < radius**2);
phi = np.zeros([Ny,Nx], dtype = float);
phi[ii] = 1.0;

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plot a contour plot of the potential in Figure 1. Mark the V = 1 region by marking those nodes red.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

figure(num=1,figsize=(7,6));
grid(linestyle=':');
xlabel(r'X$\rightarrow$',fontsize=13);
ylabel(r'Y$\rightarrow$',fontsize=13);
title(r'Contour plot: Initial Potential ',size = 18);

contourf(X,Y,phi, colors = ['#481b6d', 'blue', 'green', 'yellow', 'orange', 'red', 'white']);
plot(x[ii[0]], y[ii[1]], 'ro')
colorbar(shrink=0.8)

show();

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Updating potential:

Iterate Niter times and record the errors. The structure of your for loop should be as follows:
for k in range(Niter)
    save copy of phi
    update phi array
    assert boundaries
    errors[k]=(abs(phi-oldphi)).max();
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Initializing 'error' array with zeros
errors = np.zeros(Niter);

for k in range(Niter):

    #Copying phi to oldphi to hold the values of the previous iteration.
    oldphi = phi.copy()
    
    # The potential at any point should be the average of its neighbours
    # φ[i,j] = (φ[i+1,j] + φ[i−1,j] + φ[i,j+1] + φ[i,j−1])/4
    phi[1:-1,1:-1]=0.25*(oldphi[1:-1,0:-2]+ oldphi[1:-1,2:]+ oldphi[0:-2,1:-1] + oldphi[2:,1:-1]);


    # At boundaries where the electrode is present, the value of electrode potential = 1V = electric potential. 
    # At boundaries where there is no electrode, the gradient of φ should be tangential. This is implemented by 
    # requiring that φ should not vary in the normal direction. # On the left, right and top margin, φ should not 
    # vary normally. Since the bottom margin is connected to origin, the electric potential = 0.
    phi[:, 0] = phi[:, 1];            #Left margin              
    phi[:, Nx-1] = phi[:, Nx-2];      #Right margin
    phi[0, :] = phi[1, :];            #Top margin
    phi[Ny-1, :] = 0;                 #Bottom margin
    phi[ii] = 1.0;                    #Potential = electrode potential = 1V

    # Error is calculated. 
    errors[k]=np.max(np.abs(phi-oldphi))



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
To show how the errors are evolving, we plot a semi-log-y and log-log plot. Every 50th point is also plotted, 
To see  individual data points.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

iterations = np.array(range(Niter)) + 1

figure(num=2,figsize=(7,6));
xlabel(r'Number of iterations$\rightarrow$',fontsize=13);
ylabel(r'Error$\rightarrow$',fontsize=13);
title(r'Error vs iterations: semilogy plot',size = 18);
semilogy(iterations,errors);
semilogy(iterations[::50],errors[::50],'ro', markersize=4, label="Every 50th iteration");
legend(loc = 'upper right');
show();

figure(num=3,figsize=(7,6));
xlabel(r'Number of iterations$\rightarrow$',fontsize=13);
ylabel(r'Error$\rightarrow$',fontsize=13);
title(r'Error vs iterations: loglog plot',size = 18);
loglog(iterations,errors);
loglog(iterations[::50],errors[::50],'ro', markersize=4, label="Every 50th iteration");
legend(loc = 'upper right');
show();


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Extracting the exponent, since the error seems to be exponentially decreasing. But, note that it is an 
exponential decay only for larger iteration numbers.

Fitting the entire vector and only the portion beyond 500. 
The fit is of form 
                   y = A*exp(Bx)
                   log y = log A + Bx

Extract the above fit for the entire vector of errors and for those error entries after the 500 th iteration.
Plot the fit in both cases on the error plot itself. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Fitting_Function(X,Y):
    logY = np.transpose(log(Y));
    X = np.c_[np.ones([np.size(X),1]),X];
    return scipy.linalg.lstsq(X,logY)[0];   

A1, B1 = Fitting_Function(iterations, errors);
A1 = exp(A1);
A2, B2 = Fitting_Function(iterations[500:], errors[500:]);
A2 = exp(A2);
print("A1=",A1);
print("B1=",B1);
print("A2=",A2);
print("B2=",B2);

figure(num=4,figsize=(7,6));
xlabel(r'Number of iterations$\rightarrow$',fontsize=13);
ylabel(r'Error$\rightarrow$',fontsize=13);
title(r'Fitting: semilogy plot',size = 18);
semilogy(iterations,errors, label="Errors");
semilogy(iterations,A1*exp(B1*iterations), linewidth = 2, label="Fit1");
semilogy(iterations,A2*exp(B2*iterations), linewidth = 4, linestyle = '-.', label="Fit2");
legend(loc = 'upper right');
show();


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Cumulative error: -A/B*exp(B*(N+0.5))
Plotting cumulative error against number of iterations in both semilog and loglog plot
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

iterations = np.linspace(1,2000,400);
error_eachIteration = -A2/B2*exp(B2*(iterations+0.5));

figure(num=5,figsize=(7,6));
xlabel(r'Number of iterations$\rightarrow$',fontsize=13);
ylabel(r'Cumulative Error$\rightarrow$',fontsize=13);
title(r'Cumulative Error: semilogy plot',size = 18);
semilogy(iterations,error_eachIteration, 'ro', markersize = 2);
show();

figure(num=6,figsize=(7,6));
xlabel(r'Number of iterations$\rightarrow$',fontsize=13);
ylabel(r'Cumulative Error$\rightarrow$',fontsize=13);
title(r'Cumulative Error: loglog plot',size = 18);
loglog(iterations,error_eachIteration, 'ro', markersize = 2);
show();


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plotting 3d surface plot of final potential
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
fig7 = figure(num=7,figsize=(7,6));
ax = p3.Axes3D(fig7);
title("3-D surface plot: Potential")
xlabel("Y")
ylabel("X")
surf = ax.plot_surface(Y, X, phi, rstride=1, cstride=1, cmap=cm.jet)
fig7.colorbar(surf, shrink=0.5, aspect=5)
show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plotting 2d contour plot of final potential
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
levels = np.linspace(0, 1, 200)
figure(num=8,figsize=(7,6));
xlabel("Y")
ylabel("X")
title("2-D contour plot: Potential")
plot(x[ii[0]], y[ii[1]], 'ro')
contourf(Y,X,phi,levels, cmap=plt.cm.get_cmap("hot"));
colorbar(shrink=0.8)
show()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plotting vector potential of current
To obtain the shape of current profile, we compute the gradient. 

                 Jx[i,j] = 1/2*(phi[i,j-1]-phi[i,j+1])
                 Jy[i,j] = 1/2*(phi[i-1,j]-phi[i+1,j])

The vector plot is cerated using the 'quiver' function in the desired direction.

               quiver(y,x,Jy,Jx)

Plot the current density using quiver, and mark the electrode via red dots. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Jx = 1/2*(phi[1:-1,:-2]-phi[1:-1,2:])
Jy = 1/2*(phi[:-2,1:-1]-phi[2:,1:-1])

figure(num=9,figsize=(7,6));
xlabel("X")
ylabel("Y")
title("Current Density")
quiver(Y[1:-1,1:-1],X[1:-1,1:-1],-Jx,-Jy, label = "Current")
plot(x[ii[0]], y[ii[1]],'ro', label = "Electrode")
legend(loc = 'upper right');
show()



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Determining the temperature at each point and plotting 2d contour of final temperature
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

temperature = 300*np.ones([Ny,Nx], dtype = float);
levels = np.linspace(300, 300.12, 1000)

for k in range(Niter):

    #Copying phi to oldphi to hold the values of the previous iteration.
    oldtemperature = temperature.copy() 
    
    #Calculating the effect of current. Sigma = 0.0001
    var = (Jx**2 + Jy**2);
    
    # The potential at any point should be the average of its neighbours
    # T[i,j] = (T[i+1,j] + T[i−1,j] + T[i,j+1] + T[i,j−1] + 1/σ*|J|^2)/4    
    temperature[1:-1,1:-1]=0.25*(oldtemperature[1:-1,0:-2]+ oldtemperature[1:-1,2:]+ oldtemperature[0:-2,1:-1] + oldtemperature[2:,1:-1] + var);

    # At boundaries where the electrode is present, the value of temperature = 300K. 
    # At boundaries where there is no electrode, the gradient of temperature should be tangential. This is implemented by 
    # requiring that T should not vary in the normal direction. # On the left, right and top margin, T should not 
    # vary normally. Since the bottom margin is connected to ground, the temperature = 300K.
    temperature[:, 0] = temperature[:, 1];            #Left margin              
    temperature[:, Nx-1] = temperature[:, Nx-2];      #Right margin
    temperature[0, :] = temperature[1, :];            #Top margin
    temperature[Ny-1, :] = 300;                       #Bottom margin
    temperature[ii] = 300;                    


figure(num=10,figsize=(7,6));
grid(linestyle=':');
xlabel(r'X$\rightarrow$',fontsize=13);
ylabel(r'Y$\rightarrow$',fontsize=13);
title(r'Contour plot: temperature ',size = 18);
contourf(Y,X,temperature,levels, cmap=plt.cm.get_cmap("hot"));
colorbar(shrink=0.8)
show();


