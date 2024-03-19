"""
                             EE2703 Applied Programming Lab - 2021
                                 Assignment 4: Solution
                             Arun Krishna A M S - EE19B001
                                  10th March 2021
"""


import numpy as np
from pylab import *
import scipy
import scipy.integrate as SP 
from sys import exit
from numpy import pi,exp,cos,sin

#########################################################################################################
"""
Question 1
Define Python functions for the two functions cos(cos(x)) and exp(x), that take a vector (or scalar) input, 
and return a vector (or scalar) value. Defining a function that outputs the expected 2pi periodic fourier 
series is computed
"""

#Defining cos(cos(x)) function
def cosCos(x):
     return cos(cos(x));

#Defining exp(x) function
def exponential(x):
     return exp(x);
     
#func_dict is a dictionary which choses what function that has to be operated on based on its label
func_dict = {'exponential(x)':exponential,'cosCos(x)': cosCos}

#Defining function that outputs the expected 2pi periodic fourier series
def expectedFourier(label,x):
     function = func_dict[label];
     return function(np.remainder(x,2*pi));


##########################################################################################################
"""
Question 1
Plotting the functions over the interval [−2pi, 4pi) in Figure 1 and 2 respectively along with their expected
fourier series. 
"""
Xaxis = np.linspace(-2*pi,4*pi,6000);

#Figure 1 for exp(x) 

figure(num=1,figsize=(7,6));
grid(True);
#title(r'$e^x$ function - semilogy plot',size = 15);
xlabel(r'X$\rightarrow$',fontsize=15);	
ylabel(r'$e^X \rightarrow$',fontsize=11);

#exp(x) function
Y = exponential(Xaxis);
semilogy(Xaxis, Y, color = 'r', label = 'Exponential function');

#exp(x) function that outputs the expected 2pi periodic fourier series
Y = expectedFourier('exponential(x)',Xaxis);
semilogy(Xaxis, Y, color = 'b', label = 'Expected Fourier extension of periodic $e^X$');
legend();
show();



#Figure 2 for cos(cos(x) 

figure(num=2,figsize=(7,6));
grid(True);
#title(r'cos(cos(x)) function',size = 15);
xlabel(r'x$\rightarrow$',fontsize=13);	
ylabel(r'$cos(cos(x)) \rightarrow$',fontsize=13);

#cos(cos(x)) function
Y = cosCos(Xaxis);
semilogy(Xaxis, Y, color = 'r', label = 'cos(cos(x))');

#cos(cos(x) function that outputs the expected 2pi periodic fourier series
Y = expectedFourier('cosCos(x)',Xaxis);
semilogy(Xaxis, Y, color = 'b', label = 'Expected Fourier extension of periodic $cos(cos(x))$');
legend(loc = 'upper right');
show();

##########################################################################################################

"""
Question 2

Obtain the first 51 coefficients for the two functions above. Defining two functions u(x,k) = f(x)cos(kx) and 
v(x,k) = f(x)sin(kx) which is integrated by quad python inbuilt function from 0 to 2pi with extra arguments to
the functions being integrated.
"""

"""

Function u_func() is same as u(x,k) = f(x)cos(kx) which takes list 'argument' as input. From the 'argument', 'k'
and the function to be used is taken and u(x,k) is calculated
"""
def u_Func(x,argument):
     k = argument[0];
     label = argument[1];
     func = func_dict[label];
     return func(x)*cos(k*x);

"""
Function v_func() is same as v(x,k) = f(x)sin(kx) which takes list 'argument' as input. From the 'argument', 'k'
and the function to be used is taken and v(x,k) is calculated
"""
def v_Func(x,argument):
     k = argument[0];
     label = argument[1];
     func = func_dict[label];
     return func(x)*sin(k*x);


"""
functionCoefficient takes what function has to be used and the number of fourier calculations to be calculated. 
Using the option in quad extra arguments - 'argument = [i,label]' is passed to the u(x,k) and v(x,k) which are then
integrated:
"""
def functionCoefficient(label, numcoefficient):
     
     function = func_dict[label];
     cosCoefficient = np.zeros((numcoefficient+1)//2);
     sinCoefficient = np.zeros(numcoefficient//2);
     Coefficient = np.zeros(numcoefficient);
     
     #Calculating cosCoefficient[0] = a0
     cosCoefficient[0] = (SP.quad(function, 0, 2*pi)[0]) / (2*pi);
     Coefficient[0] = cosCoefficient[0];
     
     #cosCoefficient[k] = ak calculated by passing u(x,k) as the function to the 'quad' integrator
     for i in range(1,cosCoefficient.size,1):
          argument = [i,label];
          cosCoefficient[i] = (SP.quad(u_Func,0,2*pi,args = argument)[0])/pi;
          Coefficient[2*i-1] = cosCoefficient[i];

     #sinCoefficient[k] = bk calculated by passing v(x,k) as the function to the 'quad' integrator
     for i in range(1,sinCoefficient.size+1,1):
          argument = [i,label];
          sinCoefficient[i-1] = (SP.quad(v_Func,0,2*pi,args = argument)[0])/pi;
          Coefficient [2*i] = sinCoefficient[i-1];

     #Coefficient = [a0 a1 b1 a2 b2... ak bk...]
     #returning Coefficient,cosCoefficient,sinCoefficient  
     return Coefficient,cosCoefficient,sinCoefficient;

##############################################################################################################

"""
For each of the two functions, make two different plots using “semilogy” and “loglog” and plot the
magnitude of the coefficients vs n.
"""

#Number of fourier coefficients
numcoefficient = 51;



#Calculating fourier coefficients of exp(x)
CoefficientExpo, cosCoefficientExpo, sinCoefficientExpo = functionCoefficient('exponential(x)', numcoefficient);

#Plotting semilogy plot for exp(x)
figure(num=3,figsize=(7,6));
grid(True);
#title(r'Exponential function - Coefficients - semilog plot',size = 15);
xlabel(r'n$\rightarrow$',fontsize=13);	
ylabel(r'$coefficients \rightarrow$',fontsize=13);
semilogy(range(numcoefficient), np.abs(CoefficientExpo), 'ro');
show();

#Plotting loglog plot for exp(x)
figure(num=4,figsize=(7,6));
grid(True);
#title(r'Exponential function - Coefficients - loglog plot',size = 15);
xlabel(r'n$\rightarrow$',fontsize=15);	
ylabel(r'$coefficients \rightarrow$',fontsize=13);
loglog(range(numcoefficient), np.abs(CoefficientExpo), 'ro');
show();



#Calculating fourier coefficients of cos(cos(x))
CoefficientCosCos, cosCoefficientCosCos, sinCoefficientCosCos = functionCoefficient('cosCos(x)', numcoefficient);

#Plotting semilogy plot for cos(cos(x))
figure(num=5,figsize=(7,6));
grid(True);
#title(r'cos(cos(x)) function - Coefficients - semilog plot',size = 15);
xlabel(r'n$\rightarrow$',fontsize=15);	
ylabel(r'$coefficients \rightarrow$',fontsize=13);
semilogy(range(numcoefficient), np.abs(CoefficientCosCos), 'ro');
show();

#Plotting loglog plot for cos(cos(x))
figure(num=6,figsize=(7,6));
grid(True);
#title(r'cos(cos(x)) function - Coefficients - loglog plot',size = 15);
xlabel(r'n$\rightarrow$',fontsize=15);	
ylabel(r'$coefficients \rightarrow$',fontsize=13);
loglog(range(numcoefficient), np.abs(CoefficientCosCos), 'ro');
show();



#################################################################################################################

"""
The first 51 coefficients can also be approximately figured out by Linear Square Approach.
Define a vector X going from 0 to 2π in 400 steps using linspace.
"""
numData = 400;
X = np.linspace(0,2*pi,numData+1)[:-1];


#Matrix A contains the cos(kX[i]) and sin(kX[i]) terms
A = np.zeros((numData,numcoefficient));
A[:,0] = 1;
for i in range(1,(numcoefficient+1)//2):
     A[:,2*i-1] = cos(i*X);
     A[:,2*i] = sin(i*X);

"""
Using lstsq to solve this problem, we execute c=lstsq(A,b)[0] finds the “best fit” numbers that will satisfy AX = b
Obtain the coefficients for both the given functions for exp(x) and cosCos(x)
"""
#Evaluate the function f(x)=exp(x) at those X values and call it b.
b = exponential(X);
C_Expo = scipy.linalg.lstsq(A,b)[0];

#Evaluate the function f(x)=cos(cos(x)) at those X values and call it b.
b = cosCos(X);
C_CosCos = scipy.linalg.lstsq(A,b)[0];


#Plotting semilog plot for coefficients of exp(x)
figure(num=7,figsize=(7,6));
grid(True);
#title(r'Exponential function - Coefficients - semilog plot',size = 15);
xlabel(r'n$\rightarrow$',fontsize=13);	
ylabel(r'$coefficients \rightarrow$',fontsize=13);
semilogy(range(numcoefficient), np.abs(CoefficientExpo), 'ro',label = 'Truevalue');
semilogy(range(numcoefficient), np.abs(C_Expo), 'go',markersize=4,label = 'Least Square Value');
legend();
show();

#Plotting loglog plot for coefficients of exp(x)
figure(num=8,figsize=(7,6));
grid(True);
#title(r'Exponential function - Coefficients - loglog plot',size = 15);
xlabel(r'n$\rightarrow$',fontsize=13);	
ylabel(r'$coefficients \rightarrow$',fontsize=13);
loglog(range(numcoefficient), np.abs(C_Expo), 'go',label = 'Least Square Value');
loglog(range(numcoefficient), np.abs(CoefficientExpo), 'ro',markersize=4,label = 'Truevalue');
legend();
show();

#Plotting semilog plot for coefficients of cosCos(x)
figure(num=9,figsize=(7,6));
grid(True);
#title(r'cos(cos(x)) function - Coefficients - semilog plot',size = 15);
xlabel(r'n$\rightarrow$',fontsize=13);	
ylabel(r'$coefficients \rightarrow$',fontsize=13);
semilogy(range(numcoefficient), np.abs(CoefficientCosCos), 'ro',label = 'Truevalue');
semilogy(range(numcoefficient), np.abs(C_CosCos), 'go',markersize=4,label = 'Least Square Value');
legend();
show();

#Plotting loglog plot for coefficients of cosCos(x)
figure(num=10,figsize=(7,6));
grid(True);
#title(r'cos(cos(x)) function - Coefficients - loglog plot',size = 15);
xlabel(r'n$\rightarrow$',fontsize=13);	
ylabel(r'$coefficients \rightarrow$',fontsize=13);
loglog(range(numcoefficient), np.abs(C_CosCos), 'go',label = 'Least Square Value');
loglog(range(numcoefficient), np.abs(CoefficientCosCos), 'ro',markersize=4,label = 'Truevalue');
legend();
show();

################################################################################################################

"""
The absolute difference between the two sets of coefficients - from analysis equation and least sum squares is calculated
from which the largest deviation is calculated.
"""
errorMax_Exp = np.amax(np.abs(CoefficientExpo - C_Expo))
errorMax_CosCos = np.amax(np.abs(CoefficientCosCos - C_CosCos))
print("Max deviation in estimation of exp(x) is", errorMax_Exp);
print("Max deviation in estimation of cos(cos(x)) is", errorMax_CosCos);

################################################################################################################
"""
The first 51 coefficients can also be approximately figured out by Linear Square Approach.
Define a vector X going from -2pi to 4pi in 400 steps using linspace.
"""
numData = 400;
Xaxis = np.linspace(-2*pi,4*pi,numData);

#Matrix A contains the cos(kX[i]) and sin(kX[i]) terms
A = np.zeros((numData,numcoefficient));
A[:,0] = 1;
for i in range(1,(numcoefficient+1)//2):
     A[:,2*i-1] = cos(i*Xaxis);
     A[:,2*i] = sin(i*Xaxis);


#Plotting exp(x)
figure(num=11,figsize=(7,6));
grid(True);
#title(r'Exponential function - semilogy plot',size = 15);
xlabel(r'X$\rightarrow$',fontsize=13);	
ylabel(r'$e^X \rightarrow$',fontsize=13);

#Plotting true values of exp(x)
Y = exponential(Xaxis);
semilogy(Xaxis, Y, color = 'r', label = 'Exponential function - true value');

#Plotting fourier series of exp(x) with fourier coefficients obtained from Least Square method 
Y = np.matmul(A,C_Expo);
semilogy(Xaxis, Y, 'go', label = 'Exponential function - Least Square method');

#Plotting fourier series of exp(x) with fourier coefficients obtained from Analysis equation
Y = np.matmul(A,CoefficientExpo);
semilogy(Xaxis, Y, color = 'b',markersize=4, label = 'Exponential function - Coefficient Calculation');
legend();
show();



#Plotting cos(cos(x))
figure(num=12,figsize=(7,6));
grid(True);
#title(r'cos(cos(x)) function - plot',size = 15);
xlabel(r'X$\rightarrow$',fontsize=13);	
ylabel(r'$cos(cos(x)) \rightarrow$',fontsize=13);

#Plotting true values of cos(cos(x))
Y = cosCos(Xaxis);
semilogy(Xaxis, Y, color = 'r', label = 'Exponential function - true value');

#Plotting fourier series of cos(cos(x)) with fourier coefficients obtained from Least Square method 
Y = np.matmul(A,C_CosCos);
semilogy(Xaxis, Y, 'go', label = 'Exponential function - Least Square method');

#Plotting fourier series of cos(cos(x)) with fourier coefficients obtained from Analysis equation
Y = np.matmul(A,CoefficientCosCos);
semilogy(Xaxis, Y, color = 'b',markersize=4, label = 'Exponential function - Coefficient Calculation');
legend(loc = 'upper right');
show();

################################################################################################################






  
