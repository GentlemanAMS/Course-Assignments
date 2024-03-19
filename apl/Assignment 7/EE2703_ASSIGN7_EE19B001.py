"""
                             EE2703 Applied Programming Lab - 2021
                                 Assignment 7: Solution
                             Arun Krishna A M S - EE19B001
                                  21th April 2021
"""

import numpy as np
from pylab import *
import scipy.signal as sp


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Question 1 and 2
Laplace transform of forced oscillations for f(t) = cos(1.5t)*exp(-at)*u(t) for values of a = 0.5, 0.05
correspond to F(s) = (s + a)/((s+a)^2 + 2.25). The forced oscillation is acted on a spring whose system is
modelled as X"(t) + 2.25*X(t) = f(t) 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def forcedOscillationQ1_2(a, figureNumber):
    
    #Time vector going from 0 to 50 seconds 
    t = np.linspace(0,50,1000)
    
    #Laplace domain expression for X(s) - derivation in report
    X = sp.lti([1, a],np.polymul([1,0,2.25], np.polyadd(np.polymul([1, a],[1, a]),[2.25])))

    #Time domain function values 
    t, x = sp.impulse(X, None, t)

    #Plotting functions
    figure(figureNumber)
    
    title("x(t) time domain")
    ylabel("$x(t)\\rightarrow$")
    xlabel("$t$ (in sec)$\\rightarrow$")
    plot(t, x, label = "Decay = " + str(a))
    grid(True)
    legend()
    show() 

forcedOscillationQ1_2(0.5,0)        #When decay coefficient a = 0.5
forcedOscillationQ1_2(0.05,1)       #When decay coefficient a = 0.05 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Question 3

Using signal.lsim to simulate the previous problem. In 'a' for loop, vary the frequency of the cosine in f(t) 
from 1.4 to 1.6 in steps of 0.05 keeping the decay coefficient a = 0.05 and plot the resulting responses. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Time vector going from 0 to 50 seconds 
t = np.linspace(0,50,1000)

# Transfer function of the spring system
H = sp.lti([1], [1, 0, 2.25])

#Time domain function values of H(s)
t, h = sp.impulse(H, None, t)


figure(2)
title("Output with varying frequency. decay coefficient = 0.05 ")
ylabel("$x(t)\\rightarrow$")
xlabel("$t$ (in sec)$\\rightarrow$")
grid(True)

#Iterating for various frequencies from 1.4 to 1.6 iterating for every 0.05
for f in np.linspace(1.4, 1.6, 5):

    #Function f(t) definition
    func = np.cos(f*t) * np.exp(-0.05*t)
    t, x, svec = sp.lsim(H, func, t)
    
    #Plotting x(t) in time domain for various frequencies
    plot(t, x, label = "Frequency = " + str(f))

legend()
show() 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Question 4
Equations:
ẍ + (x − y) = 0
ÿ + 2 (y − x) = 0
where the initial condition is x(0) = 1, ẋ(0) = y(0) = ẏ(0) = 0. Substitute for y from
the first equation into the second and get a fourth order equation. Solve for its time
evolution, and from it obtain x(t) and y(t) for 0 ≤ t ≤ 20.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Time vector going from 0 to 20 seconds 
t = np.linspace(0, 20, 1000)

# Transfer functions for x and y
X = sp.lti(np.polymul([1, 0], [0.5, 0, 1]), np.polyadd(np.polymul([1, 0, 1], [0.5, 0, 1]), [-1]))
Y = sp.lti([1, 0], np.polyadd(np.polymul([1, 0, 1], [0.5, 0, 1]), [-1]))

#Time domain function values of X(s) and Y(s)
t, x = sp.impulse(X, None, t)
t, y = sp.impulse(Y, None, t)

#Plotting functions
figure(3)
plot(t, x, label="x(t)")
plot(t, y, label="y(t)")
title("x(t) & y(t) in time domain")
ylabel("$Signal\\rightarrow$")
xlabel("$t$ (in sec)$\\rightarrow$")
legend()
grid(True)
show() 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Question 5
Obtain the magnitude and phase response of the Steady State Transfer function of the given two-port 
network.
H(s) = 1/(1 + RCs + LCs^2)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Transfer functions for H(s)
H = sp.lti([1], [1e-12, 1e-4, 1])

#Obtaining bode plot
w, S, phi = H.bode()

figure(num=4,figsize=(7,10));

#Plotting functions
plot1 = subplot(2, 1, 1)
plot1.semilogx(w, S)
plot1.set_title("Magnitude Plot")
plot1.set_ylabel("Magnitude(in dB)$\\rightarrow$")
plot1.set_xlabel("$\\omega\\rightarrow$")
grid(True)

plot2 = subplot(2, 1, 2)
plot2.semilogx(w, phi)
plot2.set_title("Phase Plot")
plot2.set_ylabel("Phase$\\rightarrow$")
plot2.set_xlabel("$\\omega\\rightarrow$")
grid(True)
show()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Question 6
Suppose the input signal Vin(t) is given by: Vin(t) = cos(1000t)u(t) − cos(1000000t)u(t)
Obtain the output voltage Vout(t) by defining the transfer function as a system and obtaining the output 
using signal.lsim.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Intitial short term response
t = np.linspace(0, 30e-6, 10000)
#Input signal
Vin = np.cos(1e3*t)-np.cos(1e6*t)
#Convolution for initial response
t, Vout, svec = sp.lsim(H, Vin, t)

#Plotting functions
figure(5)
plot(t, Vout)
title("$V_{out}$ vs $t$ (Initial response)")
xlabel("$t$ (in sec)$\\rightarrow$")
ylabel("$V_{out}\\rightarrow$")
grid(True)
show()


#Long term response
t = np.linspace(0, 1e-2, 10000)
#Input signal
Vin = np.cos(1e3*t)-np.cos(1e6*t)
#Convolution for Long term response
t, Vout, svec = sp.lsim(H, Vin, t)

figure(6)
plot(t, Vout)
title("$V_{out}$ vs $t$(Steady State response)")
xlabel("$t$ (in sec)$\\rightarrow$")
ylabel("$V_{out}\\rightarrow$")
grid(True)
show()



