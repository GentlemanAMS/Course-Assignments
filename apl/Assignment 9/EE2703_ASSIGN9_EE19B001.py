"""
                             EE2703 Applied Programming Lab - 2021
                                 Assignment 9: Solution
                             Arun Krishna A M S - EE19B001
                                     21st May 2021
"""

import numpy as np
from pylab import *

# function to generate the fourier terms from
# - time domain function: function
# - time interval [xmin, xmax)
# - number of samples per cycle: N
def plotFT(function, xmin, xmax, N, titleString, figureNumber, xlimit):

    # initialize the points where we will sample the function.
    x = np.linspace(xmin, xmax, N + 1)[:-1]
    w = np.linspace(-np.pi*N/(xmax-xmin), np.pi*N/(xmax-xmin), N + 1)[:-1]
    
    # sample the function
    y = function(x)

    # calculate and normalize the frequency spectrum
    Y = np.fft.fftshift(np.fft.fft(y))/N    
    
    # Obtain the phase of the frequency spectrum
    phase = np.angle(Y)

    figure(figureNumber, figsize=(6.5,3.5))
    plot(w, abs(Y))
    title(titleString, fontsize=10)
    xlim(xlimit)
    xlabel("$\\omega$")
    ylabel("$|Y|$")    

    figure(figureNumber+1, figsize=(6.5,3.5))
    bolden = where(abs(Y) > 1e-2)
    scatter(w, phase, marker = 'o', color = 'cyan')    
    scatter(w[bolden], phase[bolden], marker = 'o', color = 'blue') 
    grid(True)
    xlim(xlimit)
    title(titleString, fontsize=10)
    xlabel("$\\omega$")
    ylabel("Phase $Y$")    
    show()
    

def sin5(x):
    return np.sin(5*x)
plotFT(sin5, 0, 2*np.pi, 256, "sin(5t) Frequency Spectrum", 1, [-10, 10])

def amplitudeModulation(x):
    return (1 + 0.1*np.cos(x))*np.cos(10*x)
plotFT(amplitudeModulation, -4*np.pi, 4*np.pi, 4028, "$(1+0.1cos(t))cos(10t)$ Frequency Spectrum", 3, [-15, 15])

def sinCube(x):
    return (np.sin(x))**3
plotFT(sinCube, -4*np.pi, 4*np.pi, 2048, "$sin^3(x)$ Frequency Spectrum", 5, [-5, 5])

def cosCube(x):
    return (np.cos(x))**3
plotFT(cosCube, -4*np.pi, 4*np.pi, 2048, "$cos^3(x)$ Frequency Spectrum", 7, [-5, 5])

def phaseModulation(x):
    return np.cos(20*x+5*np.cos(x))
plotFT(phaseModulation, -4*np.pi, 4*np.pi, 2048, "$cos(20t+5cos(t))$ Frequency Spectrum", 9, [-40, 40])

def GaussianFFT(xmin, xmax):

    # - number of samples per cycle: N
    N = 2048

    # initialize the points where we will sample the function.
    x = np.linspace(xmin, xmax, N + 1)[:-1]
    w = np.linspace(-np.pi*N/(xmax-xmin), np.pi*N/(xmax-xmin), N + 1)[:-1]

    # sample the function
    y = np.exp(-x*x/2)

    # calculate and normalize the frequency spectrum
    Y = np.fft.fftshift(np.fft.fft(y))/N
    Y = Y * np.sqrt(2*np.pi)/max(Y)

    # Obtain the phase of the frequency spectrum
    phase = np.angle(Y)
    
    # Exact DFT of the Gaussian Distribution
    correctY = np.exp(-w**2/2)*np.sqrt(2*np.pi)
    
    #Maximum error between the evaluated and the precise DFT of Gaussian Distribution
    errorMax = max(abs(correctY - abs(Y)))
    
    return w,Y,phase,errorMax

error = []
# vary the time interval as -i*pi to i*pi as i goes from 1 to 7, and print the maximum error each time
# with respect to the actual calculated answer
for i in range(1, 7):
    w, Y, phase, errorMax = GaussianFFT(-i*np.pi, i*np.pi)
    error.append(errorMax)
    print("time interval: [-"+ str(i)+"pi, " + str(i)+"pi)"+ " Error = " + str(errorMax))

figure(11, figsize=(6.5,3.5))
semilogy(range(1,7),error, linewidth = 2);
title("Errors associated with time intervals", fontsize=10)
xlabel("$\\omega$")
ylabel("$|Y|$")    

figure(12, figsize=(6.5,3.5))
plot(w, abs(Y))
title("$exp(-\\frac{t^2}{2})$ Frequency Spectrum", fontsize=10)
xlim([-10, 10])
xlabel("$\\omega$")
ylabel("$|Y|$")    

figure(13, figsize=(6.5,3.5))
bolden = where(abs(Y) > 1e-2)
scatter(w, phase, marker = 'o', color = 'cyan')    
scatter(w[bolden], phase[bolden], marker = 'o', color = 'blue') 
grid(True)
xlim([-10, 10])
title("$exp(-\\frac{t^2}{2})$ Frequency Spectrum", fontsize=10)
xlabel("$\\omega$")
ylabel("Phase $Y$")    
show()
