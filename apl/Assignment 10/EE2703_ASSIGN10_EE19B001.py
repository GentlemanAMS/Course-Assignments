"""
                             EE2703 Applied Programming Lab - 2021
                                 Assignment 10: Solution
                             Arun Krishna A M S - EE19B001
                                     2nd June 2021
"""

from pylab import *
import random

# Function to obtain the function's DFT using FFT and plot the frequency spectrum
# function to generate the fourier terms from
# - time domain function: function
# - time interval [xmin, xmax)
# - number of samples per cycle: N
def FastFourierTransform(function, plotTitle, interval, N, xlimit, boolOdd, booltemp):

    # initialize the points where we will sample the function.
    time = linspace(interval[0], interval[1], N + 1)[:-1]
    w = linspace(-pi*N/(interval[1]-interval[0]), pi*N/(interval[1]-interval[0]), N + 1)[:-1]
    
    # sample the function
    y = function(time)

    # For odd signals, y[0] = 0 to reduce the redundant phase errors
    if boolOdd == True:
        y[0] = 0

    # calculate and normalize the frequency spectrum
    y = fftshift(y) 
    Y = fftshift(fft(y))/N
    phase = angle(Y)
    
    # Plotting phase and magnitude plots for DFT of the signals
    figure(figsize=(6.5,3.5))
    plot(w, abs(Y), lw = 2)
    xlim([-xlimit, xlimit])
    title("Frequency Spectrum of " + plotTitle, fontsize=10)
    xlabel("$\\omega$")
    ylabel(r"$|Y|$")
    grid(True)
    show()

    figure(figsize=(6.5,3.5))
    bolden = where(abs(Y) > 1e-2)
    scatter(w, phase, marker = 'o', color = 'cyan')    
    scatter(w[bolden], phase[bolden], marker = 'o', color = 'blue') 
    xlim([-xlimit, xlimit])
    title("Frequency Spectrum of " + plotTitle, fontsize=10)
    xlabel("$\\omega$")
    ylabel("Phase $Y$")    
    grid(True)
    show()

    if booltemp:
        figure(figsize=(6.5,3.5))
        title("Frequency Spectrum of " + plotTitle, fontsize=10)
        semilogx(w, 20*log10(abs(Y)), lw = 2)
        xlim([1, 10])
        ylim([-20, 0])
        xlabel("$\\omega$")
        ylabel("Phase $Y$")    
        grid(True)
        show()

    return w, abs(Y), phase





# Function to return Hamming Window sequence
def hammingFunction(n, a = 0.54,b = 0.46):
    return fftshift(a + b*cos((2*pi*n)/(len(n) - 1)))


# Question 1

def sinroot2(x):
    return sin(sqrt(2) * x)

# Passing function and corresponding parameters for DFT Analysis
FastFourierTransform(function = sinroot2, plotTitle = "$sin(\\sqrt{2}t)$", interval = [-pi, pi], N = 64, xlimit = 10, boolOdd = True, booltemp = False);


# Plotting the given signal and its periodic approximation
figure(figsize=(6.5,3.5))
x1 = linspace(-10, 10, 200)
x2 = linspace(-pi, pi, 100)
y1 = sinroot2(x1)
y2 = sinroot2(x2)
plot(x1, y1, label = 'Original signal', color = 'cyan')
plot(x2, y2, label = 'Sampled signal', color = 'blue')
xlabel("t")
ylabel("$sin(\\sqrt{2}t)$")
title("Sampled interval of $sin(\\sqrt{2}t)$", fontsize=10)
grid(True)
legend()
show()

figure(figsize=(6.5,3.5))
x3 = linspace(pi, 3*pi, 100)
x4 = linspace(-3*pi, -pi, 100)
plot(x2, y2, color = 'blue')
plot(x3, y2, color = 'blue')
plot(x4, y2, color = 'blue')
xlabel("t")
ylabel("$sin(\\sqrt{2}t)$")
title("Periodic function approximation of $sin(\\sqrt{2}t)$", fontsize=10)
grid(True)
show()


# definition of ramp function
def ramp(x):
    return x

## Passing function and corresponding parameters for DFT Analysis
FastFourierTransform(function = ramp, plotTitle = "ramp function", interval = [-pi, pi], N = 64, xlimit = 10, boolOdd = True, booltemp = True);

def hammingsinroot2(x):
    return sinroot2(x)*hammingFunction(arange(len(x)))    

# Plotting the periodic approximation of the signal after Hamming   
x2 = linspace(-pi, pi, 65)[:-1]
x3 = linspace(pi, 3 * pi, 65)[:-1]
x4 = linspace(-3 * pi, -pi, 65)[:-1]
y = hammingsinroot2(x2)

figure()
plot(x2, y, color = 'blue')
plot(x3, y, color = 'blue')
plot(x4, y, color = 'blue')
xlabel("$t$ (in s)")
ylabel("$sin(\\sqrt{2}t)$")
title("$sin(\\sqrt{2}t)$ with Hamming Window applied")
grid(True)
show()

# Passing function and corresponding parameters for DFT Analysis
FastFourierTransform(function = hammingsinroot2, plotTitle = "$sin(\\sqrt{2}t)$ with Hamming", interval = [-pi, pi], N = 64, xlimit = 8, boolOdd = True, booltemp = False);
FastFourierTransform(function = hammingsinroot2, plotTitle = "$sin(\\sqrt{2}t)$ with Hamming @Higher Resolution", interval = [-4*pi, 4*pi], N = 256, xlimit = 4, boolOdd = True, booltemp = False);





# Question 2

def coscube(x):
    return (cos(0.86 *x))**3

def hammingcoscube(x):
    return coscube(x)*hammingFunction(arange(len(x)))

## Passing function definitions for DFT Analysis
FastFourierTransform(function = coscube, plotTitle = "$cos^{3}(0.86t)$ without Hamming", interval = [-4*pi, 4*pi], N = 512, xlimit = 8, boolOdd = False, booltemp = False)
FastFourierTransform(function = hammingcoscube, plotTitle = "$cos^{3}(0.86t)$ with Hamming", interval = [-4*pi, 4*pi], N = 512, xlimit = 8, boolOdd = False, booltemp = False)





# Question 3

#Generation of arbitrary frequency and phase using random function
freq = random.uniform(0.5,1.5)
delta = random.uniform(-pi,pi)
print("frequency = ",freq)
print("phase = ",delta)


def cosfunction(x):
    return cos(freq*x + delta)

def hammingcosfunction(x):
    return cosfunction(x)*hammingFunction(arange(len(x)))

## Function to estimate w0 and delta by calculating the weighted mean (by magnitude)
def estimatingWo_and_delta(w,mag,phase):
    #Only frequencies which has significant contribution are taken into consideration
    significantmag = where(mag > 0.1)
    w = w[significantmag]
    mag = mag[significantmag]
    phase = phase[significantmag]
    estimatedWo = sum((mag**2)*abs(w))/sum(mag**2)
    estimatedDelta = sum((mag**2)*abs(phase))/sum(mag**2)    
    print("Estimated frequency:", estimatedWo)
    print("Estimated phase:", estimatedDelta)

## Passing function definitions for DFT Analysis
w, mag, phase = FastFourierTransform(function = cosfunction, plotTitle = "$cos(\omega t+ \phi)$ without Hamming", interval = [-4*pi, 4*pi], N = 512, xlimit = 4, boolOdd = False, booltemp = False)
print("\nEstimations for cos(freq*t + phase) without hamming:")
estimatingWo_and_delta(w,mag,phase)

## Passing function definitions for DFT Analysis
w, mag, phase = FastFourierTransform(function = hammingcosfunction, plotTitle = "$cos(\omega t+ \phi)$ with Hamming", interval = [-4*pi, 4*pi], N = 512, xlimit = 4, boolOdd = False, booltemp = False)
print("\nEstimations for cos(freq*t + phase) with hamming:")
estimatingWo_and_delta(w,mag,phase)







# Question 4

#Function with white gaussian noise
def noisycos(x):
    return cos(freq*x + delta) + 0.1 * randn(len(x))


def hammingnoisycos(x):
    return noisycos(x)*hammingFunction(arange(len(x)))

## Passing function definitions for DFT Analysis    
w, mag, phase = FastFourierTransform(function = noisycos, plotTitle = "$cos(\omega t+ \phi)$(with noise) without Hamming", interval = [-4*pi, 4*pi], N = 512, xlimit = 4, boolOdd = False, booltemp = False)
print("\nEstimations for cos(freq*t + phase) with noise without hamming:")
estimatingWo_and_delta(w,mag,phase)

## Passing function definitions for DFT Analysis
w, mag, phase = FastFourierTransform(function = hammingnoisycos, plotTitle = "$cos(\omega t+ \phi)$(with noise) with Hamming", interval = [-4*pi, 4*pi], N = 512, xlimit = 4, boolOdd = False, booltemp = False)
print("\nEstimations for cos(freq*t + phase) with noise with hamming:")
estimatingWo_and_delta(w,mag,phase)




# Question 5

# Function definition for chirp function
def chirp(x):
    return cos(16*(1.5 + x/(2*pi))*x)

def hammingchirp(x):
    return chirp(x)*hammingFunction(arange(len(x)))

# Passing function definitions for DFT Analysis
FastFourierTransform(function = chirp, plotTitle = "$cos(16(1.5+ x/(2pi))x)$ without Hamming", interval = [-pi, pi], N = 1024, xlimit = 100, boolOdd = False, booltemp = False)
FastFourierTransform(function = hammingchirp, plotTitle = "$cos(16(1.5+ x/(2pi))x)$ with Hamming", interval = [-pi, pi], N = 1024, xlimit = 100, boolOdd = False, booltemp = False)






# Question 6

# Defining time and frequency variables
t = linspace(-pi, pi, 1025)[:-1]
t = reshape(t, (16, 64))
w = linspace(-512, 512, 65)[:-1]

mag = []
phase = []

#Computing DFT at different ranges of time by using for loop
for time in t:
    y = chirp(time)
    y[0] = 0
    Y = fftshift(fft(fftshift(y)))/64
    mag.append(abs(Y))
    phase.append(angle(Y))

mag = array(mag)
phase = array(phase)


## Plotting 3d plot of Frequency response vs frequency and time
x = w
y = linspace(-pi, pi, 17)[:-1]
X, Y = meshgrid(x, y)

fig = figure(1)
ax = fig.add_subplot(121, projection='3d')
surf = ax.plot_surface(X, Y, mag, cmap=cm.coolwarm)
fig.colorbar(surf,shrink=0.5)
ax.set_title("Surface plot of Magnitude response vs. frequency and time")
ax.set_xlabel("$\\omega$") 
ax.set_ylabel("$t$")
ax.set_zlabel("$|Y|$")
ax = fig.add_subplot(122, projection='3d')
surf = ax.plot_surface(X, Y, phase, cmap=cm.coolwarm)
fig.colorbar(surf,shrink=0.5)
ax.set_title("Surface plot of Phase response vs. frequency and time")
ax.set_xlabel("$\\omega$") 
ax.set_ylabel("$t$")
ax.set_zlabel("$Phase of Y$")
show()



#Computing DFT at different ranges of time by using for loop
mag = []
phase = []

w = linspace(-512, 512, 65)[:-1]

for time in t:
    y = hammingchirp(time)
    y[0] = 0
    Y = fftshift(fft(fftshift(y)))/64
    mag.append(abs(Y))
    phase.append(angle(Y))

mag = array(mag)
phase = array(phase)

x = w
y = linspace(-pi, pi, 17)[:-1]
X, Y = meshgrid(x, y)

## Plotting 3d plot of Frequency response vs frequency and time
fig = figure(1)
ax = fig.add_subplot(121, projection='3d')
surf = ax.plot_surface(X, Y, mag, cmap=cm.coolwarm)
fig.colorbar(surf,shrink=0.5)
ax.set_title("Surface plot of Magnitude response vs. frequency and time with hamming")
ax.set_xlabel("$\\omega$") 
ax.set_ylabel("$t$")
ax.set_zlabel("$|Y|$")
ax = fig.add_subplot(122, projection='3d')
surf = ax.plot_surface(X, Y, phase, cmap=cm.coolwarm)
fig.colorbar(surf,shrink=0.5)
ax.set_title("Surface plot of Phase response vs. frequency and time with hamming")
ax.set_xlabel("$\\omega$") 
ax.set_ylabel("$t$")
ax.set_zlabel("$Phase of Y$")
show()
