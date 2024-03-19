"""
                             EE2703 Applied Programming Lab - 2021
                                 Assignment 8: Solution
                             Arun Krishna A M S - EE19B001
                                     7th May 2021
"""

# Importing necessary libraries
import sympy as sy
import pylab as p
import scipy.signal as sp

#Setting up the variable symbol 's' globally
s = sy.symbols('s')




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Function that converts the transfer function in sympy format to scipy LTI class object by extracting the
coefficients and creating polynomial arrays to get the LTI Class object of the same transfer function
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def from_SympyToScipy(H):

    #Function returns the expression's numerator and denominator separately
    num, denom = sy.fraction(H)

    #Extracting those into polynomials
    numerator = sy.poly(num)     
    denominator = sy.poly(denom)   
    
    #Extracting the coefficients of the polynomials
    numeratorCoeff = numerator.all_coeffs()  
    denominatorCoeff = denominator.all_coeffs()  

    #Feeding the coefficients into the sp.lti system to get LTI class object with same transfer function H
    H_SP = sp.lti(p.array(numeratorCoeff, dtype=float), p.array(denominatorCoeff, dtype=float))

    return H_SP




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Analysis of Low Pass Filter Q1 and Q2
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def lowPass_filter(R1, R2, C1, C2, G, Vi):

    #Compute the symbolic representation of the matrices and returns the  transfer function by computing the inverse
    M = sy.Matrix([[0, 0, 1, -1/G], [-1/(1+R2*C2*s), 1, 0, 0], [0, -G, G, 1], [(1/R1 + 1/R2 + C1*s), -1/R2, 0, -C1*s]])
    N = sy.Matrix([0, 0, 0, Vi/R1])
    V = M.inv()*N
    
    return M, N, V


#Printing the magnitude plot of the low pass filter

#Obtaining the transfer function of the low pass filter by taking Vin = 1
M,N,H = lowPass_filter(10000,10000,1e-9,1e-9,1.586,1)

#Printing the values of the resistors, capacitors etc used in the circuit
print("R1 = 10k ohms")
print("R2 = 10k ohms")
print("C1 = 10nF")
print("C2 = 10nF")
print("G = 1.586")

#Obtaining the transfer function
H = H[3]
print("H(s) = ")
print(H)
w = p.logspace(0, 10, 1001)
ss = 1j * w
H1 = sy.lambdify(s, H, 'numpy')
H2 = H1(ss)

p.loglog(w, abs(H2), linewidth = 2)
p.title("Magnitude plot of lowpass filter")
p.ylabel("$|H(j\\omega)|\\rightarrow$")
p.xlabel("$\\omega$(in rad/s)$\\rightarrow$")
p.grid(True)
p.show()

#Question: 1 Obtaining the step response i.e., Output when the input is a unit step function. 
time = p.linspace(0, 0.001, 1000)

#Unit Step in S domain: 1/s and Vout/Vin = H(s)
Vout = from_SympyToScipy(H*1/s)

#Obtaining the information in time domain
time, stepResponse = sp.impulse(Vout, None, time)

p.plot(time, stepResponse)
p.title("Step response of lowpass filter")
p.ylabel("$v_{o}(t)$(in V)$\\rightarrow$")
p.xlabel("$t$(in s)$\\rightarrow$")
p.grid(True)
p.show()

#Question: 2 Obtaining the output response when the input is V(t) = u(t)sin(2000*pi*t) + u(t)cos(2*10^6*pi*t)  
time = p.linspace(0, 0.01,100000)
Vin = p.sin(2e3*p.pi*time)+p.cos(2e6*p.pi*time)
time, Vout, svec = sp.lsim(from_SympyToScipy(H), Vin, time)

p.plot(time, Vin, label='$V_{in}$')
p.plot(time, Vout, label='$V_{out}$')
p.title("Response for $V(t) = u(t)(sin(2000*\pi*t) + cos(2*10^6*\pi*t))$")
p.ylabel("$V(t)\\rightarrow$")
p.xlabel("$t$(in s)$\\rightarrow$")
p.grid(True)
p.legend()
p.show()






"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Analysis of High Pass Filter Q3, Q4 and Q5
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def highPass_filter(R1, R3, C1, C2, G, Vi):

    #Compute the symbolic representation of the matrices and returns the  transfer function by computing the inverse
    M = sy.Matrix([[0, 0, 1, -1/G], [-R3*C2*s/(1+R3*C2*s), 1, 0, 0], [0, -G, G, 1], [(C1*s + C2*s + 1/R1), -C2*s, 0, -1/R1]])
    N = sy.Matrix([0, 0, 0, Vi*s*C1])
    V = M.inv()*N

    return M, N, V

#Printing the magnitude plot of the high pass filter

#Obtaining the transfer function of the high pass filter by taking Vin = 1
M,N,H = highPass_filter(10000,10000,1e-9,1e-9,1.586,1)

#Printing the values of the resistors, capacitors etc used in the circuit
print("R1 = 10k ohms")
print("R3 = 10k ohms")
print("C1 = 10pF")
print("C2 = 10pF")
print("G = 1.586")

#Obtaining the transfer function
H = H[3]
print("H(s) = ")
print(H)
w = p.logspace(0, 10, 1001)
ss = 1j * w
H1 = sy.lambdify(s, H, 'numpy')
H2 = H1(ss)

p.loglog(w, abs(H2), linewidth = 2)
p.title("Magnitude plot of highpass filter")
p.ylabel("$|H(j\\omega)|\\rightarrow$")
p.xlabel("$\\omega$(in rad/s)$\\rightarrow$")
p.grid(True)
p.show()


#Question: 4 Obtaining the response of the circuit to a damped sinusoid using suitable Vi
#Let us try to plot the response for two different frequencies - 1 Hz and 1 MHz with decay factor = 0.5 and 100

#Frequency = 2pi rad/s and decay factor = 0.5
time = p.linspace(0, 5, 1000)
Vin = p.exp(-0.5*time)*p.sin(2*p.pi*time)
time, Vout, svec = sp.lsim(from_SympyToScipy(H), Vin, time)

p.plot(time, Vin, label='$V_{in}$')
p.plot(time, Vout, label='$V_{out}$')
p.title("Response for $V(t) = e^{-0.5t}*sin(2\pi t)$")
p.ylabel("$V(t)\\rightarrow$")
p.xlabel("$t$(in s)$\\rightarrow$")
p.grid(True)
p.legend()
p.show()

#Frequency = 2pi*10^6 rad/s and decay factor = 100
time = p.linspace(0, 0.01, 200000)
Vin = p.exp(-100*time)*p.sin(2*p.pi*10**6*time)
time, Vout, svec = sp.lsim(from_SympyToScipy(H), Vin, time)

p.plot(time, Vin, label='$V_{in}$')
p.plot(time, Vout, label='$V_{out}$')
p.title("Response for $V(t) = e^{-0.5t}*sin(2\pi x 10^6t)$")
p.ylabel("$V(t)\\rightarrow$")
p.xlabel("$t$(in s)$\\rightarrow$")
p.grid(True)
p.legend()
p.show()



#Question: 5 Obtaining the step response i.e., Output when the input is a unit step function. 
time = p.linspace(0, 0.001, 1000)

#Unit Step in S domain: 1/s and Vout/Vin = H(s)
Vout = from_SympyToScipy(H*1/s)

#Obtaining the information in time domain
time, stepResponse = sp.impulse(Vout, None, time)

p.plot(time, stepResponse)
p.title("Step response of highpass filter")
p.ylabel("$v_{o}(t)$(in V)$\\rightarrow$")
p.xlabel("$t$(in s)$\\rightarrow$")
p.grid(True)
p.show()







