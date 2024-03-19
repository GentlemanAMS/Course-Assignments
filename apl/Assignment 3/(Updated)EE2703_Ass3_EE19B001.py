"""
                             EE2703 Applied Programming Lab - 2021
                                 Assignment 3: Solution
                             Arun Krishna A M S - EE19B001
                                  3rd March 2021
"""

import numpy as np
from pylab import *
import scipy.special as SP 
from sys import exit;


"""
Part 2 Assignment
Load fitting.dat (look up loadtxt). The data consists of 10 columns. The first column is time, 
while the remaining columns are data. Extract these.
"""

try:
	data = np.loadtxt("fitting.dat",dtype=float);	
	time = np.array(data[:,0]);
	data = np.array(data[:,1:]);

except:
	print("File not found");
	exit(0);


	
"""
Part 3 Assignment
The data columns correspond to the function 
			'f(t) = 1.05*J2(t) − 0.105*t + n(t)' 
where n(t) corresponds to different amounts of noise added. The noise is given to be normally
distributed. Plot the curves in Figure 0 and add labels to indicate the amount of noise in each.
"""
figure(num=0, figsize=(7,6));
sigma = logspace(-1,-3,9);

for i in range(9):
	plot(time,data[:,i],label = r'$\sigma$=%.3f'%sigma[i]);



"""
Part 4 Assignment
The function f(t) has the same general shape as the data but with unknown coefficients:
					g(t; A, B) = A*J2(t) + B*t
Create a python function g(t,A,B) that computes g(t; A, B) for given A and B. Plot it in Figure 0
for A = 1.05, B = −0.105
"""
def g(t,A,B):
	return (A*SP.jv(2,t) + B*t); 


truevalue = g(time,1.05,-0.105);
plot(time, truevalue, color = 'k', label='truevalue');

title(r'Q4 Data to be fitted to theory',size = 18);
xlabel(r't$\rightarrow$',fontsize=13);	
ylabel(r'f(t)+noise$\rightarrow$',fontsize=13);
legend();
show();



"""
Part 5 Assignment
Generate a plot of the first column of data with error bars. Plot every 5th data item to make the plot
readable. Also plot the exact curve using the function written in part 4 to see how much the data diverges. 
"""
figure(num=1, figsize=(7,6));
plot(time, truevalue, color = 'k', label='f(t)');
errorbar(time[::5], data[::5,0], sigma[0], fmt='ro', label='Errorbar');
title(r'Q5 Data points for $\sigma$ = 0.10 along with exact function',size = 18);
xlabel(r't$\rightarrow$',fontsize=13);	
legend();
show();


"""
Part 6 Assignment
Construction of matrix M and testing of equality
"""
J2 = SP.jv(2,time);
M = c_[J2, time];
A0 = 1.05;
B0 = -0.105;
parameter = np.array([A0, B0]);   
testG = np.matmul(M,parameter);   # matrix obtained by matrix multiplication
# Checking with matrix obtained directly from the function 
print("Checking the equality of the matrices:");
print(np.array_equal(testG,truevalue));




"""
Part 7 Assignment
For A = 0, 0.1, . . . , 2 and B = −0.2, −0.19, . . . , 0, for the data given in columns 1 and 2 of the file,
compute the “mean squared error” between the data ( f k ) and the assumed model. Use the first column of 
data as f k for this part.
"""
#initialization
A = np.linspace(0,2,21);
B = np.linspace(-0.2,0,21);
epsilon = np.zeros((len(A),len(B)),float);

for Ai in range(len(A)):
	for Bi in range(len(B)):
		error = truevalue - g(time,A[Ai],B[Bi]);
		squareError = np.matmul(error,np.transpose(error));
		epsilon[Ai][Bi] = squareError/len(time);



"""
Part 8 Assignment
Plot a contour plot of ε[i][j] and see its structure. Does it have a minimum? Does it have several?
"""
figure(num=2, figsize=(7,6));
grid(linestyle=':');
title(r'Q8 Contour plot of $\epsilon_{ij}$ ',size = 18);
xlabel(r'A$\rightarrow$',fontsize=13);	
ylabel(r'B$\rightarrow$',fontsize=13);
T = contour(A,B,epsilon[:,:],levels = linspace(0,0.3,13));
clabel(T, inline = True, fontsize = 8);

plot(1.05,-0.105,'ro')
annotate('Exact Location',xy=(1.05,-0.105))
show();

""" 
#Calculating the X and Y for minimum value of the epsilon:
minvalue = np.unravel_index(np.argmin(epsilon[:,:]),epsilon[:,:].shape);
print("Parameters corresponding to lowest mean square error:");
print("A = ",A[minvalue[0]]);
print("B = ",B[minvalue[1]]);
plot(A[minvalue[0]],B[minvalue[1]],'ro');
annotate('(%0.2f,%0.2f)'%(A[minvalue[0]],B[minvalue[1]]),(A[minvalue[0]],B[minvalue[1]]));
"""




"""
Part 9 Assignment
Use the Python function lstsq from scipy.linalg to obtain the best estimate of A and B. The
array you created in part 6 is what you need. This is sent to the least squares program.
"""
Est = [np.linalg.lstsq(M,data[:,i],rcond = None)[0] for i in range(9)];
Est = np.asarray(Est);

# Obtaining the error in A and B 
errorX = abs(Est[:,0]-1.05);
errorY = abs(Est[:,1]+0.105);

figure(num=3, figsize=(7,6));
grid(linestyle=':');
title(r'Q9 Variation of error with noise',size = 18);
xlabel(r'Noise Standard Deviation $\rightarrow$',fontsize=13);	
ylabel(r'MS Error$\rightarrow$',fontsize=13);
plot(sigma, errorX, linestyle = '--', marker = 'o', color = 'r', label = 'Aerr');
plot(sigma, errorY, linestyle = '--', marker = 'o', color = 'b', label = 'Berr');
legend();
show();




"""
Part 11 Assignment
Replot the above curve using loglog
"""
figure(num=4, figsize=(7,6));
grid(linestyle=':');
title(r'Q11 Variation of error with noise in logarithmic scale',size = 18);
xlabel(r'Noise Standard Deviation $\rightarrow$',fontsize=13);	
ylabel(r'MS Error$\rightarrow$',fontsize=13);

loglog(sigma, errorX, linestyle = '--', marker = 'o', color = 'r', label = 'Aerr');
loglog(sigma, errorY, linestyle = '--', marker = 'o', color = 'b', label = 'Berr');
stem(sigma,errorX,'-ro');
stem(sigma,(errorY),'-bo');
legend();
show();


























