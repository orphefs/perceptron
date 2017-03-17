# -*- coding: utf-8 -*-
"""
Created on Sun May 15 13:56:17 2016

@author: orphefs
"""
# This script implements the perceptron algorithm on a data set of 20 training
# points in R2. Does not converge for some reason, maybe faulty algorithm.

import numpy as np  # NumPy (multidimensional arrays, linear algebra, ...)
import scipy as sp  # SciPy (signal and image processing library)
import matplotlib as mpl         # Matplotlib (2D/3D plotting library)
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
from pylab import *              # Matplotlib's pylab interface



w = np.array([[0.5, 0.3]]).transpose();  # define target function weights w
x = np.zeros(shape=(20,2)); # init x
y = np.zeros(shape=(20,1)); # init y
b = -0.4; # define threshold for target function

# np.dot(w.T,x[[i][:]].T) + b  # define target function line


for i in range(15):
    x[i][:] = np.random.rand(1,2); # populate list with random input vectors
    y[i] = int(sign(np.dot(w.T,x[[i][:]].T) + b )); # populate list
    
negIdxs = np.where(y < 0) # get negative indices of array    
negIdxs = negIdxs[0][:]
posIdxs = np.where(y > 0) # get negative indices of array    
posIdxs = posIdxs[0][:]

fig, ax = plt.subplots()
ax.scatter(x[posIdxs,0],x[posIdxs,1], color='r', marker='x', alpha=.9)
ax.scatter(x[negIdxs,0],x[negIdxs,1], color='b', marker='o', alpha=.9)
ax.plot(x[:,0], (-b - w[0] * x[:,0])/w[1])
#xlim(0,1.2)
#ylim(0,1.2)
#plt.plot() # plot the target function line

# Generate a data set of 20
s = (2,2);
w_est = 10*ones(s);
j = k = 0;
tol = 1;
signFlag = [0]*20;
# randomly pick a misclassified point by checking y(t) != sign(wT(t)x(t) ) 

while any(y != signFlag):
    signFlag[j] = int(sign(np.dot(w_est[:,k-1].T,x[[j-1][:]].T) + b))
    if int(y[j]) != signFlag[j]:
        w_est_new = transpose([w_est[:,k-1]]) + y[j]*transpose([x[j-1,:]])
        w_est = np.append(w_est,w_est_new,axis=1)
        #print w_est[:,k]
        #if k > 1:
        #    tol = norm(w_est[:,k] - w_est[:,k-1])
        #print tol
        k = k + 1
        #print k
    j = j + 1
    if j == 20:
       j = 0


fig, ax = plt.subplots()
ax.scatter(x[posIdxs,0],x[posIdxs,1], color='r', marker='x', alpha=.9)
ax.scatter(x[negIdxs,0],x[negIdxs,1], color='b', marker='o', alpha=.9)
ax.plot(x[:,0], ( -b-abs(w_est[0,k]) * x[:,0])/abs(w_est[1,k]))
xlim(0,1.2)
ylim(0,1.2)

