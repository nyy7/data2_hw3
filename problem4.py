from __future__ import division

from pulp import LpVariable,LpProblem,LpMinimize,LpStatus
import pulp
import numpy as np
import math

# read data from file

data = np.loadtxt("prostate.txt")
y = data[:,-1]
X = np.delete(data,0,1)
X = np.delete(X,-1,1)

# cast linear regression to linear programming problem
a_name = []
e_name = []
a = []
e = []
for i in range(len(X[0])):
	a_name.append('a_' + str(i))
for j in range(len(y)):
	e_name.append('e_' + str(j))
a = [LpVariable(a_name[i]) for i in range(len(X[0]))]
e = [LpVariable(e_name[i],0) for i in range(len(y))]
b = LpVariable('b')
prob = LpProblem("myProblem",LpMinimize)
sumE = 0
for i in range(len(y)):
	sumX = 0
	for j in range(len(X[i])):
		sumX += a[j] * X[i,j] 
	prob += y[i] - sumX - b >= -e[i]
	prob += y[i] - sumX - b <= e[i]
	sumE += e[i] 
prob += sumE
status = prob.solve()

A = []
E = []
print "a:"
for i in range(len(X[0])):
	A.append(pulp.value(a[i]))
	print A[i]
for i in range(len(y)):
	E.append(pulp.value(e[i]))
print "b:"
b = pulp.value(b)
print b
print

#evaluation of regression model
#goodness-of-fit
y_mean = np.mean(y)

y_exp_sum = 0
y_sum = 0
for i in range(len(y)):
	sum = 0
	for i in range(len(X[0])):
		sum += X[1,i]*A[i]
	y_exp = sum + b
	y_exp_sum += (y_exp - y_mean)**2
	y_sum += (y[i] - y_mean)**2

R2 = y_exp_sum/y_sum
print "Goodness_of_fit:"
print "R2 =",R2

print
#Standard error
e_sum = 0
for i in range(len(y)):
	e_sum += (E[i])**2
Se = (1/((len(y) - 2)*e_sum))**2
print "Standard error:"
print "Se =",Se
