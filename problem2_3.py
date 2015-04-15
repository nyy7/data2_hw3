#!/usr/bin/python
from __future__ import division
import math
from numpy import linalg as LA
import numpy as np
from sklearn.decomposition import KernelPCA

class RBFKernel:
	def __init__(self,sigma):
		self.sigma = sigma;
		self.gamma = -1/(2 * sigma**2)
	def dot_product(self,a,b):
		number = 0
		for i in range(len(a)):
			number += a[i] * b[i]
		return number
	def value(self,a,b):
		dot_a = self.dot_product(a,a)
		dot_b = self.dot_product(b,b)
		dot_ab = self.dot_product(a,b)
		difference = dot_a + dot_b - 2 * dot_ab
		return math.exp(self.gamma*difference);
	def kernel_matrix(self, data):
		matrix = []
		for k in data:
			matrix.append([])
		for i in range(len(data)):
			for j in range(len(data)):
				matrix[i].append(self.value(data[i],data[j]))
		return matrix

def kernel_matrix(matrix):
	kernel_matrix = []
	for point in matrix:
		kernel_matrix.append([])
	for i in range(len(matrix)):
		for j in range(len(matrix)):
			value = 0
			for k in range(len(matrix[0])):
				value += (matrix[i][k] - matrix[j][k])**2
			kernel_matrix[i].append(value)
	return kernel_matrix

def kernel_matrix2(matrix):
	matrix = np.array(matrix)
	kernel_matrix = []
	for point in matrix:
		kernel_matrix.append([]);
	for i in range(len(matrix)):
		for j in range(len(matrix)):
			transpose_point1 = np.matrix.transpose(matrix[i])
			point2 = matrix[j]
			value = (np.dot(transpose_point1,point2))**2
			kernel_matrix[i].append(value);
	return kernel_matrix
#kernel = RBFKernel(5.0)

#compute kernel matrix
matrix = kernel_matrix([[4,2.9],[2.5,1],[3.5,4],[2,2.1]])
matrix_np = np.array(matrix)

#matrix = kernel_matrix2(np.array([[1,2],[2,1]]))
#matrix_np = np.array(matrix)
print "The kernel matrix is:" 
print matrix_np


matrix = kernel_matrix2([[1,2],[2,1]])
matrix_np2 = np.array(matrix)

kPCA = KernelPCA(kernel='precomputed')
matrix_sklearn = kPCA.fit_transform(matrix_np)
lambdas = kPCA.lambdas_
alphas = kPCA.alphas_
print "problem2:"
print "lambdas:", lambdas
print "alphas:"
print alphas
print
print


kPCA2 = KernelPCA(kernel='precomputed')
matrix_sklearn = kPCA2.fit_transform(matrix_np2)
lambdas = kPCA2.lambdas_
alphas = kPCA2.alphas_
print "problem3:"
print "lambdas:", lambdas
print "alphas:"
print alphas



#print "principle component", kPCA.X_transformed_fit_

#k11 = 2.5**2 + 1**2
#k1x = 2.0 / 3.0 * (matrix[0][0] + matrix[0][1]+matrix[0][2])
#k_total = 0
#for i in range(3):
#	for j in range(3):
#		k_total += matrix[i][j]
#result = k11 - k1x + 1 / 9.0 * k_total
#result_sqrt = math.sqrt(result)
#print "The distance of Xi from mean is:", result_sqrt

#compute center matrix
def center(n,matrix):
	I = np.identity(n)
	s = (n,n)
	one = np.ones(n)
	k = I - np.dot(1/n,one)
	center_matrix = np.dot(np.dot(k,matrix),k)
	return center_matrix


#compute eigen values and eigen vectors
#w,v = LA.eig(np.array(centered_matrix))
#print "The dominant eigen value is:", w
#print "The dominant eigen vector is:", v[0]i


