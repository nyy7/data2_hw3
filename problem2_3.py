#!/usr/bin/python
from __future__ import division
import math
from numpy import linalg as LA
import numpy as np
from sklearn.decomposition import KernelPCA


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


#compute kernel matrix
matrix = kernel_matrix([[4,2.9],[2.5,1],[3.5,4],[2,2.1]])
matrix_np = np.array(matrix)
print "Problem2:"
print "The kernel matrix is:" 
print matrix_np


matrix = kernel_matrix2([[1,2],[2,1]])
matrix_np2 = np.array(matrix)


#compute PCA
kPCA = KernelPCA(kernel='precomputed')
matrix_sklearn = kPCA.fit_transform(matrix_np)
lambdas = kPCA.lambdas_
alphas = kPCA.alphas_

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
