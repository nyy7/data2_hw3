from __future__ import division

from pulp import LpVariable,LpProblem,LpMinimize,LpStatus
import pulp
import numpy as np

# read data from file

'''data = np.loadtxt("SAheart.txt")
y = data[:,-1]
X = np.delete(data,0,1)
X = np.delete(X,-1,1)'''
with open("SAheart.txt") as f:
	data = f.readlines()
f.close()
data_matrix = []
for i in range(len(data)):
	data[i] = data[i].replace('\n','')
	data[i] = data[i].replace(' ','')
	instance = data[i].split(",")
	#data_matrix.append(instance)
	for k in range(len(instance)):
		if instance[k] == 'Absent':
			instance[k] = 0
		elif instance[k] =='Present':
			instance[k] = 1
		else:
			instance[k] = float(instance[k])
	data_matrix.append(instance)
matrix_np = np.array(data_matrix)
data = np.delete(matrix_np,0,1)
#y = matrix_np[:,-1].astype(int)
#X = np.delete(matrix_np,0,1)
#X = np.delete(X,-1,1)

# cast linear classification to linear programming problem

class LP():

	def __init__(self,X,y):
		#split data to two set with two class label
		H = []
		M = []
		for i in range(len(y)):
			if y[i] == 1:
				H.append(X[i])
			else:
				M.append(X[i])

			# set of instances with class label 1
		self.H = np.array(H)
			# set of instances with class label 0
		self.M = np.array(M)


		# No. of features
		self.att = len(X[0])

		# No. of instances with class label 1
		self.h = len(H)

		# No. of instances with class label 0
		self.m = len(M)

		#set LP variables

		self.a = []
		self.b = 0
	
	def lp(self): 
		a_name = []
		p_name = []
		n_name = []
		a = []
		p = []
		n = []
		for i in range(self.att):
			a_name.append('a_' + str(i))
		for h in range(self.h):
			p_name.append('p_' + str(h))
		for m in range(self.m):
			n_name.append('n_' + str(m))

		a = [LpVariable(a_name[i]) for i in range(self.att)]
		b = LpVariable('b')

		p = [LpVariable(p_name[i],0) for i in range(self.h)]
		n = [LpVariable(n_name[i],0) for i in range(self.m)]

		#set objective function and constrains
		prob = LpProblem("myProblem",LpMinimize)
		sumP = 0
		sumN = 0
		for i in range(self.h):
			sumX = 0
			for j in range(self.att):
				sumX -= a[j] * self.H[i,j]
			prob += sumX + b + 1 <= p[i]
			sumP += p[i]
		for i in range(self.m):
			sumX = 0
			for j in range(self.att):
				sumX += a[j] * self.M[i,j]
			prob += sumX - b + 1 <= n[i]
			sumN += n[i]
	
		prob += (1/h) * sumP + (1/m) * sumN
		status = prob.solve()

		for i in range(self.att):
			self.a.append(pulp.value(a[i]))
		self.b = pulp.value(b)
	def PRINT(self):	
		print "ax + b = 0"
		print "a:"
		for i in range(self.att):
			print self.a[i]
		print "b:"
		print self.b

#cross validation
#lp = LP(X,y)
#lp.lp()
#lp.PRINT()

def cross_val(data,k_fold):
	test_number = int(len(data)/k_fold)
	accuracies = []
	precisions = []
	recalls = []
	for i in range(k_fold):
		np.random.shuffle(data)
		test_data = data[:test_number]
		train_data = data[test_number:]
		y_train = train_data[:,-1].astype(int)
		X_train = np.delete(train_data,-1,1)
		y_test = test_data[:,-1].astype(int)
		X_test = np.delete(test_data,-1,1)

		lp = LP(X_train,y_train)
		lp.lp()
		a = lp.a
		b = lp.b
		print "The",i + 1,"result:"
		print "a =",lp.a
		print "b =",lp.b
		print

		fp = 0
		tp = 0
		fn = 0
		tn = 0
		for i in range(len(X_test)):
			sum = 0
			for j in range(len(X_test[0])):
				sum += X_test[i,j] * a[j]
			y = sum - b
			if y > 0 and y_test[i] == 1:
				tp += 1
			elif y > 0 and y_test[i] == 0:
				fp += 1
			elif y < 0 and y_test[i] == 1:
				fn += 1
			elif y < 0 and y_test[i] == 0:
				tn += 1
		
		accuracy = (tp + tn)/len(X_test)
		accuracies.append(accuracy)

		precision = tp/(tp + fp)
		precisions.append(precision)

		recall = tp/(tp + fn)
		recalls.append(recall)
	
	
	print "Average Accuracies =",np.mean(accuracies)
	print "Average Precision =",np.mean(precisions)
	print "Average Recall =",np.mean(recalls)

cross_val(data,10)

