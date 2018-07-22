import numpy as np
from .loss import *
from .firstOrder import First_Order_Optimizer

class Second_Order_Optimizer:

	def __init__(self,learning_rate,epsilon,max_iters):

		self.gamma = learning_rate
		self.epsilon = epsilon
		self.max_iters = max_iters

	def newtons_method(self,A,b):

		'''
		 We have no guarantees on newton's method. The only guarantee is to go epsilon close using sgd.
		 Then switch to newton's method and use it's super log convergence rate to reach optimum
		'''

		full_gd = First_Order_Optimizer(A.shape[0],self.gamma,self.epsilon,None,self.max_iters)
		weight,losses = full_gd.mini_batch_stochastic_gradient_descent(A,b)

		gradient = first_order_least_squares(A,weight,b)
		hessian = second_order_least_squares(A,weight,b)
		weight = weight - np.dot(np.linalg.inv(hessian),gradient)
		current_loss = least_squares_loss(A,weight,b)
		losses.append(current_loss)

		return weight,losses

