import numpy as np
from .loss import *

class First_Order_Optimizer:

	def __init__(self,batch_size,learning_rate,epsilon,smoothness,max_iters):

		self.batch_size = batch_size
		self.gamma = learning_rate
		self.epsilon = epsilon
		self.max_iters = max_iters
		self.smoothness = smoothness

	def uniform_point_selection(self,A):

		if self.batch_size == A.shape[0]:
			return 0
		else:
			return np.random.randint(0,A.shape[0])

	def mini_batch_stochastic_gradient_descent(self,A,b):

		losses = []
		iterator = 0
		weight = np.zeros((A.shape[1],1))
		while iterator < self.max_iters:

			index = self.uniform_point_selection(A)
			current_points = A[index: index + self.batch_size,:]
			current_bias = b[index: index + self.batch_size,:]
			gradient = first_order_least_squares(current_points,weight,current_bias)
			weight = weight - self.gamma * gradient
			current_loss = least_squares_loss(current_points,weight,current_bias)
			losses.append(current_loss)
			iterator += 1

		return weight,losses


	def nestrov_accelerated_gradient_descent(self,A,b):

		losses = []
		iterator = 0

		# initial points
		x = np.zeros((A.shape[1],1))
		y = x.copy()
		z = x.copy()
	
		tao = 1 / ( (self.gamma * self.smoothness) + 1 )
		tao = tao * self.epsilon		
		

		while iterator < self.max_iters:

			x = tao * z + (1 - tao) * y	
			gradient = first_order_least_squares(A,x,b)
			y = x - ((1/self.smoothness) * self.epsilon) * gradient
			z = z - self.gamma * gradient
			current_loss = least_squares_loss(A,x,b)
			losses.append(current_loss)
			iterator += 1

		return x,losses


















