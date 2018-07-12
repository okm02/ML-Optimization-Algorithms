import numpy as np
from loss import *

class First_Order_Optimizer:

	def __init__(batch_size,learning_rate,epsilon):

		self.batch_size = batch_size
		self.gamma = learning_rate
		self.epsilon = epsilon


	def uniform_point_selection(self,A):

		return np.random.randint(0,A.shape[0])

	def mini_batch_stochastic_gradient_descent(self,A,b):

		losses = []
		difference = 0
		weight = np.zeros((A.shape[1],1))
		while difference > epsilon:

			current_loss = least_squares_loss(A,weight,b)
			difference = current_loss - difference
			index = self.uniform_point_selection(A)
			current_points = A[index: index + self.batch_size,:]
			gradient = first_order_least_squares(A,weight,b)
			weight = weight - self.gamma * gradient

		return weight,losses


	def accelerated_gradient_descent(self,A,b):

		raise NotImplementedError
