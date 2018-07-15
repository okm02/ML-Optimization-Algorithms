import numpy as np
from .loss import *

class Coordinate_Based_Optimizers:

	def __init__(self,learning_rate,epsilon,max_iters):

		self.gamma = learning_rate
		self.epsilon = epsilon
		self.max_iters = max_iters

	
	def uniform_random_sampling(self,weight):

		dimension = weight.shape[0] # column vector
		return int(np.random.uniform(0,dimension))

	def importance_sampling(self,weight,coordinate_smoothness):

		dimension = weight.shape[0]
		possible_coordinates = np.arange(0,dimension)
		probabilities = coordinate_smoothness / np.sum(coordinate_smoothness)
		return int(np.random.choice(possible_coordinates,p= probabilities)) 

	def steepest_coordinate(self,gradient):

		return np.argmax(gradient,axis = 1)
	
	
	def select_coordinate(self,A,strategy,params):

		if strategy == 'random' :
			raise NotImplementedError
		elif strategy == 'importance' :
			raise NotImplementedError
		elif strategy == 'steep':
			raise NotImplementedError
		else :
			raise raise ValueError('Method not supported')

	def coordinate_descent(self,A,b,strategy,params):

		losses = []
		iterator = 0
		weight = np.zeros((A.shape[1],1))
		while iterator < self.max_iters:

			
			index = self.uniform_random_sampling(weight)
			basis_vector = np.ones_like(weight)
			coordinate_weight = weight[index] * basis_vector
			data_coordinate = (A[:,index]).reshape((-1,1))
			gradient = first_order_least_squares(data_coordinate.T,coordinate_weight,b)
			weight = weight - self.gamma * gradient
			current_loss = least_squares_loss(A,weight,b)
			losses.append(current_loss)
			iterator += 1
			

		return weight,losses



