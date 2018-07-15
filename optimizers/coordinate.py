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

		return np.argmax(gradient)
	
	
	def select_coordinate(self,vector,strategy,params):

		if strategy == 'random' :
			return self.uniform_random_sampling(vector)
		elif strategy == 'importance' :
			return self.importance_sampling(vector,params[0])
		elif strategy == 'steep':
			if params[1] == 0:
				return self.uniform_random_sampling(vector)
			else:
				last_gradient = (params[2])[len(params[2]) - 1]
				return self.steepest_coordinate(last_gradient)
		else :
			raise ValueError('Method not supported')

	def coordinate_descent(self,A,b,strategy,params):

		losses = []
		previous_gradients = []
		iterator = 0
		weight = np.zeros((A.shape[1],1))
		while iterator < self.max_iters:

			index = self.select_coordinate(weight,strategy,(params,iterator,previous_gradients))
			
			data_coordinate = (A[:,index]).reshape((-1,1))
			bias_coordinate = b[index]
			weight_coordinate = weight[index]
			gradient = first_order_least_squares(data_coordinate,weight_coordinate,bias_coordinate)
			
			basis_vector = np.zeros_like(weight)
			basis_vector[index] = 1
			coordinate_wise_gradient = np.dot(basis_vector,gradient)
			
			previous_gradients.append(gradient)
			weight = weight - self.gamma * gradient
			current_loss = least_squares_loss(A,weight,b)
			losses.append(current_loss)
			iterator += 1
			

		return weight,losses



