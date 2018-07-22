import numpy as np
from .loss import *

class Constrained_Optimizers:

	def __init__(self,learning_rate,epsilon,max_iters):

		self.gamma = learning_rate
		self.epsilon = epsilon
		self.max_iters = max_iters

	def __aggregate_lists(self,list1,list2):
		
		return [(sum(x))[0] for x in zip(list1,list2)]
	
	def euclidean_projection_onto_simplex(self,w):

		sorted_coordinates = np.sort(w)

		vector = sorted_coordinates.cumsum()
		vector = 1 - vector
		vector_averaged = vector / np.arange(1,len(vector)+1)	
		aggr = self.__aggregate_lists(sorted_coordinates,vector_averaged)
		row = max(aggr)
		lambda_ = (1/row) * vector

		result = np.asarray(self.__aggregate_lists(w,lambda_))
		
		return (np.maximum(result,0)).reshape((-1,1))
	
	def projected_gradient_descent(self,A,b):

		losses = []
		iterator = 0
		weight = np.zeros((A.shape[1],1))
		while iterator < self.max_iters:

			gradient = first_order_least_squares(A,weight,b)
			update_step = weight - self.gamma * gradient			


			weight = self.euclidean_projection_onto_simplex(update_step)
			
			current_loss = least_squares_loss(A,weight,b)
			losses.append(current_loss)
			iterator += 1

		return weight,losses		


	def frank_wolfe(self,A,b):

		losses = []
		iterator = 0
		weight = np.zeros((A.shape[1],1))

		while iterator < self.max_iters:

			gradient = first_order_least_squares(A,weight,b)
			linear_minimization_oracle = self.euclidean_projection_onto_simplex(gradient)
			
			learning_rate = (2/ (2 + iterator)) * 0.001

			
			weight = (1 - learning_rate) * weight + (learning_rate * linear_minimization_oracle)
			

			current_loss = least_squares_loss(A,weight,b)
			losses.append(current_loss)
			iterator += 1

		return weight,losses












