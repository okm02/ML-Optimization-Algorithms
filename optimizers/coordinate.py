import numpy as np
from loss import *

class Coordinate_Based_Optimizers:

	def __init__(batch_size,learning_rate,epsilon):

		self.batch_size = batch_size
		self.gamma = learning_rate
		self.epsilon = epsilon

	
	def uniform_random_sampling(self,weight):

		raise NotImplementedError

	def importance_sampling(self,weight):

		raise NotImplementedError

	def steepest_coordinate(self,weight):

		raise NotImplementedError
	

	def coordinate_descent(self,A,b):

		raise NotImplementedError
