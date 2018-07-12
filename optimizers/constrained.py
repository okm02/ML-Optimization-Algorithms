import numpy as np
from loss import *

class Constrained_Optimizers:

	def __init__(batch_size,learning_rate,epsilon):

		self.batch_size = batch_size
		self.gamma = learning_rate
		self.epsilon = epsilon

	
	def projected_gradient_descent(self,A,b):

		raise NotImplementedError


	def frank_wolfe(self,A,b):

		raise NotImplementedError
