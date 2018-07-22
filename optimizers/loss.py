import numpy as np


def least_squares_loss(A,x,b):
	
	N = A.shape[0]
	cnst = 1/(2* N)
	return cnst * np.linalg.norm(np.dot(A,x) - b,2)

def first_order_least_squares(A,x,b):

	cnst = 1 / A.shape[0]
	return cnst * ( np.dot(A.T,np.dot(A,x) - b) )

def second_order_least_squares(A,x,b):

	cnst = 1 / A.shape[0]
	return cnst * ( np.dot(A.T,A) )
