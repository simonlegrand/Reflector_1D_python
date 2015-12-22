from __future__ import print_function

import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import scipy.sparse as sparse
import scipy.optimize
import pylab

def cost_function(x,y):
	"""
	Compute the cost from x to y
	"""
	return 0.5 * (x-y) * (x-y)
	
	
def cost_matrix(x,y):
	"""
	Compute the cost matrix from x to y
	"""
	[x,y] = np.meshgrid(x,y)
	return cost_function(x, y)
	

def get_threshold(Gamma):
	"""
	Determines the threshold under which elements
	of Gamma are considered null.
	"""
	# Maximums over each dimension of Gamma
	nb_dim = len(np.shape(Gamma))
	Max = []
	Min = []
	
	for i in range (nb_dim):
		Max.append(np.amax(Gamma, axis=i))
		Min.append(np.min(Max[i]))
		
	coeff = 0.5
	threshold = coeff * np.min(Min)
	return threshold
	

def get_nb_active_elements(M, threshold):
	"""
	Return the number of elements in M which are
	superior to the threshold.
	"""
	threshold = threshold * np.ones(np.shape(M))
	J = np.greater(M,threshold)
	return np.sum(J)

def to_coo_sparse(M, threshold):
	"""
	Replace Mij<threshold by zero and return M
	as a sparse matrix in coordinate format.
	"""
	zeros = np.zeros(np.shape(M))
	J = np.isclose(M, zeros, rtol=0, atol=threshold)
	M[J] = zeros[J]
	M = sparse.coo_matrix(M, dtype=np.float64)
	return M
	
	
def build_grid(x,y):
	"""
	Given two regular samples x and y,
	return a grid which elements i,j are
	(x[i],y[j])
	"""
	[gridx,gridy] = np.meshgrid(x,y)
	Nx = len(x)*len(y)
	gridx = np.reshape(gridx,(Nx))
	gridy = np.reshape(gridy,(Nx))
	grid = np.vstack([gridx,gridy]).T
	return grid
	
def interpolation(grid, Gamma):
	I = RegularGridInterpolator(grid, Gamma)
	return I
	
def solve_IPFP(Mu, Nu, C, epsilon):
	"""
	Solve the optimal transport problem between Mu and 
	Nu marginals.
	"""
	mu = np.reshape(Mu.values,(1,np.size(Mu.values)))
	nu = np.reshape(Nu.values,(np.size(Nu.values),1))
	b = np.ones((1,np.size(mu)))
	a = np.ones((np.size(nu),1))
	
	error = 1
	error_min = 1e-3
	count = 1

	H = np.exp(-C/epsilon)
	while(error > error_min):
	#while(count<=2):

		b = np.divide(mu, np.dot(a.T,H))
		#print(b)
		an = np.divide(nu, np.dot(H,b.T))
		#print(an)
		error = np.sum(np.absolute(an-a))/np.sum(a)
		print('error at step', count, '=', error)
		a = an
		count = count + 1

	Gamma = np.multiply(H,(a*b))
	a = np.reshape(a,(np.size(a),))
	b = np.reshape(b,(np.size(b),))
	return Gamma,a,b


def solve_IPFP_sparse(Mu, Nu, C, epsilon):

	mu = np.reshape(Mu.values,(1,np.size(Mu.values)))
	nu = np.reshape(Nu.values,(np.size(Nu.values),1))
	a = np.copy(nu)
	
	error = 1
	error_min = 1e-3
	count = 1
	
	H = np.exp(-C/epsilon)
	threshold = 1e-100
	H = to_coo_sparse(H, threshold).tocsr()
	H_T = H.transpose()

	#### Loop ####
	while(error > error_min):

		b = np.divide(mu, H_T.dot(a).T)
		an = np.divide(nu, H.dot(b.T))

		error = np.sum(np.absolute(an-a))/np.sum(a)
		print('error at step', count, '=', error)
		a = an
		count = count + 1
	
	if sparse.issparse(H):
		H = H.todense()	
	Gamma = np.multiply(H,(a*b))
	a = np.reshape(a,(np.size(a),))
	b = np.reshape(b,(np.size(b),))
	return Gamma,a,b

def solve_LP(mu, nu, C, x0=None):
	
	Nx = np.size(mu)
	Ny = np.size(nu)
	mu = np.reshape(mu,(1,Nx))
	nu = np.reshape(nu,(Ny,1))
	
	f = np.reshape(C.T, (np.size(C),))
	
	# Equality constraints matricies
	# Aeq x = beq
	beq = np.concatenate((mu.T[1:], nu))
	beq = np.reshape(beq,(np.size(beq),))
	i = np.linspace(1,Nx,Nx)
	j = np.linspace(1,Nx*Ny,Nx*Ny)
	[I,J] = np.meshgrid(i,j, indexing='ij')
	#print(I,J)
	Aeq = (J>=(I-1)*Ny+1) * (J<=I*Ny)
	#print(np.tile(np.identity(Nx),(Nx)))
	Aeq = np.concatenate((Aeq[1:,:], np.tile(np.identity(Ny),(Nx))))
	#print(Aeq)
	
	# Linear optimization
	if x0 is None:
		Gamma = scipy.optimize.linprog(f,A_eq=Aeq,b_eq=beq,method='simplex',options={"disp": True,"maxiter": 100000}).x
		Gamma = np.reshape(Gamma,(Ny,Nx),order='F')
	else:

		lp = LP()
		lp.addConstraint(Aeq, "=", beq)
		lp.setObjective(f, mode="minimize")
		lp.setOption(verbosity=4)
		
		x0 = np.reshape(x0, (Nx*Ny), order='F')
		
		print(lp.solve(guess=tx0))
		Gamma = lp.getSolution()
		
	Gamma = np.reshape(Gamma,(Ny,Nx),order='F')
	return Gamma


def derivates(x,u):

	assert(len(x)==len(u))
	
	der = np.zeros(len(u))
	h = x[1] - x[0]
	for i in range(1,len(u)-1):
		der[i] = (u[i+1] - u[i-1]) / (2*h)

	der[0] = (u[1] - u[0]) / h
	der[len(u)-1] = (u[len(u)-1] - u[len(u)-2]) / h
	
	return der
