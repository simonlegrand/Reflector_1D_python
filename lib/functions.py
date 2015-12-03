from __future__ import print_function

import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import scipy.sparse as sparse
from scipy.integrate import quad
from scipy.interpolate import interp1d
import scipy.optimize
import pylab
#from pylpsolve import LP
#from cvxopt import spmatrix, matrix, solvers

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
	
def solve_IPFP(mu, nu, C, epsilon):
	
	H = np.exp(-C/epsilon)

	oox = np.ones((1,np.size(mu)))
	ooy = np.ones((np.size(nu),1))
	mu = np.reshape(mu,(1,np.size(mu)))
	nu = np.reshape(nu,(np.size(nu),1))
	b = np.ones((1,np.size(mu)))
	a = np.ones((np.size(nu),1))

	error = 1
	error_min = 1e-3
	count = 1
	
	#### Loop ####
	if type(H) is sparse.csr.csr_matrix:
		while(error > error_min):
			an = a
			a = mu/H.dot(b)
			H_T = H.transpose()
			b = nu/H_T.dot(a)
			error = np.sum(np.absolute(an-a))/np.sum(a)
			print('error at step', count, '=', error)
			count = count + 1
			
		diag_a = sparse.diags(a,0)
		diag_b = sparse.diags(b,0)
		Gamma = diag_a.dot(H.dot(diag_b))
		
	else:
		H = np.exp(-C/epsilon)
		H_T = H.T
		while(error > error_min):

			b = np.divide(mu, np.dot(H_T,a).T)
			an = np.divide(nu, np.dot(H,b.T))

			error = np.sum(np.absolute(an-a))/np.sum(a)
			print('error at step', count, '=', error)
			a = an
			count = count + 1
	
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


def density_change_variable(p,target_plane_base,target_plane_density):

	e_ksi = target_plane_base[0]
	n_plan = target_plane_base[1]
	d = np.linalg.norm(n_plan)
	n_plan = n_plan / d
	
	s2 = np.zeros((np.size(p),2))
	s2[:,0] = 1/(p*p + 1) * 2*p
	s2[:,1] = 1/(p*p + 1) * (p*p - 1)

	ksi = -d * (e_ksi[0]*s2[:,0] + e_ksi[1]*s2[:,1])
	ksi /=  n_plan[0]*s2[:,0] + n_plan[1]*s2[:,1]
	#plt.plot(ksi, target_plane_density(ksi))
	#plt.xlim(min(ksi), max(ksi))
	#plt.show()
	
	G_tilde = (ksi*ksi + d*d)/d * target_plane_density(ksi)
	g = 2 * G_tilde / (p*p + 1)
	g = np.reshape(g,(np.size(g),1))
	
	return g

def planar_to_gradient_boundaries(target_plane_box,target_plane_base):
	"""
	Computes the p-domain boundaries from the target box
	"""
	e_ksi = target_plane_base[0]
	n_plan = target_plane_base[1]
	s1 = np.array([0.,1.])
	# Distance target_plane/reflector
	d = np.linalg.norm(n_plan)

	n_plan = n_plan / d

	s2_bound = np.zeros((2,2))
	s2_bound[0,0] = target_plane_box[0]*e_ksi[0] - d*n_plan[0]
	s2_bound[0,1] = target_plane_box[0]*e_ksi[1] - d*n_plan[1]
	s2_bound[1,0] = target_plane_box[1]*e_ksi[0] - d*n_plan[0]
	s2_bound[1,1] = target_plane_box[1]*e_ksi[1] - d*n_plan[1]

	s2_bound = s2_bound / np.linalg.norm(s2_bound, axis=1)
	
	pmin = -(s2_bound[0,0] - s1[0])/(s2_bound[0,1] - s1[1])
	pmax = -(s2_bound[1,0] - s1[0])/(s2_bound[1,1] - s1[1])

	return pmin,pmax
	
def planar_to_gradient(ksi, base, s1=None):
	"""
	This function computes the surface derivatives of the reflector
	for incident rays s1 and impact points of reflected rays in (eta,ksi)
	Parameters
	----------
	ksi : 1D array
		Coordinate ksi on the target plane
	base : [0]e_ksi : Basis of the target plan
		   [1]n_plan : Normal vector to the target plan
		   Its norm equals distance from plane to reflector.
	s1 : (1,2) array
		Incident ray direction

	Returns
	-------
	p : 1D array
		surface derivatives of the reflector
		
	See Also
	--------
	Inverse Methods for Illumination Optics, Corien Prins, chapter 5.3.1
    """
	e_ksi =base[0]
	n_plan = base[1]
	
	if s1 is None:
		s1 = np.array([0.,1.])
	else:
		s1 = s1 / np.linalg.norm(s1)
	try:
		# Distance target plan/reflector
		d = np.linalg.norm(n_plan)
		if d==0:
			raise ZeroDivisionError
		n_plan = n_plan / d
	
		# Reflected rays
		# The reflector is considered ponctual and
		# as the origin of the coordinate system
		s2 = np.zeros((np.size(ksi),2))
		s2[:,0] = ksi*e_ksi[0] - d*n_plan[0]
		s2[:,1] = ksi*e_ksi[1] - d*n_plan[1]
		
		s2 = s2 / np.linalg.norm(s2, axis=1)[:, np.newaxis]
	
		p = -(s2[:,0] - s1[0])/(s2[:,1] - s1[1])
	
		return p
		
	except ZeroDivisionError:
		print("****planar_to_gradient error")
		
def inverse_transform(func, a, b, N, n_sample=2000):
	"""
	Invert Transform Method implementation
	Returns N random numbers following the func
	probability law.
	f has to be positive on [a,b]
	"""
	x = np.linspace(a, b, n_sample)

	F = np.ones(n_sample)
	dx = (b-a)/float(n_sample)
	for i in range(0,n_sample):
		F[i] = quad(func, a, a + i*dx)[0]

	F /= max(F)
	
	# Interpolation de F^-1
	interp = interp1d(F,x)

	r = np.random.rand(N)
	invert = interp(r)
	#pylab.hist(invert,100)
	#pylab.show()
	return interp(r)
	
