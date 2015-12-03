""" Module containing functions for the ray tracer"""
from __future__ import print_function
import sys
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import functions as func

class GeometricError(Exception):
	"""
	Base class for Geometric exceptions
	"""
	def __init__(self, arg):
		# Set some exception infomation
		self.msg = arg

class NotProperShapeError(GeometricError):
	"""Raised when inputs have not the proper shape"""
	pass

def reflection(points, I, s1):
	"""
	Computes directions s2 of reflected rays.
	Direct application of Snell & Descartes law.
	
	Parameters
	----------
	s1 : array (1,2)
		direction of incident rays
	points : array (N,2)
		points where the interpolant gradient is evaluated
	I : interpolant of the reflector
		
	Returns
	-------
	s2 : array (N,2)
		Direction of reflected rays(normalized).
	"""	
	gx = I(points)
	#plt.plot(points, gx, 'b.')
	#plt.show()
	#print(gx)
	#J = ~gx.mask
	#gx = gx[J]
	#print(points,gx)
	ny = np.ones(gx.shape)
	n = np.vstack([gx,-ny]).T
	n = n / np.linalg.norm(n, axis=-1)[:, np.newaxis]

	inner_product = s1[0] * n[:,0] + s1[1] * n[:,1]
	s2 = s1 - 2*inner_product[:, np.newaxis] * n
	
	return s2


def ray_tracer(s1,source_density, s_box, t_box, interpol, base, niter=None):
	"""
	This function computes the simulation of reflection on the reflector
	and plot the image produced on the target screen.
	
	Parameters
	----------
	s1 : array (N,2)
		direction of incident rays
	s_box : tuple[2]
		enclosing square box of the source support
		[xmin, xmax]
	t_box : tuple[]
		enclosing square box of the target support
		[ymin, ymax]
	interpol : Cubic interpolant of the reflector
	base : [0]e_ksi : Direct orthonormal 
		   basis of the target plan
		   [1]n_plan : Normal vector to the target plan
	"""	
	e_ksi = base[0]
	n_plan = base[1]
	
	M = None
	if niter is None:
		niter = 1
	for i in xrange(niter):
		nray = 200000
		# Generate source point uniformely
		#points = s_box[0] + (s_box[1] - s_box[0])*np.random.rand(nray)
		points = func.inverse_transform(source_density, s_box[0], s_box[1], nray)
		s2 = reflection(points, interpol, s1)
		
		##### New polar coordinates #####
		# psi is the inclination with respect
		# to -n_plan
		d = np.linalg.norm(n_plan)
		psi = np.arccos(np.inner(-s2,n_plan/d))
		#print(psi)
		
		J = np.less_equal(np.inner(s2,e_ksi),np.zeros(len(s2)))
		psi[J] = -psi[J]
		
		##### Planar coordinates #####
		# computation of intersection of reflected rays
		# on the target plan and selection of points
		# inside t_box
		ksi = d * np.tan(psi)
		ksi_min = np.ones(len(ksi)) * t_box[0]
		ksi_max = np.ones(len(ksi)) * t_box[1]
		K = np.logical_and(np.less_equal(ksi,ksi_max),
						   np.greater_equal(ksi,ksi_min))
		ksi = ksi[K]
		points = points[K]
		
		Miter = fill_sparse_vector(ksi, t_box)
		if M is None:
			M = Miter
		else:
			M += Miter
	
		print("it", i+1,":", (i+1)*nray,"rays thrown")
	M = 255.0*M/np.amax(M)
	
	return M
	
	return

def fill_sparse_vector(x,box):

	h = box[1] - box[0] 
	n_linepix = 512
	nmesh = np.size(x)
	dx = h/n_linepix
	
	i = np.floor((x-box[0])/dx)
	j = np.zeros(nmesh)
	data = np.ones(nmesh)
	
	M = sparse.coo_matrix((data, (i,j)), shape=(n_linepix,1)).todense()
	return M
