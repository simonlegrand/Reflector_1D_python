from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class Discrete_density_1D:
	def __init__(self, X, f):
		"""
        This class describes a discrete 1D density defined on the
        finite set of verticies X and which values at theses vertices is
        given by function f.

        Parameters
		---------- 
        X : (N,2) array
        	Verticies, described by an array of shape 
        f : (N,) array.
        	Values of the density at points X
        """
		self.vertices = X
		self.values = f / np.sum(f)


class Plan_1D:
	def __init__(self, bounds, base, dens):
		"""
		This class describes a 1D rectangular plan in a 2D space. It is 		represented	by its bounds [xi_min, xi_max], its local base 
		vec{e_{xi}}, its normal vector vec{n_{plan}} which norm is the 			distance from the plan to origin, and the density L(xi) of light on 		it.

		Parameters
		---------- 
		bounds : 2-list
			[xi_min, xi_max]
		base : (2,2)-list
			base[0] : Coordinates x and y of vec{e_{xi}}
			base[1] : Coordinates x and y of vec{n_{plan}}
		dens : Discrete_density_1D object
			Density on the plan
		"""
		self.bounds = bounds
		self.base = base
		self.dens = dens


def density_change_variable(Target_plan):
	"""
	Transforms the target plan intensity L(xi) into the target of
	the optimal	transport problem g(p), where p=nabla u.
	
	Parameters
	----------
	Target_plan : Plan_1D object
		Describes the target of the PIR.
	
	Returns
	-------
	out : Discrete_density_1D object
		Describes the target density g(p) of the optimal transport
		problem.
		
	References
	----------
	https://project.inria.fr/mokabajour/how-it-works/
	
	"""
	e_xi = Target_plan.base[0]
	n_plan = Target_plan.base[1]
	d = np.linalg.norm(n_plan)
	n_plan = n_plan / d
	
	t_bounds = Target_plan.bounds
	pmin, pmax = planar_to_gradient_boundaries(t_bounds,Target_plan.base)
	Np = np.size(Target_plan.dens.values)
	p = np.linspace(pmin, pmax, Np)

	s2 = np.zeros((np.size(p),2))
	s2[:,0] = 1/(p*p + 1) * 2*p
	s2[:,1] = 1/(p*p + 1) * (p*p - 1)

	xi = -d * (e_xi[0]*s2[:,0] + e_xi[1]*s2[:,1])
	xi /=  n_plan[0]*s2[:,0] + n_plan[1]*s2[:,1]

	G_tilde = (xi*xi + d*d)/d * Target_plan.dens.values
	g = 2 * G_tilde / (p*p + 1)
	
	return Discrete_density_1D(p,g)


def planar_to_gradient_boundaries(t_bounds,base):
	"""
	Computes the p-domain boundaries from the target box.
	
	Parameters
	----------
	t_bounds : 2-list.
		[xi_{min}, xi_{max}] bounds of the target plan.
	base : (2,2)-list
		[0] : Coordinates x and y of vec{e_{xi}}
		[1] : Coordinates x and y of vec{n_{plan}}
		
	Returns
	-------
	pmin, pmax : floats
		Bounds of p = nabla u between which we sample the 
		target of the optimal transport problem.
	"""
	e_xi = base[0]
	n_plan = base[1]
	s1 = np.array([0.,1.])
	d = np.linalg.norm(n_plan)

	n_plan = n_plan / d

	s2_bounds = np.zeros((2,2))
	s2_bounds[0,0] = t_bounds[0]*e_xi[0] - d*n_plan[0]
	s2_bounds[0,1] = t_bounds[0]*e_xi[1] - d*n_plan[1]
	s2_bounds[1,0] = t_bounds[1]*e_xi[0] - d*n_plan[0]
	s2_bounds[1,1] = t_bounds[1]*e_xi[1] - d*n_plan[1]

	s2_bounds = s2_bounds / np.linalg.norm(s2_bounds, axis=1)
	
	pmin = -(s2_bounds[0,0] - s1[0])/(s2_bounds[0,1] - s1[1])
	pmax = -(s2_bounds[1,0] - s1[0])/(s2_bounds[1,1] - s1[1])

	return pmin,pmax


def planar_to_gradient(xi, base, s1=None):
	"""
	This function computes the surface derivatives of the reflector
	for incident rays s1 and impact points of reflected rays in xi
	Parameters
	----------
	xi : (N,) array
		Coordinates xi on the target plane
	base : (2,2)-list
		[0] : Coordinates x and y of vec{e_{xi}}
		[1] : Coordinates x and y of vec{n_{plan}} 
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
	e_xi =base[0]
	n_plan = base[1]
	
	if s1 is None:
		s1 = np.array([0.,1.])
	else:
		s1 = s1 / np.linalg.norm(s1)
	# Distance target plan/reflector
	d = np.linalg.norm(n_plan)
	n_plan = n_plan / d

	# Reflected rays
	# The reflector is considered ponctual and
	# as the origin of the coordinate system
	s2 = np.zeros((np.size(xi),2))
	s2[:,0] = xi*e_xi[0] - d*n_plan[0]
	s2[:,1] = xi*e_xi[1] - d*n_plan[1]
	
	s2 = s2 / np.linalg.norm(s2, axis=1)[:, np.newaxis]

	p = -(s2[:,0] - s1[0])/(s2[:,1] - s1[1])

	return p
