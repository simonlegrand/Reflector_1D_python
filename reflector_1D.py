from __future__ import print_function

import sys
sys.path.append('./lib')
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

import functions as func
import plots
import math
import rayTracing as ray

# Source density	
def f(x):
	#return np.ones((np.size(x)))
	return np.exp(-2*np.power(x,2))


# Target plane density
def L(xi):
	return 0.5*np.exp(-np.power(xi-1.5,2)) + np.exp(-np.power(xi+1.5,2))
	#return np.ones((np.size(xi)))
	#return np.exp(-2*np.power(xi-1.5,2))


#### Solver choice ####
IPFP = 1
LP = 0
debut = time.clock()

#### Parameters ####
# Source and target definition
Nx = 1000
source_box = [-2.,2.]
x = np.linspace(source_box[0], source_box[1], Nx)

Np = 1000
target_plane_box = [-4.,4.]
e_xi = [0.,1.]
n_plan = [-20.,0.]
target_plane_base = [e_xi,n_plan]

#### Boundaries of the p domain ####
[pmin,pmax] = func.planar_to_gradient_boundaries(
						target_plane_box,target_plane_base)

#### Evaluation of g between pmin and pmax ####
p = np.linspace(pmin, pmax, Np)
g = func.density_change_variable(p,target_plane_base,L)

C = func.cost_matrix(x,p)

# Marginals
mu = f(x)
mu /= np.sum(mu)
nu = g
nu /= np.sum(nu)
# For plot
xi = np.linspace(target_plane_box[0], target_plane_box[1], Np)
tar_plane_dens = L(xi)
tar_plane_dens = tar_plane_dens / sum(tar_plane_dens)
print(sum(tar_plane_dens))


if IPFP:
	#### IPFP resolution ####
	epsilon = 0.006
	OT_begin = time.clock()
	[Gamma,a,b] = func.solve_IPFP(mu,nu,C,epsilon)
	print ("OT resolution:", time.clock() - OT_begin, "s")
	phi = epsilon*np.log(b)
	u = 0.5*x*x - phi
	grad_u = func.derivates(x, u)

	plots.plot_everything(x,mu,p,nu,xi,tar_plane_dens,u,Gamma)
	

if LP:
	#### Linear programming resolution ####
	Gamma = func.solve_LP(mu,nu,C)
	print ("OT resolution:", time.clock() - debut, "s")
	
	# Barycentric projection of the transport plan
	grad_u = np.zeros(np.size(x))
	for i in range(0,np.size(x)):
		grad_u[i] = np.dot(Gamma[:,i],p) / np.sum(Gamma[:,i])
	
	# Integration of grad_u
	u = np.zeros(Nx)
	a = source_box[0]
	b = source_box[1]
	dx = (b-a)/float(Nx)
	u[0] = 5.
	for i in range(1,Nx):
		u[i] = u[0] + sum(grad_u[:i-1])*dx 
	
	#plots.plot_everything(x,mu,p,nu,u,Gamma)
	
	
#### Gradient interpolation ####
der_interpol = interp.interp1d(x, grad_u, kind='cubic')
x_prime = np.linspace(source_box[0], source_box[1], 10000)
#plt.plot(x_prime,der_interpol(x_prime),'b-')
#plt.show()

#### Ray tracing ####
s1 = np.array([0.,1.])
resim = ray.ray_tracer(s1,f, source_box, target_plane_box, der_interpol, target_plane_base, niter=10)
print ("Execution time:", time.clock() - debut, "s")
x = np.linspace(target_plane_box[0],target_plane_box[1], np.size(resim))
plt.subplot(224)
plt.plot(x, resim, 'b-', ms=2)
plt.show()

