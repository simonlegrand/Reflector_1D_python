"""
This program solve the inverse reflector problem in 1D with a planar source emitting towars its normal and and a target located on a plane at infinite distance from the reflector. See https://github.com/simonlegrand/Reflector_1D_julia/blob/master/image/schema.png.
"""
from __future__ import print_function

import sys
sys.path.append('./lib')
import time
import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

import functions as func
import density
import plots
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

debut = time.time()


#### Parameters ####
# Source density
Nx = 1000
source_bounds = [-2.,2.]
x = np.linspace(source_bounds[0], source_bounds[1], Nx)
Mu = density.Discrete_density_1D(x, f(x))

# Target plane density
Np = 1000
target_plan_bounds = [-4.,4.]
xi = np.linspace(target_plan_bounds[0], target_plan_bounds[1], Np)
e_xi = [0.,1.]
n_plan = [-20.,0.]
target_plan_base = [e_xi, n_plan]
Target_plan = density.Plan_1D(target_plan_bounds, target_plan_base,  density.Discrete_density_1D(xi,L(xi)))

# Target for the optimal transport solver
Nu = density.density_change_variable(Target_plan)

C = func.cost_matrix(Mu.vertices,Nu.vertices)
epsilon = 0.006


#### IPFP resolution ####
OT_begin = time.time()
Gamma,a,b = func.solve_IPFP(Mu,Nu,C,epsilon)
#Gamma,a,b = func.solve_IPFP_sparse(Mu,Nu,C,epsilon)
print ("OT resolution:", time.time() - OT_begin, "s")
phi = epsilon*np.log(b)
u = 0.5*x*x - phi


#### Ray tracing ####
grad_u = func.derivates(x, u)
der_interpol = interp.interp1d(x, grad_u, kind='cubic')
s1 = np.array([0.,1.])  #Source rays direction
resim = ray.ray_tracer(s1,f, source_bounds, target_plan_bounds, 					   der_interpol, target_plan_base, niter=5)
print ("Execution time:", time.time() - debut, "s")

#### Plots ####
plots.plot_everything(Mu,Nu,Target_plan,u,Gamma,resim)
plt.show()
