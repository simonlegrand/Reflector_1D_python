from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.sparse as sparse

def plot_marginals(x, mu, p, nu, xi, L, ax=None):
	if ax is None:
		fig = plt.figure()
		ax = fig.add_subplot(111)
	ax.plot(x, mu, 'b.', p, nu, 'g.',xi, L, "r.")
	ax.set_title('Marginals - blue=source - red=target plane density - green=target of opt. transport')
	return ax
	
def plot_potential(x, u, ax=None):
	if ax is None:
		fig = plt.figure()
		ax = fig.add_subplot(111)
	ax.set_xlabel('x')
	ax.set_ylabel('u')
	ax.set_title('Convex potential')
	
	ax.plot(x,u ,'r-', ms=1)
	return ax
	
def plot_transport_plan(x, y, Gamma, ax=None):
	if type(Gamma) is sparse.csr.csr_matrix:
		Gamma = Gamma.todense()
	if ax is None:
		fig = plt.figure()
		ax = fig.add_subplot(111)
	ax.set_title('Transport plan')
	ax.set_xlabel('x')
	ax.set_ylabel('p')

	im = plt.imshow(Gamma, cmap=cm.jet, interpolation='none',
                origin='lower', aspect='auto',extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
	im.set_cmap('jet')
	plt.colorbar()
	return ax
	
def plot_resim(bounds, resim, ax=None):
	if ax is None:
		fig = plt.figure()
		ax = fig.add_subplot(111)
	ax.set_title('Resimulation')
	ax.set_xlabel('xi')
	
	x = np.linspace(bounds[0], bounds[1], np.size(resim))
	ax.plot(x, resim, 'b-', ms=2)
	return ax
	
def plot_everything(Mu, Nu, Target_plan, u, Gamma, resim):
	"""
	Plot all the graphs on the same figure
	"""
	
	x,p,xi = Mu.vertices, Nu.vertices, Target_plan.dens.vertices
	mu,nu,L = Mu.values, Nu.values, Target_plan.dens.values
	fig = plt.figure()
	
	ax1 = fig.add_subplot(221)
	plot_marginals(x, mu, p, nu, xi, L, ax=ax1)
	ax2 = fig.add_subplot(222)
	plot_potential(x, u, ax=ax2)
	ax3 = fig.add_subplot(223)
	plot_transport_plan(x, p, Gamma, ax=ax3)
	ax4 = fig.add_subplot(224)
	plot_resim(Target_plan.bounds,resim, ax=ax4)
