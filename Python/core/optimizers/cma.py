import cma
import numpy as np

from core.optimizers import SMLAOptimizer
from time import time

import warnings


class CMAESOptimizer(SMLAOptimizer):

	def __init__(self, n_features, sigma0=1.0, restarts=20, *args, **kwargs):
		self.n_features = n_features
		self.sigma0     = sigma0
		self.restarts   = restarts

	@staticmethod
	def cli_key():
		return 'cmaes'

	def minimize(self, evalf, n_iters, theta0=None):
		has_theta0 = not(theta0 is None)
		np.random.seed(np.floor(100000*(time()%10)).astype(int))
		theta_opt = None
		val_min = np.inf
		

		theta0 = theta0 if not(theta0 is None) else np.zeros(self.n_features)
		next_theta0 = theta0.copy()
		for _ in range(self.restarts):
			options = {'verb_log':0, 'verbose':-9, 'verb_disp':0, 'tolfun':1e-12, 'seed':make_seed(), 'maxiter':n_iters}
			es = cma.CMAEvolutionStrategy(next_theta0, self.sigma0, options)
			with warnings.catch_warnings():				
				warnings.simplefilter('ignore', category=RuntimeWarning)
				es.optimize(evalf)
			theta = es.result.xbest
			if not(theta is None):
				value = evalf(theta)
				if value < val_min:
					theta_opt = theta.copy()
					val_min = value
			next_theta0 = theta0 + np.random.normal(size=theta0.shape)
		return theta_opt, {}

	def get_theta(self):
		return self._es.ask(1)[0]

def make_seed(digits=8, random_state=np.random):
    return np.floor(random_state.rand()*10**digits).astype(int)