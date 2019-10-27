import cma
import numpy as np

from core.optimizers import SMLAOptimizer
from time import time

from scipy.optimize import minimize

class BFGSOptimizer(SMLAOptimizer):

	def __init__(self, n_features, sigma0=1.0, restarts=5, *args, **kwargs):
		self.n_features = n_features
		self.sigma0     = sigma0
		self.restarts   = restarts
	
	@staticmethod
	def cli_key():
		return 'bfgs'

	def minimize(self, evalf, n_iters):
		np.random.seed(np.floor(100000*(time()%10)).astype(int))
		theta_opt = None
		val_min = np.inf
		for _ in range(self.restarts):
			x0 = 2.0 * np.random.random(self.n_features) - 1.0
			res = minimize(evalf, x0=x0, method='Powell', options={'maxiter':n_iters})
			theta = res.x
			if not(theta is None):
				value = evalf(theta)
				if value < val_min:
					theta_opt = theta.copy()
					val_min = value
		return theta_opt, {}
	
	def get_theta(self):
		return None