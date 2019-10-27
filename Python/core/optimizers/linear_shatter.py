import numpy as np
from scipy.spatial          import ConvexHull

from core.optimizers import SMLAOptimizer


class LinearShatterBFOptimizer(SMLAOptimizer):	

	def __init__(self, X, buffer_angle=5.0, has_intercept=False, use_chull=True):
		self._samplef = self.get_linear_samplef(X, buffer_angle, has_intercept, use_chull)

	@staticmethod
	def cli_key():
		return 'linear-shatter'

	def minimize(self, evalf, n_iters, theta0=None):
		min_val   = np.inf
		theta_opt = None
		for inum in range(n_iters):
			theta = self.get_theta()
			val = evalf(theta)  
			if val < min_val:
				theta_opt = theta.copy()
				min_val   = val
		return theta_opt, {}

	def get_theta(self):
		return self._samplef()

	@staticmethod
	def get_linear_samplef(X, buffer_angle=5.0, has_intercept=False, use_chull=True):
		nf = X.shape[1]
		c  = 360.0 / (2*np.pi)
		Z  = X[:,:-1] if has_intercept else X.copy()
		W  = X if not(use_chull) else X[ConvexHull(Z, qhull_options='QJ').vertices]
		WN = W / np.linalg.norm(W,axis=1)[:,None]
		
		def samplef():
			while True:
				t = np.random.normal(0,1,(nf,1))
				t = t / np.linalg.norm(t)
				s = W.dot(t).flatten()
				y = np.sign(s)
				# If the solution shatters the samples, return
				if len(np.unique(y)) == 2:
					return t.flatten()
				# If the DB is close enough to the convex hull, return
				s = WN.dot(t).flatten()
				a = 90.0 - c*np.arccos(np.abs(s).min())
				if a < buffer_angle:
					return t.flatten()
		return samplef