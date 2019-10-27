import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

class Dataset(object):
	def __init__(self, X, Y, T, n_candidate, n_safety, n_test, seed=None, include_T=False, include_intercept=True, standardize=False):
		n_train   = n_candidate + n_safety
		n_samples = n_train + n_test
		if include_T and not(T is None):
			X = np.hstack((X, T[:,None]))
		if standardize:
			X = X - X.mean(0)[None,:]
			X = X / X.std(0,ddof=1)[None,:]
		if include_intercept:
			X = np.hstack((X, np.ones(len(X))[:,None]))
		# Store the base datasets
		self._X = X

		self._R = Y
		self._T = T
		self.n_classes = np.unique(Y).shape[0]
		# Compute indices for the splits
		self._inds = {
			'all'   : np.arange(0, n_samples),
			'train' : np.arange(0, n_train),
			'test'  : np.arange(n_train, n_samples),
			'opt'   : np.arange(0, n_candidate),
			'saf'   : np.arange(n_candidate, n_train)
		}
		# Compute indices for T=0/T=1 splits
		for k, inds in list(self._inds.items()):
			self._inds['%s_0'%k] = inds[T[inds]==0]
			self._inds['%s_1'%k] = inds[T[inds]==1]
		# Store the default seed
		self._seed = seed
		

	@property
	def X(self):
		return self._X.copy()
	@property
	def Y(self):
		return self._R.copy()
	@property
	def T(self):
		return self._T.copy()

	@property
	def n_features(self):
		return self._X.shape[1]
	@property
	def n_train(self):
		return len(self._inds['train'])
	@property
	def n_test(self):
		return len(self._inds['test'])
	@property
	def n_optimization(self):
		return len(self._inds['opt'])
	@property
	def n_safety(self):
		return len(self._inds['saf'])


	def _get_splits(self, index_key, t=None):
		if not(t is None):
			index_key += ('_%d' % t)
		inds = self._inds[index_key]
		return self._X[inds], self._R[inds], self._T[inds]

	def _get_splits_by_type(self, index_key, truncate=True, reorder=False, seed=None):
		X0, Y0, _ = self._get_splits(index_key, t=0)
		X1, Y1, _ = self._get_splits(index_key, t=1)
		if reorder:
			rnd = np.random.RandomState(self._seed if (seed is None) else seed)
			I0 = rnd.choice(X0.shape[0], X0.shape[0], replace=False)
			I1 = rnd.choice(X1.shape[0], X1.shape[0], replace=False)
			X0, Y0 = X0[I0], Y0[I0]
			X1, Y1 = X1[I1], Y1[I1]
		if truncate:
			k = min(X0.shape[0], X1.shape[0])
			X0, Y0 = X0[:k], Y0[:k]
			X1, Y1 = X1[:k], Y1[:k]
		return X0, Y0, X1, Y1

	def all_sets(self, t=None):
		return self._get_splits('all', t=t)
	def training_splits(self, t=None):
		return self._get_splits('train', t=t)
	def testing_splits(self, t=None):
		return self._get_splits('test', t=t)
	def optimization_splits(self, t=None):
		return self._get_splits('opt', t=t)
	def safety_splits(self, t=None):
		return self._get_splits('saf', t=t)
		
	def all_sets_by_type(self, truncate=True, reorder=False, seed=None):
		return self._get_splits_by_type('all', truncate=truncate, reorder=reorder, seed=seed)
	def training_splits_by_type(self, truncate=True, reorder=False, seed=None):
		return self._get_splits_by_type('train', truncate=truncate, reorder=reorder, seed=seed)
	def testing_splits_by_type(self, truncate=True, reorder=False, seed=None):
		return self._get_splits_by_type('test', truncate=truncate, reorder=reorder, seed=seed)
	def optimization_splits_by_type(self, truncate=True, reorder=False, seed=None):
		return self._get_splits_by_type('opt', truncate=truncate, reorder=reorder, seed=seed)
	def safety_splits_by_type(self, truncate=True, reorder=False, seed=None):
		return self._get_splits_by_type('saf', truncate=truncate, reorder=reorder, seed=seed)





class RLDataset(object):
	def __init__(self, S, A, R, T, P, n_actions, n_candidate, n_safety, n_test, min_reward, max_reward, gamma=1.0, seed=None, Rc_func=(lambda s,a,r,t: r)):
		n_train   = n_candidate + n_safety
		n_samples = n_train + n_test
		# Store the base datasets
		T = T if not(T is None) else np.zeros(len(S))
		self.gamma = gamma
		self._S = S
		self._A = A
		self._R_raw = R
		self._Rc = np.array([ Rc_func(s,a,r,t) for (s,a,r,t) in zip(S,A,R,T) ])
		self._apply_corrections = True
		self._T = T
		self._P = P
		self.n_actions = n_actions
		self.max_reward = max_reward
		self.min_reward = min_reward
		# Compute indices for the splits
		self._inds = {
			'all'   : np.arange(0, n_samples),
			'train' : np.arange(0, n_train),
			'test'  : np.arange(n_train, n_samples),
			'opt'   : np.arange(0, n_candidate),
			'saf'   : np.arange(n_candidate, n_train)
		}
		# Compute indices for T=0/T=1 splits
		for k, inds in list(self._inds.items()):
			self._inds['%s_0'%k] = inds[T[inds]==0]
			self._inds['%s_1'%k] = inds[T[inds]==1]
		# Store the default seed
		self._seed = seed		

	def enable_R_corrections(self):
		self._apply_corrections = True

	def disable_R_corrections(self):
		self._apply_corrections = False

	@property
	def _R(self):
		if self._apply_corrections:
			return self._Rc
		return self._R_raw

	@property
	def n_features(self):
		return self._S[0].shape[1]
	@property
	def n_train(self):
		return len(self._inds['train'])
	@property
	def n_test(self):
		return len(self._inds['test'])
	@property
	def n_optimization(self):
		return len(self._inds['opt'])
	@property
	def n_safety(self):
		return len(self._inds['saf'])

	def _get_splits(self, index_key, t=None, corrected_R=True):
		if not(t is None):
			index_key += ('_%d' % t)
		inds = self._inds[index_key]
		R = self._R[inds] if corrected_R else self._R_raw[inds]
		return self._S[inds], self._A[inds], R, self._T[inds], self._P[inds]

	def _get_splits_by_type(self, index_key, truncate=True, reorder=False, seed=None, corrected_R=True):
		S0, A0, R0, _ = self._get_splits(index_key, t=0, corrected_R=corrected_R)
		S1, A1, R1, _ = self._get_splits(index_key, t=1, corrected_R=corrected_R)
		if reorder:
			rnd = np.random.RandomState(self._seed if (seed is None) else seed)
			I0 = rnd.choice(S0.shape[0], S0.shape[0], replace=False)
			I1 = rnd.choice(S1.shape[0], S1.shape[0], replace=False)
			S0, A0, R0, P0 = S0[I0], A0[I0], R0[I0], P0[I0]
			S1, A1, R1, P1 = S1[I1], A1[I1], R1[I1], P1[I1]
		if truncate:
			k = min(S0.shape[0], S1.shape[0])
			S0, A0, R0, P0 = S0[:k], A0[:k], R0[:k], P0[:k]
			S1, A1, R1, P1 = S1[:k], A1[:k], R1[:k], P1[:k]
		return S0, A0, R0, P0, S1, A1, R1, P1

	@property
	def S(self):
		return self._S.copy()
	@property
	def A(self):
		return self._A.copy()
	@property
	def R(self):
		return self._R.copy()
	@property
	def T(self):
		return self._T.copy()
	@property
	def P(self):
		return self._P.copy()

	def all_sets(self, t=None, corrected_R=True):
		return self._get_splits('all', t=t, corrected_R=corrected_R)
	def training_splits(self, t=None, corrected_R=True):
		return self._get_splits('train', t=t, corrected_R=corrected_R)
	def testing_splits(self, t=None, corrected_R=True):
		return self._get_splits('test', t=t, corrected_R=corrected_R)
	def optimization_splits(self, t=None, corrected_R=True):
		return self._get_splits('opt', t=t, corrected_R=corrected_R)
	def safety_splits(self, t=None, corrected_R=True):
		return self._get_splits('saf', t=t, corrected_R=corrected_R)
		
	def all_sets_by_type(self, truncate=True, reorder=False, seed=None):
		return self._get_splits_by_type('all', truncate=truncate, reorder=reorder, seed=seed)
	def training_splits_by_type(self, truncate=True, reorder=False, seed=None):
		return self._get_splits_by_type('train', truncate=truncate, reorder=reorder, seed=seed)
	def testing_splits_by_type(self, truncate=True, reorder=False, seed=None):
		return self._get_splits_by_type('test', truncate=truncate, reorder=reorder, seed=seed)
	def optimization_splits_by_type(self, truncate=True, reorder=False, seed=None):
		return self._get_splits_by_type('opt', truncate=truncate, reorder=reorder, seed=seed)
	def safety_splits_by_type(self, truncate=True, reorder=False, seed=None):
		return self._get_splits_by_type('saf', truncate=truncate, reorder=reorder, seed=seed)




class BanditDataset(RLDataset):
	def __init__(self, S, A, R, n_actions, n_candidate, n_safety, n_test, min_reward, max_reward, seed=None, P=None, T=None, Rc_func=(lambda s,a,r,t: r)):
		S = S[:,None,:] # Convert S into a sequence of length-1 trajectories
		A = A[:,None]
		# Compute reference probabilities if they aren't provided
		if P is None:
			self.train_proba_gp()
			X = np.hstack((self._S[:,0,:],self._T[:,None]))
			Ps = self._proba_gp.predict_proba(X)
			P = np.array([ [Ps[i,a]] for i,a in enumerate(A) ])
		else:
			P = P[:,None]
			self._proba_gp = None
		self._return_gp = None
		super().__init__(S, A, R, T, P, n_actions, n_candidate, n_safety, n_test, min_reward, max_reward, seed=seed, gamma=1.0, Rc_func=Rc_func)
		
	def train_proba_gp(self, use_pct=0.1):
		kernel = 1.0 * RBF(1.0)
		self._proba_gp = GaussianProcessClassifier(kernel)
		X = np.hstack((self._S[:,0,:],self._T[:,None]))
		I = np.arange(X.shape[0])
		np.random.shuffle(I)
		n_attempts = 0
		n_train = int(use_pct*X.shape[0])
		while len(np.unique(self._A[I[:n_train]])) < self.n_actions and n_attempts < 100:
			np.random.shuffle(I)
			n_attempts += 1
		if len(np.unique(self._A[I[:n_train]])) < self.n_actions and n_attempts == 100:
			raise RuntimeError('Unable to train GP on a representative sample of actions')
		I = I[:n_train]
		self._proba_gp.fit(X[I],self._A[I][:,0])

	def train_return_gp(self, returns, use_pct=0.1):
		Y = np.zeros(len(self._R))
		for i,r in enumerate(self._R):
			Y[i] = np.where(r==returns)[0]
		kernel = 1.0 * RBF(1.0)
		self._return_gp = GaussianProcessClassifier(kernel)
		
		X = np.hstack((self._S[:,0,:],self._T[:,None],self._A))
		n_train = int(use_pct*X.shape[0])
		I = np.arange(X.shape[0])
		np.random.shuffle(I)
		I = I[:n_train]
		self._return_gp.fit(X[I],Y[I])


	def _get_splits(self, index_key, t=None, flatten=False, corrected_R=True):
		if not(t is None):
			index_key += ('_%d' % t)
		inds = self._inds[index_key]
		S = self._S[inds][:,0,:] if flatten else self._S[inds]
		A = self._A[inds][:,0]   if flatten else self._A[inds]
		P = self._P[inds][:,0]   if flatten else self._P[inds]
		R = self._R[inds] if corrected_R else self._R_raw[inds]
		return S, A, R, self._T[inds], P

	def all_sets(self, t=None, flatten=False, corrected_R=True):
		return self._get_splits('all', t=t, flatten=flatten, corrected_R=corrected_R)
	def training_splits(self, t=None, flatten=False, corrected_R=True):
		return self._get_splits('train', t=t, flatten=flatten, corrected_R=corrected_R)
	def testing_splits(self, t=None, flatten=False, corrected_R=True):
		return self._get_splits('test', t=t, flatten=flatten, corrected_R=corrected_R)
	def optimization_splits(self, t=None, flatten=False, corrected_R=True):
		return self._get_splits('opt', t=t, flatten=flatten, corrected_R=corrected_R)
	def safety_splits(self, t=None, flatten=False, corrected_R=True):
		return self._get_splits('saf', t=t, flatten=flatten, corrected_R=corrected_R)

	def get_simulator(self, use_classification=False):
		# Train the predictor for reference probabilities if it hasn't been defined yet
		if self._proba_gp is None:
			self.train_proba_gp()

		# Train the predictor for returns if it hasn't been defined yet
		returns = np.unique(self._R)
		if self._return_gp is None:
			self.train_return_gp(returns)

		def predict(S, A, T):
			if S.ndim == 1:
				X = np.hstack((S[None,:],np.array([[T]])))
				P = self._proba_gp.predict(X).astype(int)
				X = np.hstack((X, np.array([[A]])))
				R = returns[self._return_gp.predict(X).astype(int)]	
				return S, A, R[0], T, P[0]
			else:
				X = np.hstack((S[:,0,:],T[:,None]))
				P = self._proba_gp.predict(X).astype(int)[:,None]
				X = np.hstack((X, A))
				R = returns[self._return_gp.predict(X).astype(int)]
				return S, A, R, T, P
		return predict