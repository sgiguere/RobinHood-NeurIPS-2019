import numpy as np
from functools import partial
from itertools import product
from collections import defaultdict

from core.optimizers import OPTIMIZERS
from utils import rvs

from utils import keyboard
from time import time

import warnings

class SeldonianRLBase:
	def __init__(self, epsilons, deltas, model_type, minimum_return, iw_type_corrections={}):
		self.epsilons = epsilons
		self.deltas   = deltas
		self.model_type      = model_type
		self.model_variables = {}
		self.minimum_return  = minimum_return
		self._vm = rvs.VariableManager(self._preprocessor)
		self._scheck_rvs  = []
		self._ccheck_rvs  = []
		self._eval_rvs    = {}
		self._iw_type_corrections = defaultdict(lambda : 1.0)
		for k, v in iw_type_corrections.items():
			self._iw_type_corrections[k] = v
		self._make_rvs()

	##############################################
	#   Per-definition Override Base Functions   #
	##############################################

	@property
	def n_weights(self):
		# Note: n_features and n_actions are determined when fit() is called, so calling 
		#		this before calling fit() will throw an error.
		return self.n_features * self.n_actions

	# Functions for proccesing random variables
	
	def _make_rvs(self):
		''' Overwrite in subclass.
			Must define SampleSets and RVs for the safety and candidate check scores,
		    and add them to the VariableManager (self._vm).
		    How this function is defined, decides what input samples are needed (and
		    thus, what (key,value) pairs are expected to be returned by preprocessor),
		    and also what variables are used for the safety and candidate checks.'''
		raise NotImplementedError

	def _preprocessor(self, data):
		''' Overwrite in subclass.
		    data is expected to be a dict containing values for X, Y, Yp, and T. 
		    Must output a dict containing (key,value) pairs for all SampleSets.'''
		return data


	#########################
	#   Return Evaluation   #
	#########################

	def get_return(self, S, T, P, returnf=None, A_ref=None, P_ref=None, R_ref=None, seed=None):
		if not(returnf is None): # We have the return function: evaluate true return
			A = sample_actions(P, seed=seed)
			return returnf(S, A, T)
		else: # We don't have the return function. Do an IS estimate
			# This part can throw a divide by zero error if Pr(A|new_policy) = 0, but
			#   it is not a problem. However, if Pr(A|old_policy) = 0, that should 
			#   throw an error because the IS estimate is undefined in that case.
			with warnings.catch_warnings(record=True) as w:
				warnings.simplefilter("ignore", category=RuntimeWarning)
				lP  = np.array([ np.log(_P)[range(len(_P)), _A].sum() for (_P,_A) in zip(P,A_ref) ])
			lPR = np.array([ np.log(_P).sum() for _P in P_ref ])
			C   = np.array([ self._iw_type_corrections[t] for t in T ]) 
			return C * np.exp(lP - lPR) * R_ref


	############################
	#   Action Probabilities   #
	############################

	def get_probf(self, theta=None):
		return partial(self.probs, theta=theta)

	def probs(self, S, theta=None):
		theta = self.theta if (theta is None) else theta
		theta = theta.reshape((self.n_features, self.n_actions))
		pvals = []
		if self.model_type == 'argmax':
			for _S in S:
				V = _S.dot(theta)
				P = np.zeros_like(V)
				P[range(P.shape[0]), np.argmax(V,axis=1)] = 1.0
				pvals.append(P)
			return np.array(pvals)
		elif self.model_type == 'softmax':
			for _S in S:
				V = _S.dot(theta)
				P = np.exp(V)
				P = P / np.sum(P, axis=1)[:,None]
				pvals.append(P)
			return np.array(pvals)
		raise ValueError('NaiveSafeRLBase.action(): Unknown model type \'%s\'.' % self.model_type)


	###################################
	#   Model Evaluation given Data   #
	###################################

	def load_split(self, dataset, split_name, probf=None, returnf=None, seed=None, probf_by_type=False):
		S, A_ref, R_ref, T, P_ref = {
			'safety'    : dataset.safety_splits,
			'candidate' : dataset.optimization_splits,
			'train'     : dataset.training_splits,
			'test'      : dataset.testing_splits
			}[split_name]()
		P = probf(S) if not(probf_by_type) else probf(T)(S)
		R = self.get_return(S, T, P, returnf=returnf, A_ref=A_ref, P_ref=P_ref, R_ref=R_ref, seed=seed)
		vm_data = dict(R=R, R_ref=R_ref, S=S, A_ref=A_ref, T=T, P=P, P_ref=P_ref)
		self._vm.set_data(vm_data)
		return vm_data
		
	def load_safety(self, dataset, probf=None, returnf=None, seed=None, probf_by_type=False):
		return self.load_split(dataset, 'safety', probf=probf, returnf=returnf, seed=seed, probf_by_type=probf_by_type)

	def load_candidate(self, dataset, probf=None, returnf=None, seed=None, probf_by_type=False):
		return self.load_split(dataset, 'candidate', probf=probf, returnf=returnf, seed=seed, probf_by_type=probf_by_type)

	def load_training(self, dataset, probf=None, returnf=None, seed=None, probf_by_type=False):
		return self.load_split(dataset, 'train', probf=probf, returnf=returnf, seed=seed, probf_by_type=probf_by_type)

	def load_testing(self, dataset, probf=None, returnf=None, seed=None, probf_by_type=False):
		return self.load_split(dataset, 'test', probf=probf, returnf=returnf, seed=seed, probf_by_type=probf_by_type)


	##############################
	#   Safety Test Evaluation   #
	##############################

	def safety_test(self):
		''' Compute the safety test thresholds. Assumes the safety split has been loaded. '''
		return np.array([ rv.upper(d, split_delta=True) for (rv,d) in zip(self._scheck_rvs, self.deltas) ])
		
	def predict_safety_test(self):
		''' Estimate the safety test thresholds. Assumes the candidate split has been loaded. '''
		return np.array([ rv.upper(d, split_delta=True, n_scale=self.data_ratio) for (rv,d) in zip(self._ccheck_rvs, self.deltas) ])


	###########################
	#   Candidate Objective   #
	###########################

	def candidate_objective(self, theta, dataset, returnf=None):
		''' Compute the value of the candidate objective given theta. '''
		probf      = self.get_probf(theta)
		vm_data    = self.load_candidate(dataset, probf=probf, returnf=returnf, seed=None, probf_by_type=False)
		thresholds = self.predict_safety_test()
		if any(np.isnan(thresholds)):
			return np.inf
		elif (thresholds <= 0.0).all():
			return -np.mean(vm_data['R'])
		return -self.minimum_return + np.maximum(thresholds,0.0).sum()
		
	def get_optimizer(self, name, dataset, opt_params={}):
		if name == 'cmaes':
			return OPTIMIZERS[name](self.n_weights, sigma0=0.0001, n_restarts=1)
		raise ValueError('RatioNDLC.get_optimizer(): Unknown optimizer \'%s\'.' % name)


	################
	#   Training   #
	################

	def fit(self, dataset, n_iters=1000, optimizer_name='cmaes', theta0=None, opt_params={}, returnf=None):
		self.n_features = dataset.n_features
		self.n_actions  = dataset.n_actions
		self.gamma      = dataset.gamma
		self.data_ratio = dataset.n_safety/dataset.n_optimization

		# Perform candidate selection using the optimizer
		opt = self.get_optimizer(optimizer_name, dataset, opt_params=opt_params)
		c_objective = partial(self.candidate_objective, dataset=dataset, returnf=returnf)
		self.theta, _ = opt.minimize(c_objective, n_iters, theta0=theta0)

		# Record the result of the safety test
		probf      = self.get_probf(self.theta)
		vm_data    = self.load_safety(dataset, probf=probf, returnf=returnf, seed=None)
		thresholds = self.safety_test()
		return is_safe(thresholds)


	##################
	#   Evaluation   #
	##################

	def evaluate(self, dataset, probf=None, override_is_seldonian=False, probf_by_type=False, returnf=None, seed=None):
		meta = {}
		
		# We don't assume to know what model probf uses, so assume that
		#   that any probf passed in as an argument is Non-Seldonian
		if probf is None:
			probf = self.get_probf()
			meta['is_seldonian'] = True
		else:
			meta['is_seldonian'] = override_is_seldonian

		# Record evaluation statistics for each split of the dataset
		for name in ['candidate', 'safety', 'train', 'test']:
			vm_data = self.load_split(dataset, name, probf=probf, returnf=returnf, seed=seed, probf_by_type=probf_by_type)
			for rv_name, rv in self._eval_rvs.items():
				meta['%s_%s' % (name, rv_name)] = self._vm.get(rv.name).value()
			meta['return_%s' % name] = np.mean(vm_data['R'])

		# Record SMLA-specific values or add baseline defaults
		if meta['is_seldonian']:
			# Get safety test thresholds
			self.load_safety(dataset, probf=probf, returnf=returnf, seed=seed, probf_by_type=probf_by_type)
			stest = self.safety_test()
			meta['accept'] = is_safe(stest)
			# Get predicted safety test thresholds
			self.load_candidate(dataset, probf=probf, returnf=returnf, seed=seed, probf_by_type=probf_by_type)
			pred_stest = self.predict_safety_test()
			meta['predicted_accept'] = is_safe(pred_stest)
			# Record threshold values
			for i,(st, pst) in enumerate(zip(stest, pred_stest)):
				meta['co_%d_safety_thresh'  % i] = st
				meta['co_%d_psafety_thresh' % i] = pst
		else:
			meta['accept']                = True
			meta['predicted_accept']      = True
			for i in range(len(self._scheck_rvs)):
				meta['co_%d_safety_thresh'  % i] = np.nan
				meta['co_%d_psafety_thresh' % i] = np.nan
		return meta


###############
#   Helpers   #
###############

def is_safe(thresholds):
	return not(any(np.isnan(thresholds))) and (thresholds <= 0).all()

def sample_actions(probs, seed=None):
	random = np.random.RandomState(seed)
	if np.isnan(probs).any():
		print(probs)
	return np.array([ np.array([ np.random.choice(P.shape[1], p=p) for p in P ]) for P in probs ])