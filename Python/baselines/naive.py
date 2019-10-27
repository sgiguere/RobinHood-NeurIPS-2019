import numpy as np
from functools import partial
from itertools import product
from collections import defaultdict

from core import srl_fairness as FairSRL
from core.base.srl import SeldonianRLBase
from core.optimizers import OPTIMIZERS
from utils import rvs

from time import time



class NaiveSafeRLBase(SeldonianRLBase):
	# def __init__(self, epsilons, deltas, model_type, minimum_return, iw_type_corrections={}):
	# 	self.epsilons = epsilons
	# 	self.deltas   = deltas
	# 	self.model_type      = model_type
	# 	self.model_variables = {}
	# 	self.minimum_return  = minimum_return
	# 	self._vm = rvs.VariableManager(self._preprocessor)
	# 	self._scheck_rvs  = []
	# 	self._ccheck_rvs  = []
	# 	self._eval_rvs    = {}
	# 	self._iw_type_corrections = defaultdict(lambda : 1.0)
	# 	for k, v in iw_type_corrections.items():
	# 		self._iw_type_corrections[k] = v
	# 	self._make_rvs()

	# @property
	# def n_weights(self):
	# 	# Note: n_features and n_actions are determined when fit() is called, so calling 
	# 	#		this before calling fit() will throw an error.
	# 	return self.n_features * self.n_actions

	# # Functions for proccesing random variables
	
	# def _make_rvs(self):
	# 	''' Overwrite in subclass.
	# 		Must define SampleSets and RVs for the safety and candidate check scores,
	# 	    and add them to the VariableManager (self._vm).
	# 	    How this function is defined, decides what input samples are needed (and
	# 	    thus, what (key,value) pairs are expected to be returned by preprocessor),
	# 	    and also what variables are used for the safety and candidate checks.'''
	# 	raise NotImplementedError

	# def _preprocessor(self, data):
	# 	''' Overwrite in subclass.
	# 	    data is expected to be a dict containing values for X, Y, Yp, and T. 
	# 	    Must output a dict containing (key,value) pairs for all SampleSets.'''
	# 	return data


	# # Computing return

	# def iw_returns(self, S, A, T, R_ref, P_ref, probf=None, theta=None):
	# 	probf = self.get_probf(theta) if probf is None else probf
	# 	probs = probf(S)
	# 	P = np.array([ p[a] for (p,a) in zip(probs,A) ])
	# 	C = np.array([ self._iw_type_corrections[t] for t in T ]) 
	# 	return C * (P/P_ref) * R_ref

	# # Accessor for different optimizers

	# def get_optimizer(self, name, dataset, opt_params={}):
	# 	if name == 'cmaes':
	# 		return OPTIMIZERS[name](self.n_weights, sigma0=0.0001, n_restarts=1)
	# 	raise ValueError('RatioNDLC.get_optimizer(): Unknown optimizer \'%s\'.' % name)


	# #############################################
	# #   Get the probabilities for each action   #
	# #############################################

	# def get_probf(self, theta=None):
	# 	return partial(self.probs, theta=theta)

	# def probs(self, S, theta=None):
	# 	theta   = self.theta if (theta is None) else theta
	# 	theta = theta.reshape((self.n_features, self.n_actions))
	# 	vals  = S.dot(theta)
	# 	if self.model_type == 'argmax':
	# 		pvals = np.zeros_like(vals)
	# 		pvals[range(pvals.shape[0]), np.argmax(vals,axis=1)] = 1.0
	# 		return pvals
	# 	elif self.model_type == 'softmax':
	# 		evals = np.exp(vals)
	# 		return evals / np.sum(evals, axis=1)[:,None]
	# 	raise ValueError('NaiveSafeRLBase.action(): Unknown model type \'%s\'.' % self.model_type)


	# ###################
	# #   Get actions   #
	# ###################

	# def get_actionf(self, theta=None):
	# 	return partial(self.get_action, theta=theta)

	# def action(self, S=None, theta=None, probf=None, probs=None):
	# 	if (probs is None) and (probf is None):
	# 		probs = self.probs(S, theta=theta)
	# 	elif (probs is None):
	# 		probs = probf(S)
	# 	return np.array([ np.random.choice(probs.shape[1], p=p) for p in probs ])


	# Constraint evaluation

	def safety_test(self, probf, Ss, As, Rs, Ts, Ps, returnf=None):
		P = probf(Ss)
		A = self.action(probs=P)
		R = self.iw_returns(Ss, As, Ts, Rs, Ps, probf) if (returnf is None) else returnf(Ss, A, Ts)
		self._vm.set_data({ 'R':R, 'R_ref':Rs, 'S':Ss, 'A':A, 'A_ref':As, 'T':Ts, 'P':P })
		return np.array([ rv.upper(d, split_delta=True) for (rv,d) in zip(self._scheck_rvs, self.deltas) ])


	# Candidate selection

	def candidate_objective(self, theta, S, A, R, T, P, returnf=None):
		probf  = self.get_probf(theta)
		sc_ubs = self.safety_test(probf, S, A, R, T, P, returnf=returnf)
		if any(np.isnan(sc_ubs)):
			return np.inf
		elif (sc_ubs <= 0.0).all():
			if (returnf is None):
				return -np.mean(self.iw_returns(S, A, T, R, P, theta=theta))
			else:
				-np.mean(returnf(S, A, T))
		return -self.minimum_return + np.maximum(sc_ubs,0.0).sum()

	# Model training

	def fit(self, dataset, n_iters=1000, optimizer_name='cmaes', theta0=None, opt_params={}, returnf=None):
		self.n_features = dataset.n_features
		self.n_actions  = dataset.n_actions

		# Get the optimizer
		opt = self.get_optimizer(optimizer_name, dataset, opt_params=opt_params)

		# Fix the non-theta arguments of the candidate objective
		S, A, R, T, P = dataset.optimization_splits()
		c_objective  = partial(self.candidate_objective, S=S, A=A, R=R, T=T, P=P, returnf=returnf)

		# Perform candidate selection using the optimizer
		self.theta,_ = opt.minimize(c_objective, n_iters, theta0=theta0)
		return True

	# Model evaluation

	def evaluate(self, dataset, probf=None, override_is_seldonian=False, probf_by_type=False, returnf=None):
		ds_ratio = dataset.n_safety / dataset.n_optimization
		meta   = {}
		splits = {'candidate' : dataset.optimization_splits(),
				  'safety'    : dataset.safety_splits(),
				  'train'     : dataset.training_splits(),
				  'test'      : dataset.testing_splits()}

		# We don't assume to know what model probf uses, so assume that
		#   that any probf passed in as an argument is Non-Seldonian
		meta['is_seldonian'] = False

		if probf_by_type:
			probf_gen = probf

		# Record statistics for each split of the dataset
		for name, (S,A_ref,R_ref,T,P_ref) in splits.items():
			if probf_by_type:
				probf = probf_gen(T)
			P = probf(S)
			A = self.action(probs=P)
			R = self.iw_returns(S, A_ref, T, R_ref, P_ref, probf=probf) if (returnf is None) else returnf(S, A, T)
			meta['return_%s' % name] = np.mean(R)
			self._vm.set_data({ 'R':R, 'R_ref':R_ref, 'S':S, 'A':A, 'A_ref':A_ref, 'T':T, 'P':P })
			for rv_name, rv in self._eval_rvs.items():
				meta['%s_%s' % (name, rv_name)] = self._vm.get(rv.name).value()

		meta['accept']                = True
		meta['predicted_accept']      = True
		for i in range(len(self._scheck_rvs)):
			meta['co_%d_safety_thresh'  % i] = np.nan
			meta['co_%d_psafety_thresh' % i] = np.nan
		return meta



######################
#   Group Fairness   #
######################

class GroupFairnessNaiveSRL(FairSRL.GroupFairnessRLMixin(mode='ttest', scaling=2.0), NaiveSafeRLBase):
	def __init__(self, is_positivef, epsilon=0.05, delta=0.05, model_type='softmax', minimum_return=1):
		epsilons = np.array([ epsilon ])
		deltas   = np.array([   delta ])
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return)
		self._is_positive = is_positivef



########################
#   Disparate Impact   #
########################

class DisparateImpactNaiveSRL(FairSRL.DisparateImpactMixin(mode='ttest', scaling=2.0), NaiveSafeRLBase):
	def __init__(self, is_positivef, epsilon=0.05, delta=0.05, model_type='softmax', minimum_return=1):
		epsilons = np.array([ epsilon ])
		deltas   = np.array([   delta ])
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return)	
		self._is_positive = is_positivef



################################################################
#   Tutoring System (Both Genders, Bounded Reference Return)   #
################################################################

class TutoringSystemNaiveSRL(FairSRL.TutoringSystemRLMixin(mode='ttest', scaling=2.0), NaiveSafeRLBase):
	def __init__(self, min_return, max_return, ref_return_T0, ref_return_T1, epsilon_f=0.0, epsilon_m=0.0, delta=0.05, model_type='softmax', minimum_return=1, female_iw_correction=1.0, male_iw_correction=1.0):
		iw_corrections = {0:male_iw_correction, 1:female_iw_correction}
		epsilons = np.array([ epsilon_f, epsilon_m ])
		deltas   = np.array([   delta, delta ]) # Same delta for each constraint
		self._min_return = min_return
		self._max_return = max_return
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return, iw_type_corrections=iw_corrections)
	def _get_return_ranges(self):
		return (self._min_return, self._max_return)

class TutoringSystemBootstrapNaiveSRL(FairSRL.TutoringSystemRLMixin(mode='bootstrap', scaling=2.0), NaiveSafeRLBase):
	def __init__(self, min_return, max_return, ref_return_T0, ref_return_T1, epsilon_f=0.0, epsilon_m=0.0, delta=0.05, model_type='softmax', minimum_return=1, female_iw_correction=1.0, male_iw_correction=1.0):
		iw_corrections = {0:male_iw_correction, 1:female_iw_correction}
		epsilons = np.array([ epsilon_f, epsilon_m ])
		deltas   = np.array([   delta, delta ]) # Same delta for each constraint
		self._min_return = min_return
		self._max_return = max_return
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return, iw_type_corrections=iw_corrections)
	def _get_return_ranges(self):
		return (self._min_return, self._max_return)


#################################################################
#   Tutoring System (Both Genders, Observed Reference Return)   #
#################################################################

class TutoringSystemEmpNaiveSRL(FairSRL.TutoringSystemEmpRLMixin(mode='ttest', scaling=2.0), NaiveSafeRLBase):
	def __init__(self, min_return, max_return, ref_return_T0, ref_return_T1, epsilon_f=0.0, epsilon_m=0.0, delta=0.05, model_type='softmax', minimum_return=1, female_iw_correction=1.0, male_iw_correction=1.0):
		iw_corrections = {0:male_iw_correction, 1:female_iw_correction}
		epsilons = np.array([ epsilon_f, epsilon_m ])
		deltas   = np.array([   delta, delta ]) # Same delta for each constraint
		self._min_return = min_return
		self._max_return = max_return
		self._ref_return_T0 = ref_return_T0
		self._ref_return_T1 = ref_return_T1
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return, iw_type_corrections=iw_corrections)
	def _get_return_ranges(self):
		return (self._min_return, self._max_return)

class TutoringSystemBootstrapEmpNaiveSRL(FairSRL.TutoringSystemEmpRLMixin(mode='bootstrap', scaling=2.0), NaiveSafeRLBase):
	def __init__(self, min_return, max_return, ref_return_T0, ref_return_T1, epsilon_f=0.0, epsilon_m=0.0, delta=0.05, model_type='softmax', minimum_return=1, female_iw_correction=1.0, male_iw_correction=1.0):
		iw_corrections = {0:male_iw_correction, 1:female_iw_correction}
		epsilons = np.array([ epsilon_f, epsilon_m ])
		deltas   = np.array([   delta, delta ]) # Same delta for each constraint
		self._min_return = min_return
		self._max_return = max_return
		self._ref_return_T0 = ref_return_T0
		self._ref_return_T1 = ref_return_T1
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return, iw_type_corrections=iw_corrections)
	def _get_return_ranges(self):
		return (self._min_return, self._max_return)


################################################################
#   Tutoring System (Females only, Bounded Reference Return)   #
################################################################

class TutoringSystemFemaleNaiveSRL(FairSRL.TutoringSystemFemaleRLMixin(mode='ttest', scaling=2.0), NaiveSafeRLBase):
	def __init__(self, min_return, max_return, ref_return_T0, ref_return_T1, epsilon_f=0.0, epsilon_m=0.0, delta=0.05, model_type='softmax', minimum_return=1, female_iw_correction=1.0, male_iw_correction=1.0):
		iw_corrections = {0:male_iw_correction, 1:female_iw_correction}
		epsilons = np.array([ epsilon_f ])
		deltas   = np.array([   delta ])
		self._min_return = min_return
		self._max_return = max_return
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return, iw_type_corrections=iw_corrections)
	def _get_return_ranges(self):
		return (self._min_return, self._max_return)

class TutoringSystemFemaleBootstrapNaiveSRL(FairSRL.TutoringSystemFemaleRLMixin(mode='bootstrap', scaling=2.0), NaiveSafeRLBase):
	def __init__(self, min_return, max_return, ref_return_T0, ref_return_T1, epsilon_f=0.0, epsilon_m=0.0, delta=0.05, model_type='softmax', minimum_return=1, female_iw_correction=1.0, male_iw_correction=1.0):
		iw_corrections = {0:male_iw_correction, 1:female_iw_correction}
		epsilons = np.array([ epsilon_f ])
		deltas   = np.array([   delta ])
		self._min_return = min_return
		self._max_return = max_return
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return, iw_type_corrections=iw_corrections)
	def _get_return_ranges(self):
		return (self._min_return, self._max_return)



#################################################################
#   Tutoring System (Females only, Observed Reference Return)   #
#################################################################

class TutoringSystemFemaleEmpNaiveSRL(FairSRL.TutoringSystemFemaleEmpRLMixin(mode='bootstrap', scaling=2.0), NaiveSafeRLBase):
	def __init__(self, min_return, max_return, ref_return_T0, ref_return_T1, epsilon_f=0.0, epsilon_m=0.0, delta=0.05, model_type='softmax', minimum_return=1, female_iw_correction=1.0, male_iw_correction=1.0):
		iw_corrections = {0:male_iw_correction, 1:female_iw_correction}
		epsilons = np.array([ epsilon_f ])
		deltas   = np.array([   delta ]) # Same delta for each constraint
		self._min_return = min_return
		self._max_return = max_return
		self._ref_return_T0 = ref_return_T0
		self._ref_return_T1 = ref_return_T1
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return, iw_type_corrections=iw_corrections)
	def _get_return_ranges(self):
		return (self._min_return, self._max_return)

class TutoringSystemFemaleBootstrapEmpNaiveSRL(FairSRL.TutoringSystemFemaleEmpRLMixin(mode='bootstrap', scaling=2.0), NaiveSafeRLBase):
	def __init__(self, min_return, max_return, ref_return_T0, ref_return_T1, epsilon_f=0.0, epsilon_m=0.0, delta=0.05, model_type='softmax', minimum_return=1, female_iw_correction=1.0, male_iw_correction=1.0):
		iw_corrections = {0:male_iw_correction, 1:female_iw_correction}
		epsilons = np.array([ epsilon_f ])
		deltas   = np.array([   delta ]) # Same delta for each constraint
		self._min_return = min_return
		self._max_return = max_return
		self._ref_return_T0 = ref_return_T0
		self._ref_return_T1 = ref_return_T1
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return, iw_type_corrections=iw_corrections)
	def _get_return_ranges(self):
		return (self._min_return, self._max_return)

##################################################################################
#   Credit Score (Combination of GroupFairness and Fairness through Awareness)   #
##################################################################################

class CreditScoreNaiveSRL(FairSRL.CreditScoreRLMixin(mode='ttest', scaling=2.0), NaiveSafeRLBase):
	def __init__(self, is_positivef, sample_distf, prob_distf, epsilon_fta=0.05, epsilon_gf=0.05, delta_fta=0.05, delta_gf=0.05, model_type='softmax', minimum_return=1):
		epsilons = np.array([ epsilon_fta, epsilon_gf ])
		deltas   = np.array([   delta_fta,   delta_gf ])
		self._sample_distf = sample_distf
		self._prob_distf = prob_distf
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return)
		self._is_positive = is_positivef
	def _sample_distance(self, S, T):
		return self._sample_distf(S,T)
	def _distribution_distance(self, P):
		return self._prob_distf(P)

class CreditScoreBootstrapNaiveSRL(FairSRL.CreditScoreRLMixin(mode='bootstrap', scaling=2.0), NaiveSafeRLBase):
	def __init__(self, is_positivef, sample_distf, prob_distf, epsilon_fta=0.05, epsilon_gf=0.05, delta_fta=0.05, delta_gf=0.05, model_type='softmax', minimum_return=1):
		epsilons = np.array([ epsilon_fta, epsilon_gf ])
		deltas   = np.array([   delta_fta,   delta_gf ])
		self._sample_distf = sample_distf
		self._prob_distf = prob_distf
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return)
		self._is_positive = is_positivef
	def _sample_distance(self, S, T):
		return self._sample_distf(S,T)
	def _distribution_distance(self, P):
		return self._prob_distf(P)
