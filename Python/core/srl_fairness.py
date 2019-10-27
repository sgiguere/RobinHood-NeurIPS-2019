import numpy as np

from utils import rvs
from core.base.srl import  SeldonianRLBase


def PolicyImprovementRLMixin(mode='ttest', scaling=2.0):
	class PolicyImprovementRLMixin:
		def _get_return_ranges(self):
			raise NotImplementedError('TutoringSystem._get_return_ranges() not defined')
		def _preprocessor(self, data):
			R = data['R']
			R_ref = data['R_ref']
			return { 'R' : R, 'R_ref' : R_ref }
		def _make_rvs(self):
			r_min, r_max = self._get_return_ranges()
			# Variables for expected return given T under the candidate policy
			R  = rvs.BoundedRealSampleSet(name='R', lower=r_min, upper=r_max)
			ER = R.expected_value('E[R]', mode=mode)
			# Variables for expected return given T under the reference policy
			R_ref = rvs.BoundedRealSampleSet(name='R_ref', lower=r_min, upper=r_max)
			ER_ref = R_ref.expected_value('E[R_ref]', mode=mode)
			# Constants
			e = rvs.constant(self.epsilons[0], name='e')
			# BQF and Constraint Objectives
			#   g0(theta) := (E[R_ref] - E[R]) - e
			BQF = rvs.sum( ER_ref, -ER, name='BQF' )
			CO  = rvs.sum( BQF, -e, name='CO' )
			SCO = rvs.sum( BQF, -e, name='SCO', scaling=scaling )
			# Store the sample sets and variables
			self._scheck_rvs  = [ CO  ]
			self._ccheck_rvs  = [ SCO ]
			self._eval_rvs    = { 'bqf_0_mean' : BQF, 'co_0_mean' : CO } 
			# Add the sample sets and variables to the manager
			self._vm = rvs.VariableManager(self._preprocessor)
			self._vm.add_sample_set(R, R_ref)
			self._vm.add(ER, ER_ref, BQF, CO, SCO)
	return PolicyImprovementRLMixin

class PolicyImprovementSRL(PolicyImprovementRLMixin(mode='ttest', scaling=2.0), SeldonianRLBase):
	def __init__(self, min_return, max_return, epsilon=0.0, delta=0.05, model_type='softmax', minimum_return=1):
		epsilons = np.array([ epsilon ])
		deltas   = np.array([   delta ])
		self._min_return = min_return
		self._max_return = max_return
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return)
	def _get_return_ranges(self):
		return (self._min_return, self._max_return)

class PolicyImprovementBootstrapSRL(PolicyImprovementRLMixin(mode='bootstrap', scaling=2.0), SeldonianRLBase):
	def __init__(self, min_return, max_return, epsilon=0.0, delta=0.05, model_type='softmax', minimum_return=1):
		epsilons = np.array([ epsilon ])
		deltas   = np.array([   delta ])
		self._min_return = min_return
		self._max_return = max_return
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return)
	def _get_return_ranges(self):
		return (self._min_return, self._max_return)





def PolicyImprovementEmpRLMixin(mode='ttest', scaling=2.0):
	class PolicyImprovementEmpRLMixin:
		def _get_return_ranges(self):
			raise NotImplementedError('TutoringSystem._get_return_ranges() not defined')
		def _preprocessor(self, data):
			R = data['R']
			return { 'R' : R }
		def _make_rvs(self):
			r_min, r_max = self._get_return_ranges()
			# Variables for expected return given T under the candidate policy
			R  = rvs.BoundedRealSampleSet(name='R', lower=r_min, upper=r_max)
			ER = R.expected_value('E[R]', mode=mode)
			ER_ref = rvs.constant(self._ref_return, name='E[R_ref]')
			# Constants
			e = rvs.constant(self.epsilons[0], name='e')
			# BQF and Constraint Objectives
			#   g0(theta) := (E[R_ref] - E[R]) - e
			BQF = rvs.sum( ER_ref, -ER, name='BQF' )
			CO  = rvs.sum( BQF, -e, name='CO' )
			SCO = rvs.sum( BQF, -e, name='SCO', scaling=scaling )
			# Store the sample sets and variables
			self._scheck_rvs  = [ CO  ]
			self._ccheck_rvs  = [ SCO ]
			self._eval_rvs    = { 'bqf_0_mean' : BQF, 'co_0_mean' : CO } 
			# Add the sample sets and variables to the manager
			self._vm = rvs.VariableManager(self._preprocessor)
			self._vm.add_sample_set(R)
			self._vm.add(ER, ER_ref, BQF, CO, SCO)
	return PolicyImprovementEmpRLMixin

class PolicyImprovementEmpSRL(PolicyImprovementEmpRLMixin(mode='ttest', scaling=2.0), SeldonianRLBase):
	def __init__(self, min_return, max_return, epsilon=0.0, delta=0.05, ref_return=0.0, model_type='softmax', minimum_return=1):
		epsilons = np.array([ epsilon ])
		deltas   = np.array([   delta ])
		self._min_return = min_return
		self._max_return = max_return
		self._ref_return = ref_return
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return)
	def _get_return_ranges(self):
		return (self._min_return, self._max_return)

class PolicyImprovementBootstrapEmpSRL(PolicyImprovementEmpRLMixin(mode='bootstrap', scaling=2.0), SeldonianRLBase):
	def __init__(self, min_return, max_return, epsilon=0.0, delta=0.05, ref_return=0.0, model_type='softmax', minimum_return=1):
		epsilons = np.array([ epsilon ])
		deltas   = np.array([   delta ])
		self._min_return = min_return
		self._max_return = max_return
		self._ref_return = ref_return
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return)
	def _get_return_ranges(self):
		return (self._min_return, self._max_return)




######################
#   Group Fairness   #
######################

def GroupFairnessRLMixin(mode='ttest', scaling=2.0):
	class GroupFairness:
		def _is_positive(self):
			raise NotImplementedError('GroupFairness._is_positive undefined.')
		def _preprocessor(self, data):
			T = data['T']
			P = data['P'][:,0,:]
			A = np.argmax(P,1)
			# A = data['A']
			# Split Y and Yp into two sets based T
			A0 = A[np.where(T==0)[0]].astype(int)
			A1 = A[np.where(T==1)[0]].astype(int)
			# Compute indicators for "positive" action predictions
			Pos0 = np.array([ self._is_positive(a) for a in A0 ]).astype(np.float)
			Pos1 = np.array([ self._is_positive(a) for a in A1 ]).astype(np.float)
			# Compute type-dependent indicators for positive predictions
			return { 'Pos0' : Pos0,
					 'Pos1' : Pos1 }
		def _make_rvs(self):
			# Sample sets for false-positive classifications conditioned on T
			Pos0 = rvs.BoundedRealSampleSet(name='Pos0', lower=0, upper=1)
			Pos1 = rvs.BoundedRealSampleSet(name='Pos1', lower=0, upper=1)
			# Variables representing the conditional false-positive rates and the BQF
			PrPos0 = Pos0.expected_value('Pr(Pos|T=0)', mode=mode)
			PrPos1 = Pos1.expected_value('Pr(Pos|T=1)', mode=mode)
			# Constants
			e = rvs.constant(self.epsilons[0], name='e')
			# BQF
			BQF = rvs.abs( PrPos0-PrPos1, name='BQF' )
			CO  = rvs.sum( BQF, -e, name='CO' )
			SCO = rvs.sum( BQF, -e, name='SCO', scaling=scaling )
			# Store the sample sets and variables
			self._scheck_rvs  = [ CO  ]
			self._ccheck_rvs  = [ SCO ]
			self._eval_rvs    = { 'bqf_0_mean' : BQF, 'co_0_mean' : CO } 
			# Add the sample sets and variables to the manager
			self._vm = rvs.VariableManager(self._preprocessor)
			self._vm.add_sample_set(Pos0, Pos1)
			self._vm.add(PrPos0, PrPos1, BQF, CO, SCO)
	return GroupFairness

class GroupFairnessSRL(GroupFairnessRLMixin(mode='ttest', scaling=2.0), SeldonianRLBase):
	def __init__(self, is_positivef, epsilon=0.05, delta=0.05, model_type='softmax', minimum_return=1):
		epsilons = np.array([ epsilon ])
		deltas   = np.array([   delta ])
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return)
		self._is_positive = is_positivef

class GroupFairnessBootstrapSRL(GroupFairnessRLMixin(mode='bootstrap', scaling=2.0), SeldonianRLBase):
	def __init__(self, is_positivef, epsilon=0.05, delta=0.05, model_type='softmax', minimum_return=1):
		epsilons = np.array([ epsilon ])
		deltas   = np.array([   delta ])
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return)
		self._is_positive = is_positivef




########################
#   Disparate Impact   #
########################

def DisparateImpactMixin(mode='ttest', scaling=2.0):
	class DisparateImpact:
		def _is_positive(self):
			raise NotImplementedError('DisparateImpact._is_positive undefined.')
		def _preprocessor(self, data):
			T = data['T']
			P = data['P'][:,0,:]
			A = np.argmax(P,1)
			# Split Y and Yp into two sets based T
			A0 = A[np.where(T==0)[0]].astype(int)
			A1 = A[np.where(T==1)[0]].astype(int)
			# Compute indicators for "positive" action predictions
			Pos0 = np.array([ self._is_positive(a) for a in A0 ]).astype(np.float)
			Pos1 = np.array([ self._is_positive(a) for a in A1 ]).astype(np.float)
			# Compute type-dependent indicators for positive predictions
			return { 'Pos0' : Pos0,
					 'Pos1' : Pos1 }
		def _make_rvs(self):
			# Sample sets for false-positive classifications conditioned on T
			Pos0 = rvs.BoundedRealSampleSet(name='Pos0', lower=0, upper=1)
			Pos1 = rvs.BoundedRealSampleSet(name='Pos1', lower=0, upper=1)
			# Variables representing the conditional false-positive rates and the BQF
			PrPos0 = Pos0.expected_value('Pr(Pos|T=0)', mode=mode)
			PrPos1 = Pos1.expected_value('Pr(Pos|T=1)', mode=mode)
			# Constants
			pct = rvs.constant(self.epsilons[0], name='pct')
			# BQF
			BQF = rvs.maxrec( -(PrPos0/PrPos1), name='BQF')
			CO  = rvs.sum( BQF, -pct, name='CO' )
			SCO = rvs.sum( BQF, -pct, name='SCO', scaling=scaling )
			# Store the sample sets and variables
			self._scheck_rvs  = [ CO  ]
			self._ccheck_rvs  = [ SCO ]
			self._eval_rvs    = { 'bqf_0_mean' : BQF, 'co_0_mean' : CO } 
			# Add the sample sets and variables to the manager
			self._vm = rvs.VariableManager(self._preprocessor)
			self._vm.add_sample_set(Pos0, Pos1)
			self._vm.add(PrPos0, PrPos1, BQF, CO, SCO)
	return DisparateImpact

class DisparateImpactSRL(DisparateImpactMixin(mode='ttest', scaling=2.0), SeldonianRLBase):
	def __init__(self, is_positivef, epsilon=0.05, delta=0.05, model_type='softmax', minimum_return=1):
		epsilons = np.array([ epsilon ])
		deltas   = np.array([   delta ])
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return)	
		self._is_positive = is_positivef



################################################################
#   Tutoring System (Both Genders, Bounded Reference Return)   #
################################################################

def TutoringSystemRLMixin(mode='ttest', scaling=2.0):
	class TutoringSystem:
		def _get_return_ranges(self):
			raise NotImplementedError('TutoringSystem._get_return_ranges() not defined')
		def _preprocessor(self, data):
			T = data['T']
			R = data['R']
			R_ref = data['R_ref']
			# Split R and R_ref into two sets based T
			R0 = R[np.where(T==0)[0]] / self._iw_type_corrections[0] # Undo any type-based importance weighting
			R1 = R[np.where(T==1)[0]] / self._iw_type_corrections[1] # Undo any type-based importance weighting
			R_ref0 = R_ref[np.where(T==0)[0]]
			R_ref1 = R_ref[np.where(T==1)[0]]
			# Compute type-dependent indicators for positive predictions
			return { 'R0' : R0, 'R_ref0' : R_ref0,
					 'R1' : R1, 'R_ref1' : R_ref1 }
		def _make_rvs(self):
			r_min, r_max = self._get_return_ranges()
			# Variables for expected return given T under the candidate policy
			R0  = rvs.BoundedRealSampleSet(name='R0', lower=r_min, upper=r_max)
			R1  = rvs.BoundedRealSampleSet(name='R1', lower=r_min, upper=r_max)
			ER0 = R0.expected_value('E[R|T=0]', mode=mode)
			ER1 = R1.expected_value('E[R|T=1]', mode=mode)
			# Variables for expected return given T under the reference policy
			R_ref0 = rvs.BoundedRealSampleSet(name='R_ref0', lower=r_min, upper=r_max)
			R_ref1 = rvs.BoundedRealSampleSet(name='R_ref1', lower=r_min, upper=r_max)
			ER_ref0 = R_ref0.expected_value('E[R_ref|T=0]', mode=mode)
			ER_ref1 = R_ref1.expected_value('E[R_ref|T=1]', mode=mode)
			# Constants
			e = rvs.constant(self.epsilons[0], name='e')
			# BQF and Constraint Objectives
			#   g0(theta) := (E[R_ref|T=0] - E[R|T=0]) - e
			#   g1(theta) := (E[R_ref|T=1] - E[R|T=1]) - e
			BQF0 = rvs.sum( ER_ref0, -ER0, name='BQF0' )
			BQF1 = rvs.sum( ER_ref1, -ER1, name='BQF1' )
			CO0  = rvs.sum( BQF0, -e, name='CO0' )
			SCO0 = rvs.sum( BQF0, -e, name='SCO0', scaling=scaling )
			CO1  = rvs.sum( BQF1, -e, name='CO1' )
			SCO1 = rvs.sum( BQF1, -e, name='SCO1', scaling=scaling )
			# Store the sample sets and variables
			self._scheck_rvs  = [ CO0,  CO1  ]
			self._ccheck_rvs  = [ SCO0, SCO1 ]
			self._eval_rvs    = { 'bqf_0_mean' : BQF0, 'co_0_mean' : CO0, 'bqf_1_mean' : BQF1, 'co_1_mean' : CO1 } 
			# Add the sample sets and variables to the manager
			self._vm = rvs.VariableManager(self._preprocessor)
			self._vm.add_sample_set(R0, R_ref0, R1, R_ref1)
			self._vm.add(ER0, ER_ref0, ER1, ER_ref1, BQF0, CO0, SCO0, BQF1, CO1, SCO1)
	return TutoringSystem

class TutoringSystemSRL(TutoringSystemRLMixin(mode='ttest', scaling=2.0), SeldonianRLBase):
	def __init__(self, min_return, max_return, ref_return_T0, ref_return_T1, epsilon_f=0.0, epsilon_m=0.0, delta=0.05, model_type='softmax', minimum_return=1, female_iw_correction=1.0, male_iw_correction=1.0):
		iw_corrections = {0:male_iw_correction, 1:female_iw_correction}
		epsilons = np.array([ epsilon_f, epsilon_m ])
		deltas   = np.array([   delta, delta ]) # Same delta for each constraint
		self._min_return = min_return
		self._max_return = max_return
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return, iw_type_corrections=iw_corrections)
	def _get_return_ranges(self):
		return (self._min_return, self._max_return)

class TutoringSystemBootstrapSRL(TutoringSystemRLMixin(mode='bootstrap', scaling=2.0), SeldonianRLBase):
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

def TutoringSystemEmpRLMixin(mode='ttest', scaling=2.0):
	class TutoringSystemEmp:
		def _get_return_ranges(self):
			raise NotImplementedError('TutoringSystemEmp._get_return_ranges() not defined')
		def _preprocessor(self, data):
			T = data['T']
			R = data['R']
			# Split R and R_ref into two sets based T
			R0 = R[np.where(T==0)[0]] / self._iw_type_corrections[0] # Undo any type-based importance weighting
			R1 = R[np.where(T==1)[0]] / self._iw_type_corrections[1] # Undo any type-based importance weighting
			# Compute type-dependent indicators for positive predictions
			return { 'R0' : R0,
					 'R1' : R1 }
		def _make_rvs(self):
			r_min, r_max = self._get_return_ranges()
			# Variables for expected return given T under the candidate policy
			R0  = rvs.BoundedRealSampleSet(name='R0', lower=r_min, upper=r_max)
			R1  = rvs.BoundedRealSampleSet(name='R1', lower=r_min, upper=r_max)
			ER0 = R0.expected_value('E[R|T=0]', mode=mode)
			ER1 = R1.expected_value('E[R|T=1]', mode=mode)
			# Constants
			eM = rvs.constant(self.epsilons[0], name='eM')
			eF = rvs.constant(self.epsilons[1], name='eF')
			r_ref0 = rvs.constant(self._ref_return_T0, name='Avg[R_ref|T=0]')
			r_ref1 = rvs.constant(self._ref_return_T1, name='Avg[R_ref|T=1]')
			# BQF and Constraint Objectives
			#   g0(theta) := (E[R_ref|T=0] - Average(R|T=0,D) - eM
			#   g1(theta) := (E[R_ref|T=1] - Average(R|T=1,D) - eF
			BQF0 = rvs.sum( r_ref0, -ER0, name='BQF0' )
			BQF1 = rvs.sum( r_ref1, -ER1, name='BQF1' )
			CO0  = rvs.sum( BQF0, -eM, name='CO0' )
			SCO0 = rvs.sum( BQF0, -eM, name='SCO0', scaling=scaling )
			CO1  = rvs.sum( BQF1, -eF, name='CO1' )
			SCO1 = rvs.sum( BQF1, -eF, name='SCO1', scaling=scaling )
			# Store the sample sets and variables
			self._scheck_rvs  = [ CO0,  CO1  ]
			self._ccheck_rvs  = [ SCO0, SCO1 ]
			self._eval_rvs    = { 'bqf_0_mean' : BQF0, 'co_0_mean' : CO0, 'bqf_1_mean' : BQF1, 'co_1_mean' : CO1 } 
			# Add the sample sets and variables to the manager
			self._vm = rvs.VariableManager(self._preprocessor)
			self._vm.add_sample_set(R0, R1)
			self._vm.add(ER0, ER1, BQF0, CO0, SCO0, BQF1, CO1, SCO1)
	return TutoringSystemEmp

class TutoringSystemEmpSRL(TutoringSystemEmpRLMixin(mode='ttest', scaling=2.0), SeldonianRLBase):
	def __init__(self, min_return, max_return, ref_return_T0, ref_return_T1, epsilon_f=0.0, epsilon_m=0.0, delta=0.05, model_type='softmax', minimum_return=1, female_iw_correction=1.0, male_iw_correction=1.0):
		iw_corrections = {0:male_iw_correction, 1:female_iw_correction}
		epsilons = np.array([ epsilon_m, epsilon_f ])
		deltas   = np.array([   delta, delta ]) # Same delta for each constraint
		self._min_return = min_return
		self._max_return = max_return
		self._ref_return_T0 = ref_return_T0
		self._ref_return_T1 = ref_return_T1
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return, iw_type_corrections=iw_corrections)
	def _get_return_ranges(self):
		return (self._min_return, self._max_return)

class TutoringSystemBootstrapEmpSRL(TutoringSystemEmpRLMixin(mode='bootstrap', scaling=2.0), SeldonianRLBase):
	def __init__(self, min_return, max_return, ref_return_T0, ref_return_T1, epsilon_f=0.0, epsilon_m=0.0, delta=0.05, model_type='softmax', minimum_return=1, female_iw_correction=1.0, male_iw_correction=1.0):
		iw_corrections = {0:male_iw_correction, 1:female_iw_correction}
		epsilons = np.array([ epsilon_m, epsilon_f ])
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

def TutoringSystemFemaleRLMixin(mode='ttest', scaling=2.0):
	class TutoringSystemFemale:
		def _get_return_ranges(self):
			raise NotImplementedError('TutoringSystemFemale._get_return_ranges() not defined')
		def _preprocessor(self, data):
			T = data['T']
			R = data['R']
			R_ref = data['R_ref']
			# Split R and R_ref into two sets based T
			R1 = R[np.where(T==1)[0]] / self._iw_type_corrections[1] # Undo any type-based importance weighting
			R_ref1 = R_ref[np.where(T==1)[0]]
			# Compute type-dependent indicators for positive predictions
			return { 'R1' : R1, 'R_ref1' : R_ref1 }
		def _make_rvs(self):
			r_min, r_max = self._get_return_ranges()
			# Variables for expected return given T under the candidate policy
			R1  = rvs.BoundedRealSampleSet(name='R1', lower=r_min, upper=r_max)
			ER1 = R1.expected_value('E[R|T=1]', mode=mode)
			# Variables for expected return given T under the reference policy
			R_ref1 = rvs.BoundedRealSampleSet(name='R_ref1', lower=r_min, upper=r_max)
			ER_ref1 = R_ref1.expected_value('E[R_ref|T=1]', mode=mode)

			# Constants
			e = rvs.constant(self.epsilons[0], name='e')
			# BQF and Constraint Objectives
			#   g(theta) := (E[R_ref|T=1] - E[R|T=1]) - e
			BQF = rvs.sum( ER_ref1, -ER1, name='BQF' )
			CO  = rvs.sum( BQF, -e, name='CO' )
			SCO = rvs.sum( BQF, -e, name='SCO', scaling=scaling )
			# Store the sample sets and variables
			self._scheck_rvs  = [ CO  ]
			self._ccheck_rvs  = [ SCO ]
			self._eval_rvs    = { 'bqf_0_mean' : BQF, 'co_0_mean' : CO } 
			# Add the sample sets and variables to the manager
			self._vm = rvs.VariableManager(self._preprocessor)
			self._vm.add_sample_set(R1, R_ref1)
			self._vm.add(ER1, ER_ref1, BQF, CO, SCO)
	return TutoringSystemFemale

class TutoringSystemFemaleSRL(TutoringSystemFemaleRLMixin(mode='ttest', scaling=2.0), SeldonianRLBase):
	def __init__(self, min_return, max_return, ref_return_T0, ref_return_T1, epsilon_f=0.0, epsilon_m=0.0, delta=0.05, model_type='softmax', minimum_return=1, female_iw_correction=1.0, male_iw_correction=1.0):
		iw_corrections = {0:male_iw_correction, 1:female_iw_correction}
		epsilons = np.array([ epsilon_f ])
		deltas   = np.array([   delta ])
		self._min_return = min_return
		self._max_return = max_return
		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return, iw_type_corrections=iw_corrections)
	def _get_return_ranges(self):
		return (self._min_return, self._max_return)

class TutoringSystemFemaleBootstrapSRL(TutoringSystemFemaleRLMixin(mode='bootstrap', scaling=2.0), SeldonianRLBase):
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

def TutoringSystemFemaleEmpRLMixin(mode='ttest', scaling=2.0):
	class TutoringSystemFemaleEmp:
		def _get_return_ranges(self):
			raise NotImplementedError('TutoringSystemFemaleEmp._get_return_ranges() not defined')
		def _preprocessor(self, data):
			T = data['T']
			R = data['R']
			# Split R and R_ref into two sets based T
			R1 = R[np.where(T==1)[0]] / self._iw_type_corrections[1] # Undo any type-based importance weighting
			# Compute type-dependent indicators for positive predictions
			return { 'R1' : R1 }
		def _make_rvs(self):
			r_min, r_max = self._get_return_ranges()
			# Variables for expected return given T under the candidate policy
			R1  = rvs.BoundedRealSampleSet(name='R1', lower=r_min, upper=r_max)
			ER1 = R1.expected_value('E[R|T=1]', mode=mode)
			# Constants
			e = rvs.constant(self.epsilons[0], name='e')
			r_ref1 = rvs.constant(self._ref_return_T1, name='Avg[R_ref|T=1]')
			# BQF and Constraint Objectives
			#   g1(theta) := (E[R_ref|T=1] - E[R|T=1]) - e
			BQF = rvs.sum( r_ref1, -ER1, name='BQF' )
			CO  = rvs.sum( BQF, -e, name='CO' )
			SCO = rvs.sum( BQF, -e, name='SCO', scaling=scaling )
			# Store the sample sets and variables
			self._scheck_rvs  = [ CO  ]
			self._ccheck_rvs  = [ SCO ]
			self._eval_rvs    = { 'bqf_0_mean' : BQF, 'co_0_mean' : CO } 
			# Add the sample sets and variables to the manager
			self._vm = rvs.VariableManager(self._preprocessor)
			self._vm.add_sample_set(R1)
			self._vm.add(ER1, BQF, CO, SCO)
	return TutoringSystemFemaleEmp

class TutoringSystemFemaleEmpSRL(TutoringSystemFemaleEmpRLMixin(mode='bootstrap', scaling=2.0), SeldonianRLBase):
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

class TutoringSystemFemaleBootstrapEmpSRL(TutoringSystemFemaleEmpRLMixin(mode='bootstrap', scaling=2.0), SeldonianRLBase):
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


# def CreditScoreRLMixin(mode='ttest', scaling=2.0):
# 	class CreditScore:
# 		def _is_positive(self):
# 			raise NotImplementedError('GroupFairness._is_positive undefined.')
# 		def _sample_distance(self, S, T):
# 			raise NotImplementedError('CreditScore._sample_distance undefined.')
# 		def _distribution_distance(self, P):
# 			raise NotImplementedError('CreditScore._distribution_distance undefined.')
# 		def _preprocessor(self, data):
# 			P = data['P'][:,0,:]
# 			A = np.argmax(P,1)
# 			T = data['T']
# 			S = data['S']
# 			# Split Y and Yp into two sets based T
# 			A0 = A[np.where(T==0)[0]].astype(int)
# 			A1 = A[np.where(T==1)[0]].astype(int)
# 			# Compute indicators for "positive" action predictions
# 			Pos0 = np.array([ self._is_positive(a) for a in A0 ]).astype(np.float)
# 			Pos1 = np.array([ self._is_positive(a) for a in A1 ]).astype(np.float)

# 			# Compute indicators for distance violations for FTA
# 			SampleDist = self._sample_distance(S, T)
# 			DistrDist  = self._distribution_distance(P)
# 			Viols = np.array(DistrDist > SampleDist).astype(int)
# 			# print(Viols) #'S', SampleDist, 'P', DistrDist)
# 			return { 'fta_viols':Viols, 'Pos0':Pos0, 'Pos1':Pos1 }
# 		def _make_rvs(self):
# 			# Fairness through awareness
# 			fta_viols = rvs.BoundedRealSampleSet(name='fta_viols', lower=0, upper=1)
# 			e0   = rvs.constant(self.epsilons[0], name='e0')
# 			BQF0 = fta_viols.expected_value(name='BQF0', mode=mode)
# 			CO0  = rvs.sum( BQF0, -e0, name='CO0' )
# 			SCO0 = rvs.sum( BQF0, -e0, name='SCO0', scaling=scaling )
			
# 			# Group Fairness
# 			Pos0   = rvs.BoundedRealSampleSet('Pos0', lower=0, upper=1)
# 			Pos1   = rvs.BoundedRealSampleSet('Pos1', lower=0, upper=1)
# 			PrPos0 = Pos0.expected_value('Pr(Pos|T=0)', mode=mode)
# 			PrPos1 = Pos1.expected_value('Pr(Pos|T=1)', mode=mode)
# 			e1   = rvs.constant(self.epsilons[1], name='e1')
# 			BQF1 = rvs.abs( PrPos0-PrPos1, name='BQF1' )
# 			CO1  = rvs.sum( BQF1, -e1, name='CO1' )
# 			SCO1 = rvs.sum( BQF1, -e1, name='SCO1', scaling=scaling )

# 			# Store the sample sets and variables
# 			self._scheck_rvs  = [ CO0 , CO1  ]
# 			self._ccheck_rvs  = [ SCO0, SCO1 ]
# 			self._eval_rvs    = { 'bqf_0_mean' : BQF0, 'co_0_mean' : CO0, 'bqf_1_mean' : BQF1, 'co_1_mean' : CO1 } 
# 			# Add the sample sets and variables to the manager
# 			self._vm = rvs.VariableManager(self._preprocessor)
# 			self._vm.add_sample_set(fta_viols, Pos0, Pos1)
# 			self._vm.add(BQF0, CO0, SCO0, BQF1, CO1, SCO1)
# 	return CreditScore

# class CreditScoreSRL(CreditScoreRLMixin(mode='ttest', scaling=2.0), SeldonianRLBase):
# 	def __init__(self, is_positivef, sample_distf, prob_distf, epsilon_fta=0.05, epsilon_gf=0.05, delta_fta=0.05, delta_gf=0.05, model_type='softmax', minimum_return=1):
# 		epsilons = np.array([ epsilon_fta, epsilon_gf ])
# 		deltas   = np.array([   delta_fta,   delta_gf ])
# 		self._sample_distf = sample_distf
# 		self._prob_distf = prob_distf
# 		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return)
# 		self._is_positive = is_positivef
# 	def _sample_distance(self, S, T):
# 		return self._sample_distf(S,T)
# 	def _distribution_distance(self, P):
# 		return self._prob_distf(P)

# class CreditScoreBootstrapSRL(CreditScoreRLMixin(mode='bootstrap', scaling=2.0), SeldonianRLBase):
# 	def __init__(self, is_positivef, sample_distf, prob_distf, epsilon_fta=0.05, epsilon_gf=0.05, delta_fta=0.05, delta_gf=0.05, model_type='softmax', minimum_return=1):
# 		epsilons = np.array([ epsilon_fta, epsilon_gf ])
# 		deltas   = np.array([   delta_fta,   delta_gf ])
# 		self._sample_distf = sample_distf
# 		self._prob_distf = prob_distf
# 		super().__init__(epsilons, deltas, model_type=model_type, minimum_return=minimum_return)
# 		self._is_positive = is_positivef
# 	def _sample_distance(self, S, T):
# 		return self._sample_distf(S,T)
# 	def _distribution_distance(self, P):
# 		return self._prob_distf(P)
