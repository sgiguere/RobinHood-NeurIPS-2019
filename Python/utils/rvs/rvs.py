import numpy as np
import itertools 
from scipy.stats   import norm, chi, t
from scipy.special import erf, erfinv
from scipy.stats import beta

from time import time

# Sample Sets

class VectorRV:
	def __init__(self, name, scaling=1.0):
		self._name = name
		self._scaling = scaling
		self._value = None
	@property
	def name(self):
		return self._name
	def set_value(self, value):
		self._value = value
	def is_set(self):
		return not(self._value is None)
	def value(self):
		return self._value
	def __eq__(self, other):
		return 

class SampleSet:
	def __init__(self, name, scaling=1.0):
		self._name = name
		self._scaling = scaling
		self._value = None
	def set_value(self, value):
		self._value = value
	def is_set(self):
		return not(self._value is None)
	def value(self):
		return self._value
	# def __init__(self, name):
	# 	self._name  = name
	# 	super().__init(name)
	@property
	def name(self):
		return self._name
	def __getitem__(self, vrange, name=None):
		assert isinstance(vrange, IndexSampleSet)
		assert self.is_set() and vrange.is_set(), 'No data specified'
		name = ('%s[%s]' % (self.name, vrange.name)) if name is None else name
		sampleset = self.copy(name)
		sampleset.set_value(self._value[vrange._value])
		return sampleset
	def copy(self, name):
		name = (self.name + ' copy') if name is None else name
		sampleset = type(self)(name)
		sampleset.set_value(self._value)
		return sampleset

class BoundedRealSampleSet(SampleSet):
	def __init__(self, name, lower=-np.inf, upper=np.inf):
		super().__init__(name)
		self._lower = lower
		self._upper = upper
		self._range = (upper-lower) if not(np.isinf(lower) or np.isinf(upper)) else np.inf
	def expected_value(self, name=None, mode='trivial', scaling=1.0):
		name = 'E[%s]'%self.name if name is None else name
		return BoundedRealExpectedValueRV(name, self, mode=mode, scaling=scaling)
	def clamp(self, X):
		return np.maximum(self._lower, np.minimum(self._upper, X))
	def copy(self, name=None):
		name = (self.name + ' copy') if name is None else name
		sampleset = BoundedRealSampleSet(name, lower=self.lower, upper=self.upper)
		sampleset.set_value(self._value)
		return sampleset

class BinomialSampleSet(SampleSet):
	def __init__(self, name):
		super().__init__(name)
	def proportion(self, name=None, mode='trivial', scaling=1.0):
		name = 'Pr[%s=1]' % self.name if name is None else name
		return BinomialProportionRV(name, self, mode=mode, scaling=scaling)

class IndexSampleSet(SampleSet):
	def __init__(self, name):
		super().__init__(name)

################################################
#             ScalarRVs Interfaces             #
################################################

# A ScalarRV is a single quantity that can be estimated and bounded

# A ConstantScalarRV is a ScalarRV that defines a constant

# An ObservedScalarRV is a ScalarRV that is associated with a SampleSet and
#   constructs estimates/bounds based on statistics on the SampleSet's contents
#   ObservedScalarRVs should not be instantiated directly

# A FunctionScalarRVs is a ScalarRV that represents a function applied to one
#   or more other ScalarRVs, and computes estimates/bounds recursively
#   based on the estimates/bounds of the consituent ScalarRVs
#   FunctionScalarRVs should not be instantiated directly

class ScalarRV:
	def __init__(self, name, scaling=1.0):
		self._name = name
		self._scaling = scaling
	@property
	def name(self):
		return self._name
	def upper(self, delta, split_delta=False, n_scale=1.0):
		return self.bound(delta, side='upper', split_delta=split_delta, n_scale=n_scale)
	def lower(self, delta, split_delta=False, n_scale=1.0):
		return self.bound(delta, side='lower', split_delta=split_delta, n_scale=n_scale)
	def bound(self, delta, side='both', split_delta=False, n_scale=1):
		if split_delta:
			n_bounds = len(self.get_observed())
			delta = delta / n_bounds
		l,u = self._bound(delta, n_scale=n_scale)
		# rescale the bound if needed
		if not(any(np.isinf([l,u])) or any(np.isnan([l,u]))):
			mod = 0.5*(u - l)*(self._scaling-1)
			l,u = l-mod, u+mod
		if side == 'upper':
			return u
		elif side == 'lower':
			return l
		return (l,u)
	def value(self):
		raise NotImplementedError()
	def _bound(self, delta, n_scale=1.0):
		raise NotImplementedError()
	def __add__(self, other):
		return SumRV(self, other)
	def __div__(self, other):
		return RatioRV(self, other)
	def __truediv__(self, other):
		return self.__div__(other)
	def __mul__(self, other):
		return ProductRV(self, other)
	def __neg__(self):
		return NegativeRV(self)
	def __sub__(self, other):
		return SumRV(self, -other)
	def get_observed(self):
		return np.array([])
	def _recip_value(self):
		return 1.0/self.value()



class ObservedScalarRV(ScalarRV):
	''' Parent class to represent a quantity estimated from a SampleSet. '''
	def __init__(self, name, samples, scaling=1.0):
		super().__init__(name, scaling=scaling)
		self._samples = samples
	def _get_samples(self):
		assert self._samples.is_set(), 'ScalarRV: samples not set.'
		return self._samples.value()
	def get_observed(self):
		return np.array([ self.name ])

class FunctionScalarRV(ScalarRV):
	''' Parent class to represent a scalar-valued function of ScalarRVs. '''
	def __init__(self, name, rvs, scaling=1.0):
		msg = 'can only define compound RVs from other scalar RVs.'
		assert all([ isinstance(rv,ScalarRV) for rv in rvs ]), msg
		super().__init__(name, scaling=scaling)
		self._rvs = rvs
	def get_observed(self):
		return np.unique(np.concatenate([ rv.get_observed() for rv in self._rvs ]))


################################################
#             ObservedScalarRVs                #
################################################

class ConstantScalarRV(ScalarRV):
	''' Concrete class to represent constants. ''' 
	def __init__(self, value, name=None):
		name = str(value) if name is None else name
		super().__init__(name)
		self._value = value
	def value(self):
		return self._value
	def _bound(self, delta, n_scale=1.0):
		return (self._value, self._value)
	def _sign(self):
		return np.sign(self.value())
	def __repr__(self):
		return self.name


################################################
#             ObservedScalarRVs                #
################################################

class BoundedRealExpectedValueRV(ObservedScalarRV):
	def __init__(self, name, samples, mode='trivial', scaling=1.0):
		assert isinstance(samples, BoundedRealSampleSet), ('Cannot create BoundedRealExpectedValueRV from type \'%s\'' % samples.__class__.__name__)
		super().__init__(name, samples, scaling=scaling)
		self.set_mode(mode)
	def set_mode(self, mode):
		self._mode = mode
	def value(self):
		S = self._get_samples()
		return np.mean(S) if len(S) > 0 else np.nan
	def _sign(self):
		return np.sign(self.value())
	def _bound(self, delta, n_scale=1.0):
		if self._mode == 'trivial':
			return (self._samples._lower, self._samples._upper)
	
		# Get statistics of the samples
		S = self._get_samples()
		if len(S) == 0 or n_scale == 0:
			return (self._samples._lower, self._samples._upper)
		
		n, mean = len(S) * n_scale, np.mean(S)
		S_range = self._samples._range
		# Compute the bound
		if self._mode == 'hoeffding':
			offset = S_range * np.sqrt(0.5*np.log(2/delta)/n)
			l = self._samples.clamp(mean-offset)
			u = self._samples.clamp(mean+offset)
		elif self._mode == 'ttest': 
			# Note: uses incorrect assumption of normality
			if len(S) == 1:
				return (self._samples._lower, self._samples._upper)
			std = np.std(S,ddof=1)
			# If the standard deviation is zero, assume we're binomial and apply the rule of three
			if np.isclose(std, 0.0):
				if np.isclose(mean, 0.0):
					return (0, 3.0/n)
				elif np.isclose(mean, 1.0):
					return (1-3.0/n, 1.0)
			offset = std * t.ppf(1-delta/2,n-1) / np.sqrt(n-1)
			l = self._samples.clamp(mean-offset)
			u = self._samples.clamp(mean+offset)
		elif self._mode == 'bootstrap':
			n_resamples = 1000
			Z = (np.random.multinomial(S.shape[0], np.ones(S.shape[0])/S.shape[0], n_resamples) * S[None,:]).mean(1)
			l, u = np.percentile(Z, (100*delta/2, 100*(1-delta/2)))
		return (l,u)
	def __repr__(self):
		return self.name

class BinomialProportionRV(ObservedScalarRV):
	def __init__(self, name, samples, mode='trivial', scaling=1.0):
		assert isinstance(samples, BinomialSampleSet), ('Cannot create BinomialProportionRV from type \'%s\'' % samples.__class__.__name__)
		super().__init__(name, samples, scaling=scaling)
		self.set_mode(mode)
	def set_mode(self, mode):
		self._mode = mode
	def value(self):
		S = self._get_samples()
		return np.mean(S) if len(S) > 0 else np.nan
	def _sign(self):
		return np.sign(self.value())
	def _bound(self, delta, n_scale=1.0):
		if self._mode == 'trivial':
			l = 0
			u = 1
		else:
			# Get statistics of the samples
			S = self._get_samples()
			n, ns, p = len(S)*n_scale, np.sum(S), np.mean(S)
			if n == 0:
				return (0, 1)
			# Compute the bound
			if self._mode == 'jeffery':
				l = beta.ppf(  delta/2, ns+0.5, n-ns+0.5) if (ns>0) else 0 
				u = beta.ppf(1-delta/2, ns+0.5, n-ns+0.5) if (ns<n) else 1 
			elif self._mode == 'wilson':
				z = norm.ppf(1-delta/2)
				v = z**2 - (1/n) + 4*n*p*(1-p) + (4*p-2)
				den = 2 * (n + z**2)
				i = (z*np.sqrt(v) + 1)
				c = (2*n*p + z**2)
				l = max(0, (c-i) / den)
				u = min(1, (c+i) / den)
			elif self._mode == 'learned-miller':
				S = np.sort(S)
				D = np.diff(S.tolist()+[1.0]) 
				U = np.random.random((5000,n))
				U = np.sort(U, axis=1)
				M = 1 - (U*D[None]).sum(1)
				M = np.sort(M)
				i_ub = np.ceil((1-delta)*5000).astype(int)
				u = M[i_ub]
			elif self._mode == 'bootstrap':
				n_resamples = 1000
				Z = (np.random.multinomial(S.shape[0], np.ones(S.shape[0])/S.shape[0], n_resamples) * S[None,:]).mean(1)
				l, u = np.percentile(Z, (100*delta/2, 100*(1-delta/2)))
			else:
				raise Exception('Unknown mode: %s' % self._mode)
		return (l,u)
	def __repr__(self):
		return self.name


################################################
#                  Compound RVs                #
################################################

#    Unary:

class UnaryFunctionScalarRV(FunctionScalarRV):
	def __init__(self, name, rv, scaling=1.0):
		super().__init__(name, [rv], scaling=scaling)
	@property
	def _rv(self):
		return self._rvs[0]

class NegativeRV(UnaryFunctionScalarRV):
	def __init__(self, rv, name=None, scaling=1.0):
		name = ('-%s' % rv.name) if (name is None) else name
		super().__init__(name, rv, scaling=scaling)
	def value(self):
		return -self._rv.value()
	def _bound(self, delta, n_scale=1.0):
		l, u = self._rv.bound(delta, n_scale=n_scale)
		if np.isnan(l) or np.isnan(u):
			return np.nan, np.nan
		return (-u, -l)
	def __repr__(self):
		return '-%s' % repr(self._rv)
	def _recip_value(self):
		return -self._rv._recip_value()
	def _sign(self):
		return -self._rv._sign()

class InverseRV(UnaryFunctionScalarRV):
	def __init__(self, rv, name=None, scaling=1.0):
		name = ('1/%s' % rv.name) if (name is None) else name
		super().__init__(name, rv, scaling=scaling)
	def value(self):
		v = self._rv.value()
		return 1.0/v if not(v==0) else np.nan
	def _bound(self, delta, n_scale=1.0):
		l, u = self._rv.bound(delta, n_scale=n_scale)
		ll, uu = l, u
		if (l==0) and (u==0):
			l, u = np.nan, np.nan
		elif (l==0):
			l, u = 1/u, np.inf
		elif (u==0):
			l, u = -np.inf, 1/l
		elif np.sign(l) == np.sign(u):
			l, u = 1/u, 1/l
		else:
			l, u = np.nan, np.nan
		return (l, u)
	def _recip_value(self):
		return self._rv.value()
	def __repr__(self):
		return '1/%s' % repr(self._rv)

class AbsoluteValueRV(UnaryFunctionScalarRV):
	def __init__(self, rv, name=None, scaling=1.0):
		name = ('|%s|' % rv.name) if (name is None) else name
		super().__init__(name, rv, scaling=scaling)
	def value(self):
		return np.abs(self._rv.value())
	def _bound(self, delta, n_scale=1.0):
		l, u = self._rv.bound(delta, n_scale=n_scale)
		if np.isnan(l) or np.isnan(u):
			return l, u
		if np.sign(l)*np.sign(u) >= 0:
			l, u = sorted([ np.abs(l), np.abs(u) ])
		else:
			l, u = 0, max(np.abs(l), np.abs(u))
		return l, u
	def __repr__(self):
		return '|%s|' % repr(self._rv)


class TruncatedRV(UnaryFunctionScalarRV):
	def __init__(self, rv, lower=-np.inf, upper=np.inf, name=None, scaling=1.0):
		msg = 'TruncatedRV.__init__(): upper must be at least as large as lower.'
		assert lower <= upper, msg
		name = 'Trunc(%s,[%f,%f])' % (rv.name, lower, upper)
		super().__init__(name, rv, scaling=scaling)
		self._lower = lower
		self._upper = upper
	def clamp(self, X):
		return np.maximum(self._lower, np.minimum(self._upper, X))
	def value(self):
		return self.clamp(self._rv.value())
	def _bound(self, delta, n_scale=1.0):
		l, u = self._rv.bound(delta, n_scale=n_scale)
		if np.isnan(l) or np.isnan(u):
			return l, u
		l, u = self.clamp(l), self.clamp(u)
		return l, u
	def __repr__(self):
		return 'Trunc[%f,%f](%r)' %(self._lower,self._upper,self._rv)


#    Binary:

class RatioRV(FunctionScalarRV):
	def __init__(self, numerator, denominator, name=None, scaling=1.0):
		name = ('%s/%s' % (numerator.name,denominator.name)) if name is None else name
		ratio = ProductRV(numerator, InverseRV(denominator))
		super().__init__(name, [ratio], scaling=scaling)
		self._numerator   = numerator
		self._denominator = denominator
	@property
	def _rv(self):
		return self._rvs[0]
	def value(self):
		if self._numerator.value() == 0 and self._denominator.value() == 0:
			return 1.0
		return self._rv.value()
	def _recip_value(self):
		if self._numerator.value() == 0 and self._denominator.value() == 0:
			return 1.0
		return self._rv._recip_value()
	def _bound(self, delta, n_scale=1.0):
		return self._rv.bound(delta, n_scale=n_scale)
	def _sign(self):
		nv = self._numerator.value()
		dv = self._denominator.value()
		ns = 1 if (nv==0) else self._numerator._sign()
		ds = 1 if (dv==0) else self._denominator._sign()
		return ns * ds
	def __repr__(self):
		return repr(self._rv).replace('*1/','/') # '%r/%r' % (self._numerator,self._denominator)


#    Multi-nary:

class SumRV(FunctionScalarRV):
	def __init__(self, *rvs, name=None, scaling=1.0):
		name = '+'.join([ rv.name for rv in rvs ]) if name is None else name
		super().__init__(name, rvs, scaling=scaling)
	def value(self):
		if len(self._rvs) == 0:
			return 0
		return np.sum([ rv.value() for rv in self._rvs ])
	def _bound(self, delta, n_scale=1.0):
		ls, us = zip(*[ rv.bound(delta, n_scale=n_scale) for rv in self._rvs ])
		if any(np.isnan(ls)) or any(np.isnan(us)):
			return np.nan, np.nan
		return sum(ls), sum(us)
	def _sign(self):
		return np.sign(self.value())
	def __repr__(self):
		return '+'.join(map(repr,self._rvs)).replace('+-','-')


class ProductRV(FunctionScalarRV):
	def __init__(self, *rvs, name=None, scaling=1.0):
		name = '*'.join([ rv.name for rv in rvs ]) if name is None else name
		super().__init__(name, rvs, scaling=scaling)
	def value(self):
		if len(self._rvs) == 0:
			return 1
		value = 1
		for rv in self._rvs:
			v = rv.value()
			if np.isnan(v):
				return v
			value = value * v
		return value
	def _bound(self, delta, n_scale=1.0):
		if len(self._rvs) == 0:
			return (1, 1)
		rng = self._rvs[0].bound(delta)
		if any(np.isnan(rng)):
			return (np.nan, np.nan)
		for rv in self._rvs[1:]:
			_rng  = rv.bound(delta, n_scale=n_scale)

			if any(np.isnan(_rng)):
				return (np.nan, np.nan)

			vmin, vmax = np.inf, -np.inf
			for v0 in rng:
				for v1 in _rng:
					if np.isnan(v0) or np.isnan(v1):
						return (np.nan, np.nan)
					if np.isinf(v0) and (v1 == 0):
						vmin = min(vmin, 0)
						vmax = max(vmax, 0)
					elif np.isinf(v1) and (v0 == 0):
						vmin = min(vmin, 0)
						vmax = max(vmax, 0)
					else:
						vmin = min(vmin, v0*v1)
						vmax = max(vmax, v0*v1)
			rng = (vmin, vmax)
		return rng
	def _recip_value(self):
		if len(self._rvs) == 0:
			return 1
		value = 1
		for rv in self._rvs:
			v = rv._recip_value()
			if np.isnan(v):
				return v
			value = value * v
		return value
	def _sign(self):
		if len(self._rvs) == 0:
			return 1
		value = 1
		for rv in self._rvs:
			v = rv._sign()
			if np.isnan(v):
				return v
			value = value * v
		return value

	def __repr__(self):
		return '*'.join(map(repr,self._rvs))

class MaxReciprocal(UnaryFunctionScalarRV):
	def __init__(self, rv, name=None, scaling=1.0):
		if name is None:
			name = 'Max[%s, 1/%s]' % (rv.name, rv.name)
		super().__init__(name, rv, scaling=scaling)
	def value(self):
		v = self._rv.value()
		if np.isnan(v) or (v==0):
			# BAD: I'm going to assume v is a fraction, so it's only nan if the denominator was 0
			# In that case, we assume calculate the numerator and return lim_{x->0} 1/x from that direction
			return self._sign() * np.inf
		return max(v, 1/v)
	def _sign(self):
		return self._rv._sign()
	def _bound(self, delta, n_scale=1.0):
		l, u = self._rv.bound(delta, n_scale=n_scale)
		li = 1/l if not(l==0) else np.inf
		ui = 1/u if not(u==0) else np.inf

		if any(np.isnan([l,u])):
			return (np.nan, np.nan)

		elif l >= 1 and u >= 1:
			return (l, u)
		elif l >= 0 and u >= 1:
			return (1, max(u,li))
		elif l >= -1 and u >= 1:
			return (l, np.inf)
		elif l < -1 and u >= 1:
			return (-1, np.inf)

		elif l >= 0 and u >= 0:
			return (ui, li)
		elif l >= -1 and u >= 0:
			return (l, np.inf)
		elif l < -1 and u >= 0:
			return (-1, np.inf)

		elif l >= -1 and u >= -1:
			return (l, u)
		elif l < -1 and u >= -1:
			return (-1, max(li, u))

		elif l < -1 and u < -1:
			return (ui, li)






class MaximumRV(FunctionScalarRV):
	def __init__(self, *rvs, name=None, scaling=1.0):
		if name is None:
			name = 'Max[%s]' % (', '.join([ rv.name for rv in rvs ]))
		super().__init__(name, rvs, scaling=scaling)
	def value(self):
		if len(self._rvs) == 0:
			return np.nan
		vmax = None
		for rv in self._rvs:
			v = rv.value()
			if np.isnan(v):
				return np.nan
			vmax = max(vmax,v) if not(vmax is None) else v
		return vmax
	def _bound(self, delta, n_scale=1.0):
		if len(self._rvs) == 0:
			return (np.nan, np.nan)
		lmax, umax = None, None
		for rv in self._rvs:
			l, u = rv.bound(delta, n_scale=n_scale)
			if any(np.isnan([l,u])):
				return (np.nan, np.nan)
			lmax = max(lmax,l) if not(lmax is None) else l
			umax = max(umax,u) if not(umax is None) else u
		return (lmax, umax)
	def __repr__(self):
		return 'Max{%s}' % ', '.join(map(repr,self._rvs))



class MinimumRV(FunctionScalarRV):
	def __init__(self, *rvs, name=None, scaling=1.0):
		if name is None:
			name = 'Min[%s]' % (', '.join([ rv.name for rv in rvs ]))
		super().__init__(name, rvs, scaling=scaling)
	def value(self):
		if len(self._rvs) == 0:
			return np.nan
		vmax = None
		for rv in self._rvs:
			v = rv.value()
			if np.isnan(v):
				return np.nan
			vmax = max(vmax,v) if not(vmax is None) else v
		return vmax
	def _bound(self, delta, n_scale=1.0):
		if len(self._rvs) == 0:
			return (np.nan, np.nan)
		lmin, umin = None, None
		for rv in self._rvs:
			l, u = rv.bound(delta, n_scale=n_scale)
			if any(np.isnan([l,u])):
				return (np.nan, np.nan)
			lmin = min(lmin,l) if not(lmin is None) else l
			umin = min(umin,u) if not(umin is None) else u
		return (lmin, umin)
	def __repr__(self):
		return 'Min{%s}' % ', '.join(map(repr,self._rvs))







class VariableManager:
	def __init__(self, preprocessor=None):
		self._sample_sets = {}
		self._context = {}
		self._preprocessor = preprocessor
	def set_preprocessor(self, preprocessor):
		self._preprocessor = preprocessor
	def add_sample_set(self, *sample_sets):
		for ss in sample_sets:
			self._add_sample_set(ss)
	def _add_sample_set(self, samples):
		msg = 'input is not a sample set.'
		assert isinstance(samples, SampleSet), msg
		name = samples.name
		self._sample_sets[name] = samples
	def set_data(self, data):
		if self._preprocessor is None:
			processed = data
		else:
			processed = self._preprocessor(data)
		for k, rv in self._sample_sets.items():
			rv.set_value(processed[k])
	def add(self, *rvs):
		for rv in rvs:
			self._add(rv)
	def _add(self, rv):
		name = rv.name
		self._context[name] = rv
	def get(self, name):
		return self._context[name]
	def value(self, name):
		return self.get(name).value()
	def upper(self, name, delta, n_scale=1.0):
		return self.get(name).upper(delta, n_scale=n_scale)
	def lower(self, name, delta, n_scale=1.0):
		return self.get(name).lower(delta, n_scale=n_scale)
	def bound(self, name, delta, n_scale=1.0):
		return self.get(name).bound(delta, n_scale=n_scale)
	def list(self):
		print(self)
	def __repr__(self):
		s = 'SampleSets:\n'
		for ss in self._sample_sets.keys():
			s += '  %s\n' % ss
		s += 'Variables:\n'
		for ss in self._context.keys():
			s += '  %s\n' % ss
		return s





# Testng and verification

if __name__ == '__main__':

	# Create sample sets
	R1 = BoundedRealSampleSet('R1',  0, 1)
	R2 = BoundedRealSampleSet('R2', -1, 1)
	B1 = BinomialSampleSet('B1')
	B2 = BinomialSampleSet('B2')

	# Create basic random variables to estimate
	C1  = ConstantScalarRV( 2,   name='c1')
	C2  = ConstantScalarRV(-3.2, name='c1')
	E10 = R1.expected_value(mode='trivial')
	E11 = R1.expected_value(mode='ttest')
	E12 = R1.expected_value(mode='hoeffding')
	E20 = R2.expected_value(mode='trivial')
	E21 = R2.expected_value(mode='ttest')
	E22 = R2.expected_value(mode='hoeffding')
	P10 = B1.proportion('P10', mode='trivial')
	P11 = B1.proportion('P11', mode='jeffery')
	P12 = B1.proportion('P12', mode='wilson')
	P20 = B2.proportion('P20', mode='trivial')
	P21 = B2.proportion('P21', mode='jeffery')
	P22 = B2.proportion('P22', mode='wilson')


	# Create synthetic data
	p1 = np.random.random()
	p2 = np.random.random()
	ev1 = 0.5
	ev2 = 0.0
	data = {
		'R1':np.random.random(30),
		'R2':np.random.random(300)*2-1,
		'B1':np.random.choice([0,1], size=20 , p=[1-p1,p1]),
		'B2':np.random.choice([0,1], size=200, p=[1-p2,p2])
	}

	# Condition on the observable nodes
	R1.set_value(data['R1'])
	R2.set_value(data['R2'])
	B1.set_value(data['B1'])
	B2.set_value(data['B2'])

	print('\n' + ('-'*80))
	print('   Basic Random Variables')
	print('-'*80, '\n')

	# Print bounds
	print('  BoundedRead expected value: (true=%f)' % (ev1))
	print('     estimate =', E11.value())
	print('     E10 (trivial)   :', E10.bound(0.95))
	print('     E11 (ttest)     :', E11.bound(0.95))
	print('     E12 (hoeffding) :', E12.bound(0.95))
	print()
	print('  BoundedRead expected value: (true=%f)' % (ev2))
	print('     estimate =', E21.value())
	print('     E20 (trivial)   :', E20.bound(0.95))
	print('     E21 (ttest)     :', E21.bound(0.95))
	print('     E22 (hoeffding) :', E22.bound(0.95))
	print()
	print('  Binomial proportions: (true=%f)' % p1)
	print('     estimate =', P11.value())
	print('     P10 (trivial) :', P10.bound(0.95))
	print('     P11 (jeffery) :', P11.bound(0.95))
	print('     P12 (wilson)  :', P12.bound(0.95))
	print()
	print('  Binomial proportions: (true=%f)' % p2)
	print('     estimate =', P21.value())
	print('     P20 (trivial) :', P20.bound(0.95))
	print('     P21 (jeffery) :', P21.bound(0.95))
	print('     P22 (wilson)  :', P22.bound(0.95))
	print()

	print('\n' + ('-'*80))
	print('   Functional Random Variables')
	print('-'*80, '\n')

	S10 = SumRV(E10, E20)
	S11 = SumRV(E11, E21)
	S12 = SumRV(E12, E22)
	S20 = SumRV(E10, E10, E20)
	S21 = SumRV(E11, E11, E21)
	S22 = SumRV(E12, E12, E22)
	S30 = SumRV(E10, P10)
	S31 = SumRV(E11, P11)
	S32 = SumRV(E12, P12)

	# Print bounds
	print('  E[Real1] + E[Real2]: (true=%s)' % (ev1+ev2))
	print('     estimate =', S11.value())
	print('     S10 (trivial)   :', S10.bound(0.95))
	print('     S11 (ttest)     :', S11.bound(0.95))
	print('     S12 (hoeffding) :', S12.bound(0.95))
	print()
	print('  E[Real1] + E[Real1] + E[Real2]: (true=%s)' % (ev1+ev1+ev2))
	print('     estimate =', S21.value())
	print('     S20 (trivial)   :', S20.bound(0.95))
	print('     S21 (ttest)     :', S21.bound(0.95))
	print('     S22 (hoeffding) :', S22.bound(0.95))
	print()
	print('  E[Real1] + p[Binom1]: (true=%f)' % (ev1+p1))
	print('     estimate =', S31.value())
	print('     S30 (trivial+trivial)   :', S30.bound(0.95))
	print('     S31 (ttest+jeffery)     :', S31.bound(0.95))
	print('     S32 (hoeffding+wilson)  :', S32.bound(0.95))
	print()

	print('-'*80, '\n')

	M10 = ProductRV(E10, E20)
	M11 = ProductRV(E11, E21)
	M12 = ProductRV(E12, E22)
	M20 = ProductRV(E10, P10)
	M21 = ProductRV(E11, P11)
	M22 = ProductRV(E12, P12)
	M30 = ProductRV(E10, C1)
	M31 = ProductRV(E11, C1)
	M32 = ProductRV(E12, C1)

	# Print bounds
	print('  E[Real1] * E[Real2]: (true=%s)' % (ev1*ev2))
	print('     estimate =', M11.value())
	print('     M10 (trivial)   :', M10.bound(0.95))
	print('     M11 (ttest)     :', M11.bound(0.95))
	print('     M12 (hoeffding) :', M12.bound(0.95))
	print()
	print('  E[Real1] * p[Binom1]: (true=%f)' % (ev1*p1))
	print('     estimate =', M31.value())
	print('     M20 (trivial+trivial)   :', M20.bound(0.95))
	print('     M21 (ttest+jeffery)     :', M21.bound(0.95))
	print('     M22 (hoeffding+wilson)  :', M22.bound(0.95))
	print()
	print('  E[Real1] * C1: (true=%f)' % (ev1*C1._value))
	print('     estimate =', M31.value())
	print('     M30 (trivial)   :', M30.bound(0.95))
	print('     M31 (ttest)     :', M31.bound(0.95))
	print('     M32 (hoeffding) :', M32.bound(0.95))
	print()

	print('-'*80, '\n')

	N10 = NegativeRV(E10)
	N11 = NegativeRV(E11)
	N12 = NegativeRV(E12)
	AN10 = AbsoluteValueRV(N10)
	AN11 = AbsoluteValueRV(N11)
	AN12 = AbsoluteValueRV(N12)

	print('  E[Real1]: (true=%f)' % (ev1))
	print('     estimate =', E11.value())
	print('     E10 (trivial)   :', E10.bound(0.95))
	print('     E11 (ttest)     :', E11.bound(0.95))
	print('     E12 (hoeffding) :', E12.bound(0.95))
	print()
	print('  -E[Real1]: (true=%f)' % (-ev1))
	print('     estimate =', N11.value())
	print('     N10 (trivial)   :', N10.bound(0.95))
	print('     N11 (ttest)     :', N11.bound(0.95))
	print('     N12 (hoeffding) :', N12.bound(0.95))
	print()
	print('  |-E[Real1]|: (true=%f)' % (np.abs(-ev1)))
	print('     estimate =', AN11.value())
	print('     AN10 (trivial)   :', AN10.bound(0.95))
	print('     AN11 (ttest)     :', AN11.bound(0.95))
	print('     AN12 (hoeffding) :', AN12.bound(0.95))
	print()

	print('-'*80, '\n')

	inverse10 = InverseRV(E10)
	inverse11 = InverseRV(E11)
	inverse12 = InverseRV(E12)
	inverse20 = InverseRV(E20)
	inverse21 = InverseRV(E21)
	inverse22 = InverseRV(E22)
	ratio10 = RatioRV(E20, E10)
	ratio11 = RatioRV(E21, E11)
	ratio12 = RatioRV(E22, E12)
	ratio20 = RatioRV(E10, E20)
	ratio21 = RatioRV(E11, E21)
	ratio22 = RatioRV(E12, E22)

	print('  E[Real1]: (true=%f)' % (ev1))
	print('     estimate =', E11.value())
	print('     (trivial)   :', E10.bound(0.95))
	print('     (ttest)     :', E11.bound(0.95))
	print('     (hoeffding) :', E12.bound(0.95))
	print()
	print('  E[Real2]: (true=%f)' % (ev2))
	print('     estimate =', E21.value())
	print('     (trivial)   :', E20.bound(0.95))
	print('     (ttest)     :', E21.bound(0.95))
	print('     (hoeffding) :', E22.bound(0.95))
	print()
	print('  1/E[Real1]: (true=%f)' % (1/ev1 if not(ev1==0) else np.nan))
	print('     estimate =', inverse11.value())
	print('     (trivial)   :', inverse10.bound(0.95))
	print('     (ttest)     :', inverse11.bound(0.95))
	print('     (hoeffding) :', inverse12.bound(0.95))
	print()
	print('  1/E[Real2]: (true=%f)' % (1/ev2 if not(ev2==0) else np.nan))
	print('     estimate =', inverse21.value())
	print('     (trivial)   :', inverse20.bound(0.95))
	print('     (ttest)     :', inverse21.bound(0.95))
	print('     (hoeffding) :', inverse22.bound(0.95))
	print()
	print('  E[Real2]/E[Real1]: (true=%f)' % (ev2/ev1 if not(ev1==0) else np.nan))
	print('     estimate =', ratio10.value())
	print('     (trivial)   :', ratio10.bound(0.95))
	print('     (ttest)     :', ratio11.bound(0.95))
	print('     (hoeffding) :', ratio12.bound(0.95))
	print()
	print('  E[Real1]/E[Real2]: (true=%f)' % (ev1/ev2 if not(ev2==0) else np.nan))
	print('     estimate =', ratio20.value())
	print('     (trivial)   :', ratio20.bound(0.95))
	print('     (ttest)     :', ratio21.bound(0.95))
	print('     (hoeffding) :', ratio22.bound(0.95))
	print()

	print('-'*80, '\n')

	T20 = TruncatedRV(E20, lower=-0.001, upper=0.001)
	T21 = TruncatedRV(E21, lower=-0.001, upper=0.001)
	T22 = TruncatedRV(E22, lower=-0.001, upper=0.001)

	print('  E[Real2]: (true=%f)' % (ev2))
	print('     estimate =', E21.value())
	print('     (trivial)   :', E20.bound(0.95))
	print('     (ttest)     :', E21.bound(0.95))
	print('     (hoeffding) :', E22.bound(0.95))
	print()
	print('  T(E[Real2], [-0.001,0.001]): (true=%f)' % (ev2))
	print('     estimate =', T21.value())
	print('     (trivial)   :', T20.bound(0.95))
	print('     (ttest)     :', T21.bound(0.95))
	print('     (hoeffding) :', T22.bound(0.95))
	print()


	# Create sample sets

	B1 = BinomialSampleSet('I(X==1|T==0)')
	B2 = BinomialSampleSet('I(X==1|T==1)')

	# Create a manager and add the sample sets

	VM = VariableManager()
	VM.add_sample_set(B1, B2)

	# Define some variables and add them
	P1  = B1.proportion('FPR0', mode='jeffery')
	P2  = B2.proportion('FPR1', mode='jeffery')
	SP1 = B1.proportion('sFPR0', mode='jeffery', scaling=2.0)
	SP2 = B2.proportion('sFPR1', mode='jeffery', scaling=2.0)
	M1 = MaximumRV(RatioRV(P1,P2),  RatioRV(P2,P1),  name='safety_check')
	M2 = MaximumRV(RatioRV(SP1,P2), RatioRV(SP2,P1), name='candidate_check')
	VM.add(M1, M2)


	# Create and add a preprocessor
	def preprocessor(data):
		return { 'I(X==1|T==0)':data[0], 'I(X==1|T==1)':data[1] }
	VM.set_preprocessor(preprocessor)

	# Create and add some synthetic data
	p1 = np.random.random()
	p2 = np.random.random()
	data = np.array([ np.random.choice([0,1], size=300, p=[1-p1,p1]),
		   			  np.random.choice([0,1], size=200, p=[1-p2,p2]) ])
	VM.set_data(data)

	print(VM.upper('safety_check', 0.95))
	print(VM.upper('candidate_check', 0.95))


	data2 = np.array([ np.random.choice([0,1], size=300, p=[1-p1,p1]),
		   			   np.random.choice([0,1], size=200, p=[1-p2,p2]) ])
	VM.set_data(data2)

	print(VM.upper('safety_check', 0.95))
	print(VM.upper('candidate_check', 0.95))


	aP1  = B1.proportion('FPR0', mode='jeffery')
	aP2  = B2.proportion('FPR1', mode='jeffery')
	aSP1 = B1.proportion('sFPR0', mode='jeffery', scaling=2.0)
	aSP2 = B2.proportion('sFPR1', mode='jeffery', scaling=2.0)
	aM1 = MaximumRV( P1/P2,  P2/P1, name='safety_check')
	aM2 = MaximumRV(SP1/P2, SP2/P1, name='candidate_check')
