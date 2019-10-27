import numpy as np
from .rvs import *

def constant(value, **kwargs):
	return ConstantScalarRV(value, **kwargs)

def bounded_real(name, lower=-np.inf, upper=np.inf):
	return BoundedRealSampleSet(name, lower=-np.inf, upper=np.inf)

def binomial(name):
	return BinomialSampleSet(name)

def inv(rv, **kwargs):
	return InverseRV(rv, **kwargs)

def abs(rv, **kwargs):
	return AbsoluteValueRV(rv, **kwargs)

def sum(*_rvs, **kwargs):
	return SumRV(*_rvs, **kwargs)

def product(*_rvs, **kwargs):
	return ProductRV(*_rvs, **kwargs)

def ratio(*_rvs, **kwargs):
	return RatioRV(*_rvs, **kwargs)

def max(*_rvs, **kwargs):
	return MaximumRV(*_rvs, **kwargs)

def min(*_rvs, **kwargs):
	return MinimumRV(*_rvs, **kwargs)

def maxrec(rv, **kwargs):
	return MaxReciprocal(rv, **kwargs)	