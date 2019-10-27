class SMLAOptimizer:
	pass	
from .linear_shatter import LinearShatterBFOptimizer
from .cma import CMAESOptimizer
from .bfgs import BFGSOptimizer

OPTIMIZERS = { opt.cli_key():opt for opt in SMLAOptimizer.__subclasses__() }