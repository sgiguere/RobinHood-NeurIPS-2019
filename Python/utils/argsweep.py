import argparse
import enum
import functools
import numpy as np




################################
#   CLI for Parameter Sweeps   #
################################

class ArgumentSweeper(argparse.ArgumentParser):
	@functools.wraps(argparse.ArgumentParser.__init__)
	def __init__(self, *args, **kwargs):
		self._sweep_argnames = []
		return super().__init__(*args, **kwargs)
	def __enter__(self):
		@functools.wraps(add_sweepable_argument)
		def _add_sweepable_argument(*args, _argname_list=self._sweep_argnames, **kwargs):
			argname = add_sweepable_argument(*args, **kwargs)
			_argname_list.append(argname.dest)
			return argname
		argparse._ActionsContainer.add_sweepable_argument = _add_sweepable_argument
		return self
	def __exit__(self, type, value, tb):
		del argparse._ActionsContainer.add_sweepable_argument
	def get_sweep_argnames(self):
		return self._sweep_argnames.copy()

class SweepAction(argparse.Action):
	def __init__(self, option_strings, dest, nargs=None, type=None, **kwargs):
		super().__init__(option_strings, dest, type=str, nargs=nargs, **kwargs)
		self._sweep_type = type
	def __call__(self, parser, namespace, values, option_string=None):
		values = [values] if (self.nargs is None) else values
		values = SweepAction._parse(values, self._sweep_type)
		setattr(namespace, self.dest, values)
	@staticmethod
	def _parse(values, typ):
		if typ == str:
			return values
		if typ in [int, float]:
			values = values if isinstance(values, (list,np.ndarray)) else [values]
			output = []
			for val in values:
				if len(val.split(':')) == 3:  # numerical range
					s,f,i = ( typ(v) for v in val.split(':') )
					output.extend(np.arange(s,f,i,typ))
				else:
					output.append(typ(val))
			return output
		if typ == 'bool':
			def _bool(s):
				if s.lower() == 'true':
					return True
				if s.lower() == 'false':
					return False
				raise ValueError('Unable to convert value to bool: %r' % s)
			return [ _bool(v) for v in values ]
		if issubclass(typ, enum.Enum):
			return [ typ[v] for v in values ]
		raise ValueError('Invalid type for sweep argument: %r' % typ)

def add_sweepable_argument(parser, name, type=str, nargs=None, metavar=None, prefix_dashes=2, help=None, default=None, sweep_on_default=False):
	default = default if (isinstance(default,(list,np.ndarray)) and sweep_on_default) else [default]
	return parser.add_argument(name, action=SweepAction, nargs=nargs, metavar=metavar, type=type, help=help, default=default)
