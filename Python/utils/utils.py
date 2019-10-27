import sys
import code
import time
import joblib
import itertools
import os
import threading
import copy
import numpy as np
import datetime
from scipy.spatial          import ConvexHull
from matplotlib.collections import LineCollection
from matplotlib.colors      import to_rgba_array


def make_seed(digits=8, random_state=np.random):
    return np.floor(random_state.rand()*10**digits).astype(int)

def subdir_incrementer(sd):
	for i in itertools.count():
		yield (sd+'_%d') % i


######################################################
#   Helper for generating reasonable thetas for LC   #
######################################################

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
		

#####################################################
#   Helpers for dividing parameters among workers   #
#####################################################

def split_evenly(v, n):
	return [ len(a) for a in np.array_split(np.arange(v),n) ]

def _stack_dicts(base, next, n, replace=False, max_depth=np.inf, na_val=None):
	if isinstance(next,dict) and isinstance(base,dict):
		if max_depth <= 0:
			return np.array([base,next]) if (n>0) else np.array([next])
		out  = {}
		keys = set(base.keys()).union(next.keys())
		for k in keys:
			_base = base[k] if (k in base.keys()) else None
			_next = next[k] if (k in next.keys()) else None
			out[k] = _stack_dicts(_base, _next, n, replace, max_depth-1, na_val=na_val)
	elif isinstance(next,dict) and (base is None):
		out = _stack_dicts({}, next, n, replace, max_depth, na_val=na_val)
	elif isinstance(base,dict) and (next is None):
		out = _stack_dicts(base, {}, n, replace, max_depth, na_val=na_val)
	else:
		if replace:
			out = next if (base is None) else base
		else:
			base_val = np.repeat(na_val,n) if (base is None) else base
			next_val = na_val              if (next is None) else next
			out = np.array(base_val.tolist() + [next_val])
	return out

def stack_all_dicts(*dicts, na_val=None):
	out = {}
	for i,d in enumerate(dicts):
		out = _stack_dicts(out, d, i, max_depth=np.inf, na_val=na_val)
	return out

def stack_all_dicts_shallow(*dicts, na_val=None):
	out = {}
	for i,d in enumerate(dicts):
		out = _stack_dicts(out, d, i, max_depth=1, na_val=na_val)
	return out


#################
#   Profiling   #
#################

class TimeAccumulator(object):
	def __init__(self, n_events=1):
		self.t_last = time.time()
		self.times  = np.zeros(n_events)

	def tick(self, i):
		t = time.time()
		self.times[i] += t - self.t_last
		self.t_last = t

	def total(self, i=None):
		if i is None:
			return sum(self.times)
		return self.times[i]

	def percentages(self):
		tot = self.total()
		if tot == 0.0:
			return np.zeros_like(self.times)
		return 100.0 * self.times/tot

class DurationEstimator(object):
	def __init__(self, min_completed=0.3):
		self._min_completed = min_completed
		self.reset()
	def reset(self):
		self._t = datetime.datetime.utcnow()
	def time_remaining(self, r_completed):
		t_delta = datetime.datetime.utcnow() - self._t
		return t_delta * (1-r_completed) / r_completed
	def time_remaining_str(self, r_completed):
		if r_completed < self._min_completed:
			return 'N/A'
		remainder = self.time_remaining(r_completed).total_seconds()
		weeks, remainder = divmod(remainder, 604800)
		days,  remainder = divmod(remainder, 86400)
		hours, remainder = divmod(remainder, 3600)
		minutes, seconds = divmod(remainder, 60)
		remain_str  = (' %dw' % weeks  ) if weeks   > 0 else ''
		remain_str += (' %dd' % days   ) if days    > 0 else ''
		remain_str += (' %dh' % hours  ) if hours   > 0 else ''
		remain_str += (' %dm' % minutes) if minutes > 0 else ''
		remain_str += (' %ds' % seconds) if seconds > 0 else ''
		return remain_str.strip()


#################
#   Debugging   #
#################

def keyboard(quit=False, banner=''):
	''' Interrupt program flow and start an interactive session in the current frame.
		 * quit   : If True, exit the program upon terminating the session. '''
	try:
		raise None
	except:
		frame = sys.exc_info()[2].tb_frame.f_back
	namespace = frame.f_globals.copy()
	namespace.update(frame.f_locals)
	from sys import exit as quit
	namespace.update({'quit':quit})
	code.interact(banner=banner, local=namespace)
	if quit:
		sys.exit()


#######################################################
#   Plots curves as traces with fading color trails   #
#######################################################

def _get_colors(colors, n):
	colors = to_rgba_array(colors)
	if colors.shape[0] == 1:
		colors = np.tile(colors, (n,1))
	else:
		assert colors.shape[0] == n, 'utils._get_colors: Incorrect number of colors. Expected %d, recieved %d.' % (n,colors.shape[0])
	return colors

def plot_traces(ax, traces, colors='b', linewidths=2):
	# base_colors = plt.cm.jet(pp.minmax_scale(Y)) 
	n_traces    = len(traces)
	base_colors = _get_colors(colors, n_traces)
	lines  = []
	colors = []
	for i, trace in enumerate(traces):
		n_segments = len(trace) - 1
		for j in range(n_segments):
			lines.append(np.array([ trace[j], trace[j+1] ]))
			color = base_colors[i]
			color[-1] = float(j+1) / n_segments
			colors.append(tuple(color))
	lc = LineCollection(lines, colors=colors, linewidths=linewidths)
	ax.add_collection(lc)
	return ax