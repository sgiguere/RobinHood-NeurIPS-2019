import numpy as np
import os.path
import pandas as pd

from datasets.dataset import BanditDataset
from utils import keyboard
from IPython import embed

BASE_URL = os.path.join('datasets', 'mturk')
TUTOR_PATH = {
	'mturk' : os.path.join(BASE_URL, 'mturk.csv')
}

#in gender column, 0==male, 1==female
def load(r_train=0.4, r_candidate=0.2, seed=None, include_T=False, include_intercept=True, use_pct=1.0, remove_biased_tutorial=False, simulated_female_proportion=None):
	
	random = np.random.RandomState(seed)
	scores = pd.read_csv(TUTOR_PATH['mturk'])
	
	#Filter unusable rows
		#in csv file -9 indicated users that did not indicate
		#their gender
	scores = scores[scores.gender != -9]
	
	#Generate the full dataset
	n_samples = len(scores)
	S = np.ones((n_samples,1)) # A dummy context to allow learning without modifying the contextual bandit code
	A = scores.tutorial.values
	R = scores.quizScore.values.astype(np.float)
	T = scores.gender.copy().values

	if remove_biased_tutorial:
		avals = np.unique(A)
		Rm = np.array([ R[np.logical_and(T==0,A==a)].mean() for a in avals ])
		Rf = np.array([ R[np.logical_and(T==1,A==a)].mean() for a in avals ])
		I  = A != avals[np.argmax(Rm-Rf)]
		S = S[I]
		A = A[I]
		R = R[I]
		T = T[I]
		A[A > avals[np.argmax(Rm-Rf)]] -= 1

	# Rescale R to simulate different proportions of female/male samples
	if simulated_female_proportion is None:
		def correct_R_func(s,a,r,t):
			return r
	else:
		def correct_R_func(s,a,r,t, sfp=simulated_female_proportion, T=T):
			if t == 0:
				return r * (1-sfp) / np.mean(T==0)
			else:
				return r * sfp / np.mean(T==1) 


	# Shift the action ids to start at 0
	avals = np.sort(np.unique(A))
	A = np.array([ np.argmax(a==avals) for a in A ])

	# include the T variable if required
	if include_T:
		S = np.hstack((T[:,None], S))

	# Use the specified percent of the data
	n_keep = int(np.ceil(len(S) * use_pct))
	I = np.arange(len(S))
	random.shuffle(I)
	I = I[:n_keep]
	S = S[I]
	A = A[I]
	R = R[I]
	T = T[I]	
	# This is a bit of a hack: our actual proportions didn't match 1/3
	probs = [ np.mean(A==a) for a in [0,1,2]]
	P = np.array([ probs[a] for a in A ])
	# P = np.ones_like(A) * (1/3) # reference probabilities were all 1/3 in the experiment
	
	n_actions = len(np.unique(A))
	n_samples = len(S)
	n_train = int(r_train*n_samples)
	n_test = n_samples-n_train
	n_candidate = int(r_candidate*n_train)
	n_safety = n_train-n_candidate
	max_reward = 10.0
	min_reward = 0

	dataset = BanditDataset(S,A,R,n_actions, n_candidate, n_safety, n_test, min_reward, max_reward, P=P,T=T, seed=seed, Rc_func=correct_R_func)
	return dataset
	
