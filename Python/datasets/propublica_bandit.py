import numpy as np
import pandas as pd
import os.path

import joblib
from datasets.dataset import BanditDataset
from utils import keyboard


BASE_URL = os.path.join('datasets', 'propublica')
COMPAS_PATHS = {
	'nonviolent' : os.path.join(BASE_URL, 'compas-scores-two-years.csv'),
	'violent'    : os.path.join(BASE_URL, 'compas-scores-two-years-violent.csv')
}
LABELS_TO_KEEP = np.array(['sex', 'age', 'race', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'c_charge_degree'])


def load(r_train=0.4, r_candidate=0.2, T0='Caucasian', T1='African-American', dset_type='nonviolent', seed=None, include_T=False, include_intercept=True, use_pct=1.0, use_score_text=False, rwd_recid=-1.0, rwd_nonrecid=1.0, use_cached_gps=True):
	random = np.random.RandomState(seed)
	scores = pd.read_csv(COMPAS_PATHS[dset_type])

	# Filter Unusable Rows
	scores = scores[scores.days_b_screening_arrest <=  30]
	scores = scores[scores.days_b_screening_arrest >= -30]
	scores = scores[scores.is_recid != -1]
	scores = scores[scores.c_charge_degree != "0"]
	scores = scores[scores.score_text != 'N/A']

	# Generate the full dataset
	S = scores[np.logical_or(scores.race==T0, scores.race==T1)].copy()
	if use_score_text:
		st = S['score_text']
		A = (st=='Medium') + 2*(st=='High')
	else:
		A = S['decile_score'].astype(int)
	A = A.values
	A[A==-1] = 1
	A = A - A.min()
	n_actions = len(np.unique(A))
	R = np.sign(S['two_year_recid'].values-0.5)
	R = (R==-1)*rwd_nonrecid + (R==1)*rwd_recid

	S = S[LABELS_TO_KEEP]
	S = with_dummies(S, 'sex')
	S = with_dummies(S, 'c_charge_degree', label='crime_degree')
		
	T = 1 * (S.race==T1).values
	del S['race']
	L = np.array(S.columns, dtype=str)
	S = S.values
	if include_intercept:
		S = np.hstack((S, np.ones((len(S),1))))
		L = np.hstack((L, ['intercept']))
	if include_T:
		S = np.hstack((T[:,None], S))
		L = np.hstack((['type'], L))	


	n_keep = int(np.ceil(len(S) * use_pct))
	I = np.arange(len(S))
	random.shuffle(I)
	I = I[:n_keep]
	S = S[I]
	A = A[I]
	R = R[I]
	T = T[I]	

	# Compute split sizes
	n_samples   = len(S)
	n_train     = int(r_train*n_samples)
	n_test      = n_samples - n_train
	n_candidate = int(r_candidate*n_train)
	n_safety    = n_train - n_candidate
	max_reward = max(rwd_recid, rwd_nonrecid)
	min_reward = min(rwd_recid, rwd_nonrecid)

	# Load cached GPs if requested
	if use_cached_gps:
		if use_score_text:
			proba_gp_path = os.path.join(BASE_URL, '%s_score_text_proba_gp.joblib' % dset_type)
			rwd_gp_path   = os.path.join(BASE_URL, '%s_score_text_rwd_gp.joblib' % dset_type)
		else:
			proba_gp_path = os.path.join(BASE_URL, '%s_decile_score_proba_gp.joblib' % dset_type)
			rwd_gp_path   = os.path.join(BASE_URL, '%s_decile_score_rwd_gp.joblib' % dset_type)
		proba_gp  = joblib.load(proba_gp_path)
		return_gp = joblib.load(rwd_gp_path)
		X = np.hstack((S,T[:,None]))
		Ps = proba_gp.predict_proba(X)
		P = np.array([ Ps[i,a] for i,a in enumerate(A) ])
	else:
		P = None

	dataset = BanditDataset(S, A, R, n_actions, n_candidate, n_safety, n_test, min_reward, max_reward, seed=seed, P=P, T=T)
	dataset.X_labels = L
	dataset.T0_label = T0
	dataset.T1_label = T1

	# Store the GPs that were loaded, if using cached GPs
	if use_cached_gps:
		dataset._proba_gp  = proba_gp
		dataset._return_gp = return_gp
	return dataset

def with_dummies(dataset, column, label=None, keep_orig=False, zero_index=True):
	dataset = dataset.copy()
	assert column in dataset.columns, 'with_dummies(): column %r not found in dataset.'%column
	if label is None:
		label = column
	dummies = pd.get_dummies(dataset[column], prefix=label, prefix_sep=':')
	for i,col in enumerate(dummies.columns):
		col_name = col
		if zero_index and (len(dummies.columns) > 1):
			if i > 0:
				name, val = col.split(':',1)
				col_name = ':'.join([name, 'is_'+val])
				dataset[col_name] = dummies[col]
		else:
			dataset[col] = dummies[col]
	return dataset if keep_orig else dataset.drop(column,1)