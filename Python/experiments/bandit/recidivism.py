import numpy as np
import warnings
from time import time
import pandas as pd

# SeldonianML imports
from utils import argsweep, experiment
from datasets  import propublica_bandit as ProPublica
from core.srl_fairness  import GroupFairnessSRL
from baselines.naive_full  import GroupFairnessNaiveSRL

# Supress sklearn FutureWarnings for SGD
warnings.simplefilter(action='ignore', category=FutureWarning)

# Imports for baseline algorithms
from baselines.POEM.Skylines import PRMWrapper
from baselines.POEM.DatasetReader import BanditDataset
from contextualbandits.offpolicy import OffsetTree
from sklearn.linear_model import LogisticRegression



import core.srl_fairness as SRL
import baselines.naive_full   as NSRL

########################################
#   Helpers for selection SRL models   #
########################################

def get_srl_class(definition='GroupFairness', ci_type='ttest'):
	if not(ci_type in ['ttest', 'bootstrap']):
		raise ValueError('get_srl_class(): Unknown ci_type, "%s".' % ci_type)
	if not (definition in ['GroupFairness', 'DisparateImpact']):
		raise ValueError('get_srl_class(): Unknown fairness definition, "%s".' % definition)
	if definition == 'GroupFairness' and ci_type == 'ttest':
		return SRL.GroupFairnessSRL
	if definition == 'GroupFairness' and ci_type == 'bootstrap':
		return SRL.GroupFairnessBootstrapSRL
	if definition == 'DisparateImpact' and ci_type == 'ttest':
		return SRL.DisparateImpactSRL
	if definition == 'DisparateImpact' and ci_type == 'bootstrap':
		return SRL.DisparateImpactBootstrapSRL

def get_nsrl_class(definition='GroupFairness', ci_type='ttest'):
	if not(ci_type in ['ttest', 'bootstrap']):
		raise ValueError('get_nsrl_class(): Unknown ci_type, "%s".' % ci_type)
	if not (definition in ['GroupFairness', 'DisparateImpact']):
		raise ValueError('get_nsrl_class(): Unknown fairness definition, "%s".' % definition)
	if definition == 'GroupFairness' and ci_type == 'ttest':
		return NSRL.GroupFairnessNaiveSRL
	if definition == 'GroupFairness' and ci_type == 'bootstrap':
		return NSRL.GroupFairnessBootstrapNaiveSRL
	if definition == 'DisparateImpact' and ci_type == 'ttest':
		return NSRL.DisparateImpactNaiveSRL
	if definition == 'DisparateImpact' and ci_type == 'bootstrap':
		return NSRL.DisparateImpactBootstrapNaiveSRL


########################
#   Model Evaluators   #
########################

def eval_offset_trees(dataset, mp):
	t = time()
	S, A, R, T, P = dataset.training_splits(flatten=True)
	new_policy = OffsetTree(base_algorithm=LogisticRegression(), nchoices=dataset.n_actions)
	new_policy.fit(X=S, a=A, r=R, p=P)
	t_train = time() - t

	# Evaluate using SRL's evaluate method
	n_actions = dataset.n_actions
	def is_positive(A):
		return A > np.floor(n_actions/2)
	def predict_proba(S):
		S = S[:,0,:]
		AP = new_policy.predict(S)
		P  = np.zeros((len(AP), dataset.n_actions))
		for i,a in enumerate(AP):
			P[i,a] = 1.0
		return P[:,None,:]
	model_params = {
		'epsilon'        : mp['e'],
		'delta'          : mp['d'],
		'minimum_return' : dataset.min_reward }
	SC_Class = get_srl_class(mp['definition'], mp['ci_type'])
	model = SC_Class(is_positive, **model_params)
	results = model.evaluate(dataset, probf=predict_proba)
	results['train_time'] = t_train
	return results

def eval_poem(dataset, mp):
	# Represent our data in a form compatible with POEM
	bandit_dataset = BanditDataset(None, verbose=False)
	S, A, R, T, P = dataset.testing_splits(flatten=True)
	labels = np.zeros((len(A),dataset.n_actions))
	for i, a in enumerate(A):
		labels[i,a] = 1.0
	bandit_dataset.testFeatures = S
	bandit_dataset.testLabels = labels
	S, A, R, T, P = dataset.training_splits(flatten=True)
	labels = np.zeros((len(A),dataset.n_actions))
	for i, a in enumerate(A):
		labels[i,a] = 1.0
	bandit_dataset.trainFeatures = S
	bandit_dataset.trainLabels = labels
	bandit_dataset.registerSampledData(labels, np.log(P), R)
	bandit_dataset.createTrainValidateSplit(0.1)
    
    # Train POEM
	ss = np.random.random((dataset.n_features, dataset.n_actions))
	maj = PRMWrapper(bandit_dataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = R.min(), maxV = R.max(),
								minClip = 0, maxClip = 0, estimator_type = 'Stochastic', verbose = False,
								parallel = None, smartStart = ss)
	maj.calibrateHyperParams()
	t_train = maj.validate()

	# Evaluate using SRL's evaluate method
	n_actions = dataset.n_actions
	def is_positive(A):
		return A > np.floor(n_actions/2)
	def predict_proba(S):
		S = S[:,0,:]
		V = S.dot(maj.labeler.coef_).astype('float64')
		EV = np.exp(V)
		return (EV / EV.sum(axis=1)[:,None])[:,None,:]
	model_params = {
		'epsilon'        : mp['e'],
		'delta'          : mp['d'],
		'minimum_return' : dataset.min_reward }
	SC_Class = get_srl_class(mp['definition'], mp['ci_type'])
	model = SC_Class(is_positive, **model_params)
	results = model.evaluate(dataset, probf=predict_proba)
	results['train_time'] = t_train
	return results


def eval_naive(dataset, mp):
	# Train the model
	n_actions = dataset.n_actions
	def is_positive(A):
		return A > np.floor(n_actions/2)
	model_params = {
		'epsilon'        : mp['e'],
		'delta'          : mp['d'],
		'minimum_return' : dataset.min_reward }
	t = time()
	SC_Class = get_srl_class(mp['definition'], mp['ci_type'])
	model = SC_Class(is_positive, **model_params)
	model.fit(dataset, n_iters=mp['n_iters'], optimizer_name='cmaes')
	t_train = time() - t
	# Assess the model
	results = model.evaluate(dataset, probf=model.get_probf())
	results['train_time'] = t_train
	return results

def eval_sb(dataset, mp):
	# Train the model
	n_actions = dataset.n_actions
	def is_positive(A):
		return A > np.floor(n_actions/2)
	model_params = {
		'epsilon'        : mp['e'],
		'delta'          : mp['d'],
		'minimum_return' : dataset.min_reward }
	t = time()
	SC_Class = get_srl_class(mp['definition'], mp['ci_type'])
	model = SC_Class(is_positive, **model_params)
	model.fit(dataset, n_iters=mp['n_iters'], optimizer_name='cmaes')
	t_train = time() - t
	# Assess the models
	results = model.evaluate(dataset)
	results['train_time'] = t_train
	return results


######################
#   Dataset Loader   #
######################

def load_dataset(tparams, seed):
	dset_args = {
		'r_train'           : tparams['r_train_v_test'], 
		'r_candidate'       : tparams['r_cand_v_safe'], 
		'T0'                : tparams['T0_label'], 
		'T1'                : tparams['T1_label'], 
		'dset_type'         : tparams['dset_type'], 
		'seed'              : seed, 
		'include_T'         : tparams['include_T'], 
		'include_intercept' : not(tparams['omit_intercept']),
		'use_pct'           : tparams['data_pct'] ,
		'use_score_text'    : tparams['use_score_text'],
		'rwd_recid'         : tparams['rwd_recid'],
		'rwd_nonrecid'      : tparams['rwd_nonrecid'],
		'use_cached_gps'    : True
	}
	return ProPublica.load(**dset_args)	


############
#   Main   #
############

if __name__ == '__main__':
	# Note: This script computes experiments for the cross product of all values given for the
	#       sweepable arguments. 
	# Note: Sweepable arguments allow inputs of the form, <start>:<end>:<increment>, which are then
	#       expanded into ranges via np.arange(<start>, <end>, <increment>). 
	#       Eventually I'll add a nice usage string explaining this.
	with argsweep.ArgumentSweeper() as parser:
		#    Execution parameters
		parser.add_argument('base_path',      type=str)
		parser.add_argument('--n_jobs',       type=int, default=4,  help='Number of processes to use.')
		parser.add_argument('--n_trials',     type=int, default=10, help='Number of trials to run.')
		#    Dataset arguments
		parser.add_sweepable_argument('--r_train_v_test', type=float, default=0.4,  nargs='*', help='Ratio of data used for training vs testing.')
		parser.add_sweepable_argument('--r_cand_v_safe',  type=float, default=0.4,  nargs='*', help='Ratio of training data used for candidate selection vs safety checking. (SMLA only)')
		parser.add_argument('--T0_label',       type=str, default='African-American', help='ID for type 0.')
		parser.add_argument('--T1_label',       type=str, default='Caucasian',        help='ID for type 1.')
		parser.add_argument('--dset_type',      type=str, default='violent',          help='Version of the ProPublica data to use (violent or nonviolent).')
		parser.add_argument('--include_T',      action='store_true', help='Whether or not to include type as a predictive feature.')
		parser.add_argument('--omit_intercept', action='store_true', help='Whether or not to include an intercept as a predictive feature (included by default).')
		parser.add_sweepable_argument('--data_pct', type=float, default=1.0,   nargs='*', help='Percentage of the overall size of the dataset to use.')
		parser.add_argument('--use_score_text', action='store_true', help='Whether or not to base actions off of the COMPAS score text (default uses the "decile_score" feature).')
		parser.add_argument('--rwd_recid',      type=float, default=-1.0, help='Reward for instances of recidivism.')
		parser.add_argument('--rwd_nonrecid',   type=float, default=1.0,  help='Reward for instances of non-recidivism.')
		#    Seldonian algorithm parameters
		parser.add_argument('--ci_type',    type=str, default='ttest',        help='Choice of confidence interval to use in the Seldonian methods.')
		parser.add_argument('--definition',    type=str, default='GroupFairness',        help='Choice of safety definition to enforce.')
		parser.add_argument('--n_iters',    type=int,  default=10,    help='Number of SMLA training iterations.')
		parser.add_sweepable_argument('--e',          type=float,  default=0.05,  nargs='*', help='Values for epsilon.')
		parser.add_sweepable_argument('--d',          type=float,  default=0.05,  nargs='*', help='Values for delta.')
		args = parser.parse_args()
		args_dict = dict(args.__dict__)
		
		# Resolve the names for the SMLAs that will be tested
		model_name = get_srl_class(args.definition, args.ci_type).__name__
		smla_names = [model_name]

		# Define the evaluators to be included in the experiment and specify which ones are Seldonian
		model_evaluators = {
			model_name   : eval_sb,
			'POEM'       : eval_poem,
			'OffsetTree' : eval_offset_trees,
			'Naive'      : eval_naive
		}
		
		#    Store task parameters:
		tparam_names = ['n_jobs', 'base_path', 'data_pct', 'T0_label', 'T1_label', 'r_train_v_test', 'r_cand_v_safe', 'include_T', 'dset_type', 'omit_intercept', 'use_score_text', 'rwd_recid', 'rwd_nonrecid']
		tparams = {k:args_dict[k] for k in tparam_names}
		#    Store method parameters:
		srl_mparam_names  = ['e','d','n_iters', 'ci_type', 'definition']
		bsln_mparam_names = ['e','d','n_iters', 'ci_type', 'definition']
		
		mparams = {}
		for name in model_evaluators.keys():
			if name in smla_names:
				mparams[name] = {k:args_dict[k] for k in srl_mparam_names}
			else:
				mparams[name] = {k:args_dict[k] for k in bsln_mparam_names}
		#    Expand the parameter sets into a set of configurations
		tparams, mparams = experiment.make_parameters(tparams, mparams, expand=parser._sweep_argnames)
		# Create a results file and directory
		print()
		save_path = experiment.prepare_paths(args.base_path, tparams, mparams, smla_names, root='results', filename=None)
		# Run the experiment
		print()
		experiment.run(args.n_trials, save_path, model_evaluators, load_dataset, tparams, mparams, n_workers=args.n_jobs, seed=None)