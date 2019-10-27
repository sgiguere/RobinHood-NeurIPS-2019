import numpy as np
import warnings
from time import time
import pandas as pd

# SeldonianML imports
from utils import argsweep, experiment, keyboard
from datasets  import tutoring_bandit as TutoringSystem
import core.srl_fairness as SRL
import baselines.naive_full as NSRL

# Supress sklearn FutureWarnings for SGD
warnings.simplefilter(action='ignore', category=FutureWarning)

# Imports for baseline algorithms
from baselines.POEM.Skylines import PRMWrapper
from baselines.POEM.DatasetReader import BanditDataset
from contextualbandits.offpolicy import OffsetTree
from sklearn.linear_model import LogisticRegression


########################################
#   Helpers for selection SRL models   #
########################################

def get_srl_class(bound_ref_return=True, females_only=False, ci_type='ttest'):
	if not(ci_type in ['ttest', 'bootstrap']):
		raise ValueError('get_srl_class(): Unknown ci_type, "%s".' % ci_type)
	if bound_ref_return:
		if females_only:
			if ci_type == 'ttest':
				return SRL.TutoringSystemFemaleSRL
			elif ci_type == 'bootstrap':
				return SRL.TutoringSystemFemaleBootstrapSRL
		else:
			if ci_type == 'ttest':
				return SRL.TutoringSystemSRL
			elif ci_type == 'bootstrap':
				return SRL.TutoringSystemBootstrapSRL
	else:
		if females_only:
			if ci_type == 'ttest':
				return SRL.TutoringSystemFemaleEmpSRL
			elif ci_type == 'bootstrap':
				return SRL.TutoringSystemFemaleBootstrapEmpSRL
		else:
			if ci_type == 'ttest':
				return SRL.TutoringSystemEmpSRL
			elif ci_type == 'bootstrap':
				return SRL.TutoringSystemBootstrapEmpSRL

def get_nsrl_class(bound_ref_return=True, females_only=False, ci_type='ttest'):
	if not(ci_type in ['ttest', 'bootstrap']):
		raise ValueError('get_srl_class(): Unknown ci_type, "%s".' % ci_type)
	if bound_ref_return:
		if females_only:
			if ci_type == 'ttest':
				return NSRL.TutoringSystemFemaleNaiveSRL
			elif ci_type == 'bootstrap':
				return NSRL.TutoringSystemFemaleBootstrapNaiveSRL
		else:
			if ci_type == 'ttest':
				return NSRL.TutoringSystemNaiveSRL
			elif ci_type == 'bootstrap':
				return NSRL.TutoringSystemBootstrapNaiveSRL
	else:
		if females_only:
			if ci_type == 'ttest':
				return NSRL.TutoringSystemFemaleEmpNaiveSRL
			elif ci_type == 'bootstrap':
				return NSRL.TutoringSystemFemaleBootstrapEmpNaiveSRL
		else:
			if ci_type == 'ttest':
				return NSRL.TutoringSystemEmpNaiveSRL
			elif ci_type == 'bootstrap':
				return NSRL.TutoringSystemBootstrapEmpNaiveSRL


########################
#   Model Evaluators   #
########################

def eval_offset_trees(dataset, mp):
	n_actions = dataset.n_actions

	t = time()
	dataset.enable_R_corrections()
	S, A, R, _, P = dataset.training_splits(flatten=True)
	new_policy = OffsetTree(base_algorithm=LogisticRegression(solver='lbfgs'), nchoices=dataset.n_actions)
	new_policy.fit(X=S, a=A, r=R, p=P)
	t_train = time() - t

	def predict_proba(S):
		S = S[:,0,:] 
		AP = new_policy.predict(S)
		P  = np.zeros((len(AP), dataset.n_actions))
		for i,a in enumerate(AP):
			P[i,a] = 1.0
		return P[:,None,:]
	
	# Evaluate using SRL's evaluate method
	dataset.disable_R_corrections()
	sfp = mp['simulated_female_proportion']
	model_params = {
		'epsilon_f'    : mp['e_f'],
		'epsilon_m'    : mp['e_m'],
		'delta'        : mp['d'] }
	if not(mp['simulated_female_proportion'] is None):
		model_params['male_iw_correction']   = (1-mp['simulated_female_proportion'])/np.mean(dataset._T==0)
		model_params['female_iw_correction'] =     mp['simulated_female_proportion']/np.mean(dataset._T==1) 
	min_reward, max_reward = dataset.min_reward, dataset.max_reward
	_, _, R, T, _ = dataset.testing_splits(flatten=True)
	r_ref_T0 = np.mean(R[T==0])
	r_ref_T1 = np.mean(R[T==1])
	TutoringSystemSRL = get_srl_class(mp['bound_ref_return'], mp['females_only'], mp['ci_type'])
	model = TutoringSystemSRL(min_reward, max_reward, r_ref_T0, r_ref_T1, **model_params)
	results = model.evaluate(dataset, probf=predict_proba)
	results['train_time'] = t_train
	return results

def eval_poem(dataset, mp):
	n_actions = dataset.n_actions

	# Represent our data in a form compatible with POEM
	dataset.enable_R_corrections()
	bandit_dataset = BanditDataset(None, verbose=False)
	S, A, R, _, P = dataset.testing_splits(flatten=True)
	labels = np.zeros((len(A),dataset.n_actions))
	for i, a in enumerate(A):
		labels[i,a] = 1.0
	bandit_dataset.testFeatures = S
	bandit_dataset.testLabels = labels
	S, A, R, _, P = dataset.training_splits(flatten=True)
	labels = np.zeros((len(A),dataset.n_actions))
	for i, a in enumerate(A):
		labels[i,a] = 1.0
	bandit_dataset.trainFeatures = S
	bandit_dataset.trainLabels = labels
	bandit_dataset.registerSampledData(labels, np.log(P), -R)  # POEM expects penalties not rewards
	bandit_dataset.createTrainValidateSplit(0.1)
    
    # Train POEM
	ss = np.random.random((dataset.n_features, dataset.n_actions))
	maj = PRMWrapper(bandit_dataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = -6, maxV = 0, minClip = 0, maxClip = 0, estimator_type = 'Stochastic', verbose = False, parallel = None, smartStart = ss)
	maj.calibrateHyperParams()
	t_train = maj.validate()

	# Extract the predictor and construct a proba function
	def predict_proba(S):
		S = S[:,0,:]
		V = S.dot(maj.labeler.coef_).astype('float64')
		EV = np.exp(V)
		return (EV / EV.sum(axis=1)[:,None])[:,None,:]

	# Evaluate using SRL's evaluate method
	dataset.disable_R_corrections()
	model_params = {
		'epsilon_f'    : mp['e_f'],
		'epsilon_m'    : mp['e_m'],
		'delta'        : mp['d'] }
	if not(mp['simulated_female_proportion'] is None):
		model_params['male_iw_correction']   = (1-mp['simulated_female_proportion'])/np.mean(dataset._T==0)
		model_params['female_iw_correction'] =     mp['simulated_female_proportion']/np.mean(dataset._T==1) 
	min_reward, max_reward = dataset.min_reward, dataset.max_reward
	_, _, R, T, _ = dataset.testing_splits(flatten=True)
	r_ref_T0 = np.mean(R[T==0])
	r_ref_T1 = np.mean(R[T==1])
	TutoringSystemSRL = get_srl_class(mp['bound_ref_return'], mp['females_only'], mp['ci_type'])
	model = TutoringSystemSRL(min_reward, max_reward, r_ref_T0, r_ref_T1, **model_params)
	results = model.evaluate(dataset, probf=predict_proba)
	results['train_time'] = t_train
	return results


def eval_naive(dataset, mp):
	n_actions = dataset.n_actions

	# Train the model
	t = time()
	dataset.disable_R_corrections()
	model_params = {
		'epsilon_f'    : mp['e_f'],
		'epsilon_m'    : mp['e_m'],
		'delta'        : mp['d'] }
	if not(mp['simulated_female_proportion'] is None):
		model_params['male_iw_correction']   = (1-mp['simulated_female_proportion'])/np.mean(dataset._T==0)
		model_params['female_iw_correction'] =     mp['simulated_female_proportion']/np.mean(dataset._T==1) 

	min_reward, max_reward = dataset.min_reward, dataset.max_reward
	_, _, R, T, _ = dataset.testing_splits(flatten=True)
	r_ref_T0 = np.mean(R[T==0])
	r_ref_T1 = np.mean(R[T==1])
	TutoringSystemNaiveSRL = get_nsrl_class(mp['bound_ref_return'], mp['females_only'], mp['ci_type'])
	model = TutoringSystemNaiveSRL(min_reward, max_reward, r_ref_T0, r_ref_T1, **model_params)
	model.fit(dataset, n_iters=mp['n_iters'], optimizer_name='cmaes')
	t_train = time() - t
	# Assess the model
	results = model.evaluate(dataset, probf=model.get_probf())
	results['train_time'] = t_train
	return results

def eval_sb(dataset, mp):
	n_actions = dataset.n_actions

	# Train the model
	t = time()
	dataset.disable_R_corrections()
	model_params = {
		'epsilon_f'    : mp['e_f'],
		'epsilon_m'    : mp['e_m'],
		'delta'        : mp['d'] }
	if not(mp['simulated_female_proportion'] is None):
		model_params['male_iw_correction']   = (1-mp['simulated_female_proportion'])/np.mean(dataset._T==0)
		model_params['female_iw_correction'] =    mp['simulated_female_proportion'] /np.mean(dataset._T==1) 

	min_reward, max_reward = dataset.min_reward, dataset.max_reward
	_, _, R, T, _ = dataset.testing_splits(flatten=True)
	r_ref_T0 = np.mean(R[T==0])
	r_ref_T1 = np.mean(R[T==1])
	TutoringSystemSRL = get_srl_class(mp['bound_ref_return'], mp['females_only'], mp['ci_type'])
	model = TutoringSystemSRL(min_reward, max_reward, r_ref_T0, r_ref_T1, **model_params)
	model.fit(dataset, n_iters=mp['n_iters'], optimizer_name='cmaes')
	t_train = time() - t
	# Assess the model
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
		'include_T'         : tparams['include_T'], 
		'include_intercept' : not(tparams['omit_intercept']),
		'use_pct'           : tparams['data_pct'],
		'remove_biased_tutorial' : tparams['remove_biased_tutorial'],
		'simulated_female_proportion' : tparams['simulated_female_proportion']
	}
	return TutoringSystem.load(**dset_args)	


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
		parser.add_argument('--status_delay', type=int, default=30, help='Number of seconds between status updates when running multiple jobs.')
		parser.add_argument('base_path',      type=str)
		parser.add_argument('--n_jobs',       type=int, default=4,  help='Number of processes to use.')
		parser.add_argument('--n_trials',     type=int, default=10, help='Number of trials to run.')
		#    Dataset arguments
		parser.add_sweepable_argument('--r_train_v_test', type=float, default=0.4,  nargs='*', help='Ratio of data used for training vs testing.')
		parser.add_sweepable_argument('--r_cand_v_safe',  type=float, default=0.4,  nargs='*', help='Ratio of training data used for candidate selection vs safety checking. (SMLA only)')
		parser.add_argument('--include_T',      action='store_true', help='Whether or not to include type as a predictive feature.')
		parser.add_argument('--omit_intercept', action='store_false', help='Whether or not to include an intercept as a predictive feature (included by default).')
		parser.add_sweepable_argument('--data_pct', type=float, default=1.0,   nargs='*', help='Percentage of the overall size of the dataset to use.')
		parser.add_argument('--use_score_text', action='store_true', help='Whether or not to base actions off of the COMPAS score text (default uses the "decile_score" feature).')
		parser.add_argument('--rwd_recid',      type=float, default=-1.0, help='Reward for instances of recidivism.')
		parser.add_argument('--rwd_nonrecid',   type=float, default=1.0,  help='Reward for instances of non-recidivism.')
		parser.add_argument('--simulated_female_proportion',  type=float, default=None, help='If specified, rescales the importance weight terms to simulate having the specified proportion of females.')
		#    Seldonian algorithm parameters
		parser.add_argument('--females_only',      action='store_true', help='If enabled, only enforce the constraint for females.')
		parser.add_argument('--bound_ref_return',  action='store_true', help='If enabled, also bound the expected return of the behavior policy.')
		parser.add_argument('--ci_type',    type=str, default='ttest',        help='Choice of confidence interval to use in the Seldonian methods.')
		parser.add_argument('--n_iters',    type=int,  default=10,    help='Number of SMLA training iterations.')
		parser.add_argument('--remove_biased_tutorial',  action='store_true',      help='If true, remove the tutorial that is slanted against females (default is to include all data).')
		parser.add_sweepable_argument('--e_f', type=float,  default=0.00,  nargs='*', help='Values for epsilon for the female constraint.')
		parser.add_sweepable_argument('--e_m', type=float,  default=0.00,  nargs='*', help='Values for epsilon for the male constraint (no effect if --females_only).')
		parser.add_sweepable_argument('--d',          type=float,  default=0.05,  nargs='*', help='Values for delta.')
		args = parser.parse_args()
		args_dict = dict(args.__dict__)
		
		# Define the evaluators to be included in the experiment and specify which ones are Seldonian
		model_name = get_srl_class(args.bound_ref_return, args.females_only, args.ci_type).__name__
		model_evaluators = {
			model_name   : eval_sb,
			'POEM'       : eval_poem,
			'OffsetTree' : eval_offset_trees,
			'Naive'      : eval_naive
		}
		smla_names = [model_name]
		
		# Store task parameters:
		tparam_names = ['n_jobs', 'base_path', 'data_pct', 'r_train_v_test', 'r_cand_v_safe', 'include_T', 'omit_intercept', 'use_score_text', 'rwd_recid', 'rwd_nonrecid', 'remove_biased_tutorial', 'simulated_female_proportion']
		tparams = {k:args_dict[k] for k in tparam_names}
		# Store method parameters:
		srl_mparam_names  = ['e_f', 'e_m', 'd', 'n_iters', 'ci_type', 'females_only', 'bound_ref_return', 'simulated_female_proportion']
		bsln_mparam_names = ['e_f', 'e_m', 'd', 'n_iters', 'ci_type', 'females_only', 'bound_ref_return', 'simulated_female_proportion']
		mparams = {}
		for name in model_evaluators.keys():
			if name in smla_names:
				mparams[name] = {k:args_dict[k] for k in srl_mparam_names}
			else:
				mparams[name] = {k:args_dict[k] for k in bsln_mparam_names}
		# Expand the parameter sets into a set of configurations
		tparams, mparams = experiment.make_parameters(tparams, mparams, expand=parser._sweep_argnames)
		# Create a results file and directory
		print()
		save_path = experiment.prepare_paths(args.base_path, tparams, mparams, smla_names, root='results', filename=None)
		# Run the experiment
		print()
		experiment.run(args.n_trials, save_path, model_evaluators, load_dataset, tparams, mparams, n_workers=args.n_jobs, seed=None)
