import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import hex2color

import utils
from utils.io import SMLAResultsReader
from datasets import propublica_bandit as propublica
from datasets import credit_bandit as credit
from datasets import credit_bandit as credit
from datasets import tutoring_bandit as tutoring
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")



if __name__ == '__main__':
	filetype = 'png'

	gf_path  = 'results/recidivism_gf_0/recidivism_gf.h5'

	ts_tr_path  = 'results/tutoring_0p5_0/tutoring_0p5.h5'
	ts_sim_path = 'results/tutoring_skew_0p5_0/tutoring_skew_0p5.h5'

	cs_di_path  = 'results/credit_assignment_di_0/credit_assignment_di.h5'
	cs_gf_path  = 'results/credit_assignment_gf_0/credit_assignment_gf.h5'
	cs_di_online_path = 'results/credit_assignment_online_di_0/credit_assignment_online_di.h5'


	figpath = 'figures/neurips'
	dpi = 200
	c_smla = '#219d5cff'
	c_bsln = '#0080bfff'

	delta = 0.05

	base_smla_names = ['SRL']
	base_bsln_names = ['POEM','OffsetTree','Naive']

	n_propublica  = len(propublica.load(r_train=0.4).training_splits()[0])
	n_credit      = len(credit.load(r_train=0.4).training_splits()[0])
	n_tutoring_tp = len(tutoring.load(r_train=0.8, remove_biased_tutorial=True).training_splits()[0])
	n_tutoring_sp = len(tutoring.load(r_train=0.8).training_splits()[0])

	if not(os.path.isdir(figpath)):
		os.makedirs(figpath)		

	at_least_one_exp = False

	smla_color = hex2color('#4daf4a')
	baseline_colormap = {
		'IntervalChaining' : hex2color('#ffff33'),
		'RidgeFair'        : hex2color('#377eb8'),
		'POEM'             : hex2color('#984ea3'),
		'OffsetTree'       : hex2color('#ff7f00'),
		'Naive'            : hex2color('#e41a1c')
	}
	default_bsln_color = hex2color('#000000')

	###############
	#   Helpers   #
	###############

	def get_ls(mn):
		if mn.endswith('SRL'):
			return '--'
		elif mn.endswith('Naive'):
			return '-'
		return '-'

	def save(fig, path, *args, **kwargs):
		if not(os.path.isdir(figpath)):
			os.makedirs(figpath)
		path = os.path.join(figpath, path)
		print('Saving figure to \'%s\'' % path)
		fig.savefig(path, *args, **kwargs)

	def get_samples(results, header, dpct, n_constraints=1, base_bsln_names=base_bsln_names):
		''' Helper for filtering results files. '''
		_smla_names = [header+s for s in base_smla_names]
		_bsln_names = base_bsln_names.copy()

		fields = ['accept','return_test', 'data_pct']
		for i in range(n_constraints):
			fields.append('test_bqf_%d_mean' % i)
			fields.append('test_co_%d_mean'  % i)

		# Get the SMLA samples
		smla_samples = []
		smla_names   = []
		for nm in _smla_names:
			smla_names.append(nm)
			sample = results.extract(fields, name=nm, data_pct=dpct)
			smla_samples.append(sample)

		# get the baseline samples (note different versions of SGD)
		bsln_samples = []
		bsln_names   = []
		for nm in _bsln_names:
			bsln_names.append(nm)
			sample = results.extract(fields, name=nm, data_pct=dpct)
			bsln_samples.append(sample)

		is_smla = np.array([True]*len(smla_names) + [False]*len(bsln_names))
		return is_smla, (smla_names, smla_samples), (bsln_names, bsln_samples)


	def get_stats(path, header, n_total, n_constraints=1, base_bsln_names=base_bsln_names):
		''' Helper for extracting resutls from brazil results files. '''
		results = SMLAResultsReader(path)
		results.open()

		dpcts = np.array(sorted(results.extract(['data_pct']).data_pct.unique()))
		nvals = np.array(sorted(np.floor(dpcts * n_total).astype(int)))
		is_smla, (smla_names, smla_samples), (bsln_names, bsln_samples) = get_samples(results, header, dpcts.max(), n_constraints=n_constraints, base_bsln_names=base_bsln_names)
		all_samples = smla_samples + bsln_samples
		mnames  = np.array(smla_names + bsln_names)
		
		# Compute statistics and close the results file
		arates, arates_se = [], [] # Acceptance rates and SEs
		frates, frates_se = [], [] # Failure rates ans SEs (rate that accepted solutions have g(theta) > 0 on the test set)
		rrates, rrates_se = [], [] # Test set returns and SEs
		for _dpct in dpcts:
			_, (_,_smla_samples), (_,_bsln_samples) = get_samples(results, header, _dpct, n_constraints=n_constraints, base_bsln_names=base_bsln_names)
			_arates, _arates_se = [], []
			_frates, _frates_se = [], []
			_rrates, _rrates_se = [], []
			for s in _smla_samples + _bsln_samples:
				accepts  = 1 * s.accept
				_arates.append(np.mean(accepts))
				_arates_se.append(np.std(accepts,ddof=1)/np.sqrt(len(accepts)))

				any_violations = np.logical_or.reduce([ (s['test_co_%d_mean'%c]>0) for c in range(n_constraints) ])
				failures = 1 * np.logical_and(any_violations, s.accept)
				_frates.append(np.mean(failures))
				_frates_se.append(np.std(failures,ddof=1)/np.sqrt(len(failures)))

				if any(s.accept):
					returns = s.return_test[s.accept]
					_rrates.append(np.mean(returns))
					_rrates_se.append(np.std(returns,ddof=1)/np.sqrt(len(returns)))
				else:
					_rrates.append(np.nan)
					_rrates_se.append(np.nan)
			arates.append(_arates)
			arates_se.append(_arates_se)
			frates.append(_frates)
			frates_se.append(_frates_se)
			rrates.append(_rrates)
			rrates_se.append(_rrates_se)
		results.close()

		# Assign colors
		colors = []
		for smla, nm in zip(is_smla, mnames):
			if smla:
				colors.append(smla_color)
			elif nm in baseline_colormap.keys():
				colors.append(baseline_colormap[nm])
			else:
				colors.append(default_bsln_color)
		
		return {
			'arate_v_n'    : np.array(arates),
			'arate_se_v_n' : np.array(arates_se),
			'frate_v_n'    : np.array(frates),
			'frate_se_v_n' : np.array(frates_se),
			'acc_v_n'      : np.array(rrates),
			'acc_se_v_n'   : np.array(rrates_se),
			'mnames'       : mnames,
			'colors'       : colors,
			'nvals'        : nvals,
			'is_smla'      : is_smla
		}





	# #########################
	# #   ProPublica Bandit   #
	# #########################

	if os.path.exists(gf_path):
		at_least_one_exp = True

		D = get_stats(gf_path, 'GroupFairness', n_propublica)
		
		arates     = D['arate_v_n']
		arates_se  = D['arate_se_v_n']
		frates     = D['frate_v_n']
		frates_se  = D['frate_se_v_n']
		acc_v_n    = D['acc_v_n']
		acc_se_v_n = D['acc_se_v_n']
		mnames  = D['mnames']
		is_smla = D['is_smla']
		colors  = D['colors']
		nvals   = D['nvals']

		fig, (ax_acc, ax_ar, ax_fr) = plt.subplots(1, 3, figsize=(11, 1.8))
		legend_data = []
		added = []
		
		# Subplot for IW_return
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(acc_v_n.T)[::-1],(acc_se_v_n.T)[::-1]):
			line = ax_acc.plot(nvals, acc, c=c, ls=get_ls(mn))[0]
			ax_acc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
			if mn.endswith('SRL') and not('RobinHood' in added):
				added.append('RobinHood')
				legend_data.append(line)
			elif not(mn in added):
				added.append(mn)
				legend_data.append(line)
		legend_data = legend_data[::-1]
		added = added[::-1]
		ax_acc.set_xlabel('Training Samples', labelpad=1.5)
		ax_acc.text(-0.18, 0.5, 'Reward', horizontalalignment='center', verticalalignment='center', rotation=90, transform=ax_acc.transAxes)
		ax_acc.set_xscale("log")
		ax_acc.set_xlim(right=max(nvals))
		ax_acc.set_ylim((-1.05,1.05))
		ax_acc.xaxis.set_major_formatter(mtick.ScalarFormatter())

		# Subplot for accept rate, 1-Pr(NSF)
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			line = ax_ar.plot(nvals, ar*100, c=c, ls=get_ls(mn))[0]
			ax_ar.fill_between(nvals, 100*(ar+se), 100*(ar-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=1.5)
		ax_ar.set_ylabel('Solution Rate', labelpad=-3)
		ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
		ax_ar.xaxis.set_major_formatter(mtick.ScalarFormatter())

		# Subplot for failure rate Pr(~NSF ^ unsafety)
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(frates.T)[::-1], (frates_se.T)[::-1]):
			line = ax_fr.plot(nvals, fr*100, c=c, ls=get_ls(mn))[0]
			ax_fr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_fr.set_xlabel('Training Samples', labelpad=1.5)
		ax_fr.set_ylabel('Failure Rate')
		ax_fr.axhline(delta*100, color='k', linestyle=':')
		ax_fr.set_xscale("log")
		ax_fr.set_ylim((-2,102))
		ax_fr.set_xlim(right=max(nvals))
		ax_fr.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
		ax_fr.xaxis.set_major_formatter(mtick.ScalarFormatter())

		for ax in [ax_acc, ax_ar, ax_fr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(bottom=0.24, wspace=0.3, top=0.96, left=0.065, right=0.98)
		# fig.legend(legend_data, added, 'lower center', fancybox=True, ncol=len(legend_data), columnspacing=1, fontsize=11, handletextpad=0.5)
		save(fig,'results_criminal_statpar.%s' % filetype, dpi=dpi)



	#########################################################################
	#  Tutoring System: Bounded Reference Return, Biased Tutorial Removed   #
	#########################################################################

	if os.path.exists(ts_tr_path):
		at_least_one_exp = True
		D = get_stats(ts_tr_path, 'TutoringSystemBootstrapEmp', n_tutoring_tp, n_constraints=2)
		
		arates     = D['arate_v_n']
		arates_se  = D['arate_se_v_n']
		frates     = D['frate_v_n']
		frates_se  = D['frate_se_v_n']
		acc_v_n    = D['acc_v_n']
		acc_se_v_n = D['acc_se_v_n']
		mnames  = D['mnames']
		is_smla = D['is_smla']
		colors  = D['colors']
		nvals   = D['nvals']

		fig, (ax_acc, ax_ar, ax_fr) = plt.subplots(1, 3, figsize=(11, 1.8))
		legend_data = []
		added = []
		
		# Subplot for IW_return
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(acc_v_n.T)[::-1],(acc_se_v_n.T)[::-1]):
			line = ax_acc.plot(nvals, acc, c=c, ls=get_ls(mn))[0]
			ax_acc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
			if mn.endswith('SRL') and not('RobinHood' in added):
				added.append('RobinHood')
				legend_data.append(line)
			elif not(mn in added):
				added.append(mn)
				legend_data.append(line)
		legend_data = legend_data[::-1]
		added = added[::-1]
		ax_acc.set_xlabel('Training Samples', labelpad=1.5)
		ax_acc.text(-0.18, 0.5, 'Reward', horizontalalignment='center', verticalalignment='center', rotation=90, transform=ax_acc.transAxes)

		ax_acc.set_xscale("log")
		ax_acc.set_xlim(right=max(nvals))
		ax_acc.set_ylim((-0.5,10.5))
		ax_acc.xaxis.set_major_formatter(mtick.ScalarFormatter())

		# Subplot for accept rate, 1-Pr(NSF)
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			line = ax_ar.plot(nvals, ar*100, c=c, ls=get_ls(mn))[0]
			ax_ar.fill_between(nvals, 100*(ar+se), 100*(ar-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=1.5)
		ax_ar.set_ylabel('Solution Rate', labelpad=-3)
		ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
		ax_ar.xaxis.set_major_formatter(mtick.ScalarFormatter())

		# Subplot for failure rate Pr(~NSF ^ unsafety)
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(frates.T)[::-1], (frates_se.T)[::-1]):
			line = ax_fr.plot(nvals, fr*100, c=c, ls=get_ls(mn))[0]
			ax_fr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_fr.set_xlabel('Training Samples', labelpad=1.5)
		ax_fr.set_ylabel('Failure Rate')
		ax_fr.axhline(delta*100, color='k', linestyle=':')
		ax_fr.set_xscale("log")
		ax_fr.set_ylim((-2,102))
		ax_fr.set_xlim(right=max(nvals))
		ax_fr.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
		ax_fr.xaxis.set_major_formatter(mtick.ScalarFormatter())

		for ax in [ax_acc, ax_ar, ax_fr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(bottom=0.24, wspace=0.3, top=0.96, left=0.065, right=0.98)
		save(fig,'results_tsys_uniform.%s' % filetype, dpi=dpi)



	#######################################################################
	#  Tutoring System: Bounded Reference Return, Simulated proportions   #
	#######################################################################

	if os.path.exists(ts_sim_path):
		at_least_one_exp = True
		D = get_stats(ts_sim_path, 'TutoringSystemBootstrapEmp', n_tutoring_sp, n_constraints=2)
		
		arates     = D['arate_v_n']
		arates_se  = D['arate_se_v_n']
		frates     = D['frate_v_n']
		frates_se  = D['frate_se_v_n']
		acc_v_n    = D['acc_v_n']
		acc_se_v_n = D['acc_se_v_n']
		mnames  = D['mnames']
		is_smla = D['is_smla']
		colors  = D['colors']
		nvals   = D['nvals']

		fig, (ax_acc, ax_ar, ax_fr) = plt.subplots(1, 3, figsize=(11, 1.8))
		legend_data = []
		added = []
		
		# Subplot for IW_return
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(acc_v_n.T)[::-1],(acc_se_v_n.T)[::-1]):
			line = ax_acc.plot(nvals, acc, c=c, ls=get_ls(mn))[0]
			ax_acc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
			if mn.endswith('SRL') and not('RobinHood' in added):
				added.append('RobinHood')
				legend_data.append(line)
			elif not(mn in added):
				added.append(mn)
				legend_data.append(line)
		legend_data = legend_data[::-1]
		added = added[::-1]
		ax_acc.set_xlabel('Training Samples', labelpad=1.5)
		ax_acc.text(-0.18, 0.5, 'Reward', horizontalalignment='center', verticalalignment='center', rotation=90, transform=ax_acc.transAxes)
		ax_acc.set_xscale("log")
		ax_acc.set_xlim(right=max(nvals))
		ax_acc.set_ylim((-0.5,10.5))
		ax_acc.xaxis.set_major_formatter(mtick.ScalarFormatter())

		# Subplot for accept rate, 1-Pr(NSF)
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			line = ax_ar.plot(nvals, ar*100, c=c, ls=get_ls(mn))[0]
			ax_ar.fill_between(nvals, 100*(ar+se), 100*(ar-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=1.5)
		ax_ar.set_ylabel('Solution Rate', labelpad=-3)
		ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
		ax_ar.xaxis.set_major_formatter(mtick.ScalarFormatter())

		# Subplot for failure rate Pr(~NSF ^ unsafety)
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(frates.T)[::-1], (frates_se.T)[::-1]):
			line = ax_fr.plot(nvals, fr*100, c=c, ls=get_ls(mn))[0]
			ax_fr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_fr.set_xlabel('Training Samples', labelpad=1.5)
		ax_fr.set_ylabel('Failure Rate')
		ax_fr.axhline(delta*100, color='k', linestyle=':')
		ax_fr.set_xscale("log")
		ax_fr.set_ylim((-2,102))
		ax_fr.set_xlim(right=max(nvals))
		ax_fr.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
		ax_fr.xaxis.set_major_formatter(mtick.ScalarFormatter())

		for ax in [ax_acc, ax_ar, ax_fr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(bottom=0.24, wspace=0.3, top=0.96, left=0.065, right=0.98)
		save(fig,'results_tsys_skewed.%s' % filetype, dpi=dpi)





	####################################
	#   Credit Score : Group Fairness  #
	####################################

	if os.path.exists(cs_gf_path):
		at_least_one_exp = True
		D = get_stats(cs_gf_path, 'GroupFairness',  n_credit)
		
		arates     = D['arate_v_n']
		arates_se  = D['arate_se_v_n']
		frates     = D['frate_v_n']
		frates_se  = D['frate_se_v_n']
		acc_v_n    = D['acc_v_n']
		acc_se_v_n = D['acc_se_v_n']
		mnames  = D['mnames']
		is_smla = D['is_smla']
		colors  = D['colors']
		nvals   = D['nvals']

		fig, (ax_acc, ax_ar, ax_fr) = plt.subplots(1, 3, figsize=(11, 1.8))
		legend_data = []
		added = []
		
		# Subplot for IW_return
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(acc_v_n.T)[::-1],(acc_se_v_n.T)[::-1]):
			line = ax_acc.plot(nvals, acc, c=c, ls=get_ls(mn))[0]
			ax_acc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
			if mn.endswith('SRL') and not('RobinHood' in added):
				added.append('RobinHood')
				legend_data.append(line)
			elif not(mn in added):
				added.append(mn)
				legend_data.append(line)
		legend_data = legend_data[::-1]
		added = added[::-1]
		ax_acc.set_xlabel('Training Samples', labelpad=1.5)
		ax_acc.text(-0.18, 0.5, 'Reward', horizontalalignment='center', verticalalignment='center', rotation=90, transform=ax_acc.transAxes)
		ax_acc.set_xscale("log")
		ax_acc.set_xlim(right=max(nvals))
		ax_acc.set_ylim((-1.05,1.05))
		ax_acc.xaxis.set_minor_formatter(mtick.ScalarFormatter())
		ax_acc.xaxis.set_ticklabels([])

		# Subplot for accept rate, 1-Pr(NSF)
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			line = ax_ar.plot(nvals, ar*100, c=c, ls=get_ls(mn))[0]
			ax_ar.fill_between(nvals, 100*(ar+se), 100*(ar-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=1.5)
		ax_ar.set_ylabel('Solution Rate', labelpad=-3)
		ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
		ax_ar.xaxis.set_minor_formatter(mtick.ScalarFormatter())
		ax_ar.xaxis.set_ticklabels([])

		# Subplot for failure rate Pr(~NSF ^ unsafety)
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(frates.T)[::-1], (frates_se.T)[::-1]):
			line = ax_fr.plot(nvals, fr*100, c=c, ls=get_ls(mn))[0]
			ax_fr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_fr.set_xlabel('Training Samples', labelpad=1.5)
		ax_fr.set_ylabel('Failure Rate')
		ax_fr.axhline(delta*100, color='k', linestyle=':')
		ax_fr.set_xscale("log")
		ax_fr.set_ylim((-2,102))
		ax_fr.set_xlim(right=max(nvals))
		ax_fr.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
		ax_fr.xaxis.set_minor_formatter(mtick.ScalarFormatter())
		ax_fr.xaxis.set_ticklabels([])

		for ax in [ax_acc, ax_ar, ax_fr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(bottom=0.24, wspace=0.3, top=0.96, left=0.065, right=0.98)
		save(fig,'results_loanapp_statpar.%s' % filetype, dpi=dpi)


	######################################
	#   Credit Score : Disparate Impact  #
	######################################

	if os.path.exists(cs_di_path):
		at_least_one_exp = True
		D = get_stats(cs_di_path, 'DisparateImpact',  n_credit)
		
		arates     = D['arate_v_n']
		arates_se  = D['arate_se_v_n']
		frates     = D['frate_v_n']
		frates_se  = D['frate_se_v_n']
		acc_v_n    = D['acc_v_n']
		acc_se_v_n = D['acc_se_v_n']
		mnames  = D['mnames']
		is_smla = D['is_smla']
		colors  = D['colors']
		nvals   = D['nvals']

		fig, (ax_acc, ax_ar, ax_fr) = plt.subplots(1, 3, figsize=(11, 1.8))
		legend_data = []
		added = []
		
		# Subplot for IW_return
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(acc_v_n.T)[::-1],(acc_se_v_n.T)[::-1]):
			line = ax_acc.plot(nvals, acc, c=c, ls=get_ls(mn))[0]
			ax_acc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
			if mn.endswith('SRL') and not('RobinHood' in added):
				added.append('RobinHood')
				legend_data.append(line)
			elif not(mn in added):
				added.append(mn)
				legend_data.append(line)
		legend_data = legend_data[::-1]
		added = added[::-1]
		ax_acc.set_xlabel('Training Samples', labelpad=1.5)
		ax_acc.text(-0.18, 0.5, 'Reward', horizontalalignment='center', verticalalignment='center', rotation=90, transform=ax_acc.transAxes)
		ax_acc.set_xscale("log")
		ax_acc.set_xlim(right=max(nvals))
		ax_acc.set_ylim((-1.05,1.05))
		ax_acc.xaxis.set_minor_formatter(mtick.ScalarFormatter())
		ax_acc.xaxis.set_ticklabels([])

		# Subplot for accept rate, 1-Pr(NSF)
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			line = ax_ar.plot(nvals, ar*100, c=c, ls=get_ls(mn))[0]
			ax_ar.fill_between(nvals, 100*(ar+se), 100*(ar-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=1.5)
		ax_ar.set_ylabel('Solution Rate', labelpad=-3)
		ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
		ax_ar.xaxis.set_minor_formatter(mtick.ScalarFormatter())
		ax_ar.xaxis.set_ticklabels([])

		# Subplot for failure rate Pr(~NSF ^ unsafety)
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(frates.T)[::-1], (frates_se.T)[::-1]):
			line = ax_fr.plot(nvals, fr*100, c=c, ls=get_ls(mn))[0]
			ax_fr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_fr.set_xlabel('Training Samples', labelpad=1.5)
		ax_fr.set_ylabel('Failure Rate')
		ax_fr.axhline(delta*100, color='k', linestyle=':')
		ax_fr.set_xscale("log")
		ax_fr.set_ylim((-2,102))
		ax_fr.set_xlim(right=max(nvals))
		ax_fr.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
		ax_fr.xaxis.set_minor_formatter(mtick.ScalarFormatter())
		ax_fr.xaxis.set_ticklabels([])

		for ax in [ax_acc, ax_ar, ax_fr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(bottom=0.24, wspace=0.3, top=0.96, left=0.065, right=0.98)
		save(fig,'results_loanapp_disp.%s' % filetype, dpi=dpi)


	if at_least_one_exp:
		fig = plt.figure(figsize=(9.75,0.3))
		added = [ a.replace('Naive', 'Naïve') for a in added ]
		fig.legend(legend_data, added, 'center', fancybox=True, ncol=len(legend_data), columnspacing=1, fontsize=11, handletextpad=0.5)
		save(fig, 'legend.%s' % filetype, dpi=dpi)


	###############################################
	#   Credit Score : Disparate Impract, online  #
	###############################################

	if os.path.exists(cs_di_online_path):
		base_bsln_names = ['POEM','OffsetTree','Naive', 'IntervalChaining', 'RidgeFair']
		D = get_stats(cs_di_online_path, 'DisparateImpact',  n_credit, base_bsln_names=base_bsln_names)
		
		arates     = D['arate_v_n']
		arates_se  = D['arate_se_v_n']
		frates     = D['frate_v_n']
		frates_se  = D['frate_se_v_n']
		acc_v_n    = D['acc_v_n']
		acc_se_v_n = D['acc_se_v_n']
		mnames  = D['mnames']
		is_smla = D['is_smla']
		colors  = D['colors']
		nvals   = D['nvals']

		fig, (ax_acc, ax_ar, ax_fr) = plt.subplots(1, 3, figsize=(11, 1.8))
		legend_data = []
		added = []
		
		# Subplot for IW_return
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(acc_v_n.T)[::-1],(acc_se_v_n.T)[::-1]):
			line = ax_acc.plot(nvals, acc, c=c, ls=get_ls(mn))[0]
			ax_acc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
			if mn.endswith('SRL') and not('SRL' in added):
				added.append('SRL')
				legend_data.append(line)
			elif not(mn in added):
				added.append(mn)
				legend_data.append(line)
		legend_data = legend_data[::-1]
		added = added[::-1]
		ax_acc.set_xlabel('Training Samples', labelpad=1.5)
		ax_acc.text(-0.18, 0.5, 'Reward', horizontalalignment='center', verticalalignment='center', rotation=90, transform=ax_acc.transAxes)
		ax_acc.set_xscale("log")
		ax_acc.set_ylim((-1.05,1.05))
		ax_acc.set_xlim(right=max(nvals))
		ax_acc.xaxis.set_minor_formatter(mtick.ScalarFormatter())
		ax_acc.xaxis.set_ticklabels([])

		# Subplot for accept rate, 1-Pr(NSF)
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			line = ax_ar.plot(nvals, ar*100, c=c, ls=get_ls(mn))[0]
			ax_ar.fill_between(nvals, 100*(ar+se), 100*(ar-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=1.5)
		ax_ar.set_ylabel('Solution Rate', labelpad=-3)
		ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
		ax_ar.xaxis.set_minor_formatter(mtick.ScalarFormatter())
		ax_ar.xaxis.set_ticklabels([])

		# Subplot for failure rate Pr(~NSF ^ unsafety)
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(frates.T)[::-1], (frates_se.T)[::-1]):
			line = ax_fr.plot(nvals, fr*100, c=c, ls=get_ls(mn))[0]
			ax_fr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_fr.set_xlabel('Training Samples', labelpad=1.5)
		ax_fr.set_ylabel('Failure Rate')
		ax_fr.axhline(delta*100, color='k', linestyle=':')
		ax_fr.set_xscale("log")
		ax_fr.set_ylim((-2,102))
		ax_fr.set_xlim(right=max(nvals))
		ax_fr.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
		ax_fr.xaxis.set_minor_formatter(mtick.ScalarFormatter())
		ax_fr.xaxis.set_ticklabels([])

		for ax in [ax_acc, ax_ar, ax_fr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(bottom=0.24, wspace=0.3, top=0.96, left=0.065, right=0.98)
		save(fig,'results_loanapp_disp_online.%s' % filetype, dpi=dpi)

		fig = plt.figure(figsize=(9.75,0.3))
		added = [ a.replace('Naive', 'Naïve') for a in added ]
		fig.legend(legend_data, added, 'center', fancybox=True, ncol=len(legend_data), columnspacing=1, fontsize=11, handletextpad=0.5)
		save(fig, 'online-legend.%s' % filetype, dpi=dpi)