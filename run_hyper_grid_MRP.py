######## Code structure ######
# --- simulations
# for i_rep
# for i_hyper_grid
# for reg_type
# for i_pram_grid
# ray put
# for i_pram_grid
# ray get

# print finished i_rep/n_reps
# --- results processing
# for i_hyper_grid
# for reg_type
# average the n_reps results (or how much finished)
# find best reg param
# record best_loss[reg_type][i_hyper_grid]
# record best_param[reg_type][i_hyper_grid]
########
########

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse
import timeit
import time
import ray
from copy import deepcopy
from collections import OrderedDict

from utils.mrp_utils import MRP, SetMrpArgs
from utils.learning_utils import run_value_estimation_method
from utils.planing_utils import evaluate_value_estimation
from utils.common_utils import set_random_seed, create_result_dir, save_run_data, load_run_data, write_to_log, \
	get_grid, start_ray, set_default_plot_params, save_fig


set_default_plot_params()

# -------------------------------------------------------------------------------------------
#  Run mode
# -------------------------------------------------------------------------------------------
run_mode = 'New'   #  'New'  / 'Load' / 'Continue'
result_dir_to_load = './saved/run_hyper_grid/2020_05_17_14_52_29'
save_PDF = False  # False/True - save figures as PDF file/

local_mode = False  # True/False - run non-parallel to get error messages and debugging

# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

args = argparse.Namespace()

# -----  Run Parameters ---------------------------------------------#
args.run_name = ''  # 'Name of dir to save results in (if empty, name by time)'
args.seed = 1  # random seed
args.n_reps = 10  # default 1000  # number of experiment repetitions

args.reg_types = ['gamma_guidance', 'l2_factor', 'NoReg']

args.evaluation_loss_type = 'L2_uni_weight' #  'rankings_kendalltau' | 'L2_uni_weight | 'L2' | 'one_pol_iter_l2_loss'


# -----  Hyper-grid definition ---------------------------------------------#

# args.hyper_grid_def = {'type': 'states_entropy', 'spacing': 'linspace', 'start': 0.4, 'stop': 1.0, 'num': 11,
#                             'decimals': 5} # note: below 0.4 is very slow

args.hyper_grid_def = {'type': 'states_TV_dist_from_uniform', 'spacing': 'linspace', 'start': 0.0, 'stop': 0.8,
                       'num': 9, 'decimals': 5}  # note: above 0.9 is very slow

# args.hyper_grid_def = {'type': 'Exploration_epsilon', 'spacing': 'linspace', 'start': 0.0, 'stop': 0.9, 'num': 9,
#                             'decimals': 5}

# args.hyper_grid_def = {'type': 'Connectivity_k', 'spacing': 'list', 'list': list(np.arange(1, 6, dtype=np.int))}


# ----- Search grid definition ---------------------------------------------#
args.search_grid_def = dict()

args.search_grid_def['gamma_guidance'] = {'type': 'gamma_guidance', 'spacing': 'linspace',
                                          'start': 0.7, 'stop': 0.99, 'num': 100, 'decimals': 10}

args.search_grid_def['l2_factor'] = {'type': 'l2_factor', 'spacing': 'linspace',
                                     'start': 0, 'stop': 1.2, 'num': 100, 'decimals': 10}

args.search_grid_def['NoReg'] = {'type': 'NoReg', 'spacing': 'list', 'list': [None]}

# ----- Problem Parameters ---------------------------------------------#
args.depth = 50  # default: 10  # Length of trajectory
args.gammaEval = 0.99  # default: 0.99  # gammaEval
args.n_trajectories = 2  # number of trajectories to generate per episode

# args.mrp_def = {'type': 'ToyMRP', 'p01': 0.5, 'p10': 0.5,  'reward_std': 0.1}
# args.mrp_def = {'type': 'Chain', 'p_left': 0.5, 'length': 9,  'mean_reward_range': (0, 1), 'reward_std': 0.1}
# args.mrp_def = {'type': 'RandomMDP', 'S': 100, 'A': 2, 'k': 5, 'reward_std': 0.1, 'policy': 'uniform'}
args.mrp_def = {'type': 'GridWorld', 'N0': 4, 'N1': 4, 'reward_std': 0.5, 'forward_prob_distrb': 'uniform', 'goal_reward': 1, 'R_low': -0.5, 'R_high': 0.5, 'policy': 'uniform'}

args.initial_state_distrb_type = 'uniform'  # 'uniform' | 'middle'

# ----- Algorithm Parameters ---------------------------------------------#
args.default_gamma = None  # default: None  # The default guidance discount factor (if None use gammaEval)

args.alg_type = 'batch_TD_value_evaluation'  # 'LSTD' | 'LSTD_Nested' | 'batch_TD_value_evaluation' | 'LSTD_Nested_Standard' | 'model_based_pol_eval' | 'model_based_known_P'
args.use_reward_scaling = False  # False | True.  set False for LSTD

# args.base_lstd_l2_fp = 1e-5
args.base_lstd_l2_proj = 1e-4

# if batch_TD_value_evaluation is used:
args.default_l2_TD = None  # default: None  # The default L2 factor for TD (if using discount regularization)
args.TD_Init_type = 'zero'  # How to initialize V # Options: 'Vmax' | 'zero' | 'random_0_1' |  'random_0_Vmax' | '0.5_'Vmax' |
args.n_TD_iter = 5000  # Default: 500 for RandomMDP, 5000 for GridWorld  # number of TD iterations
args.learning_rate_def = {'type': 'a/(b+i_iter)', 'a': 500, 'b': 1000, 'scale': False}


# -------------------------------------------------------------------------------------------
def set_hyper_param(args_r, hyper_grid_val):
	hyper_grid_type = args.hyper_grid_def['type']

	if hyper_grid_type == 'states_entropy':
		assert 'train_sampling_def' not in args_r
		args_r.train_sampling_def = {'type': 'Generative', 'states_entropy': hyper_grid_val}

	elif hyper_grid_type == 'states_TV_dist_from_uniform':
		assert 'train_sampling_def' not in args_r
		args_r.train_sampling_def = {'type': 'Generative', 'states_TV_dist_from_uniform': hyper_grid_val}

	else:
		raise AssertionError("Invalid args.hyper_grid_def['type']")
# -------------------------------------------------------------------------------------------


def get_regularization_params(args_r, reg_param, reg_type):
	gammaEval = args_r.gammaEval
	if args_r.default_gamma is None:
		gamma_guidance = gammaEval
	else:
		gamma_guidance = args_r.default_gamma

	alg_type = args_r.alg_type
	l2_proj = 0
	l2_fp = 0
	l2_TD = 0
	if alg_type in {'LSTD'}:
		l2_proj = args_r.base_lstd_l2_proj
	elif alg_type in {'LSTD_Nested', 'LSTD_Nested_Standard'}:
		l2_fp = args_r.base_lstd_l2_fp
		l2_proj = args_r.base_lstd_l2_proj
	elif alg_type in {'batch_TD_value_evaluation'}:
		l2_TD = args_r.default_l2_TD
	else:
		raise AssertionError

	if reg_type == 'gamma_guidance':
		gamma_guidance = reg_param

	elif reg_type == 'l2_factor':
		l2_proj += reg_param
		l2_TD = reg_param
		l2_fp += reg_param

	elif reg_type == 'NoReg':
		pass

	else:
		raise AssertionError
	return gamma_guidance, l2_TD, l2_fp, l2_proj
# -------------------------------------------------------------------------------------------


# A Ray remote function.
# Runs a single  experiment
@ray.remote
def run_exp(i_rep, args_run, reg_type, reg_param):
	# set seed
	set_random_seed(args_run.seed + i_rep)

	# Generate MDP and sampling distribution (with specified uniformity)
	M = MRP(args_run)

	gammaEval = args_run.gammaEval

	# set regularisation parameters
	gamma_guidance, l2_TD, l2_fp, l2_proj = get_regularization_params(args_run, reg_param, reg_type)

	# Generate data:
	data = M.SampleDataMrp(args_run)

	V_est, V_true = run_value_estimation_method(data, M, args_run, gamma_guidance, l2_proj, l2_fp, l2_TD)

	loss_type = args_run.evaluation_loss_type
	pi = None
	eval_loss = evaluate_value_estimation(loss_type, V_true, V_est, M, pi, gammaEval, gamma_guidance)

	return eval_loss

# -------------------------------------------------------------------------------------------

def run_simulation(args, hyper_grid_vals, loss, reg_grids, local_mode):
	start_ray(local_mode)
	write_to_log('local_mode == {}'.format(local_mode), args)
	SetMrpArgs(args)
	start_time = timeit.default_timer()
	set_random_seed(args.seed)

	reg_types = args.reg_types

	n_hyper_grid = len(hyper_grid_vals)
	n_reps = args.n_reps
	results_dict = dict()

	write_to_log('***** Starting  {} reps'.format(n_reps), args)
	for i_rep in range(n_reps):

		for i_hyper_grid, hyper_grid_val in enumerate(hyper_grid_vals):
			args_run = deepcopy(args)
			set_hyper_param(args_run, hyper_grid_val)

			# send jobs:
			out_ids = {reg_type: [None for _ in range(len(reg_grids[reg_type]))] for reg_type in reg_types}
			for reg_type in reg_types:

				for i_reg_pram, reg_param in enumerate(reg_grids[reg_type]):
					# ray put
					if np.isnan(loss[reg_type][i_hyper_grid, i_reg_pram, i_rep]):
						out_ids[reg_type][i_reg_pram] = run_exp.remote(i_rep, args_run, reg_type, reg_param)
				# end if
			# end for i_reg_pram
			# end for reg_type

			# Gather results:
			for reg_type in reg_types:
				for i_reg_pram, reg_param in enumerate(reg_grids[reg_type]):
					# ray get
					if out_ids[reg_type][i_reg_pram] is not None:
						out = ray.get(out_ids[reg_type][i_reg_pram])
						loss[reg_type][i_hyper_grid, i_reg_pram, i_rep] = out
				# end if
			# end for i_reg_pram
		# end for reg_type
		# end for i_hyper_grid

		# Save results so far
		results_dict = {'hyper_grid_vals': hyper_grid_vals, 'loss': loss, 'reg_grids': reg_grids, 'n_reps_finished': i_rep + 1}
		save_run_data(args, results_dict, verbose=0)
		time_str = time.strftime("%H hours, %M minutes and %S seconds",  time.gmtime(timeit.default_timer() - start_time))
		write_to_log('Finished: {} out of {} reps, time: {}'.format(i_rep + 1, n_reps, time_str), args)

	# end for i_rep

	stop_time = timeit.default_timer()
	write_to_log('Total runtime: ' + time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)), args)
	return results_dict


# end  run_simulations

# -------------------------------------------------------------------------------------------

if __name__ == "__main__":
	# *********************************
	if run_mode == 'Load':
		args, results_dict = load_run_data(result_dir_to_load)
	# *********************************
	elif run_mode == 'New':

		hyper_grid_vals = get_grid(args.hyper_grid_def)
		create_result_dir(args)
		n_hyper_grid = len(hyper_grid_vals)
		n_reps = args.n_reps
		reg_types = args.reg_types

		# define search grids for regularization parameters
		reg_grids = dict()
		for reg_type in reg_types:
			reg_param_grid_def = args.search_grid_def[reg_type]
			reg_grids[reg_type] = get_grid(reg_param_grid_def)

		# init result matrix with nan (no result)
		loss = {reg_type: np.full((n_hyper_grid, len(reg_grids[reg_type]), n_reps), np.nan) for reg_type in reg_types}

		# run
		results_dict = run_simulation(args, hyper_grid_vals, loss, reg_grids, local_mode)
		save_run_data(args, results_dict)
	# *********************************
	elif run_mode == 'Continue':

		loaded_args, loaded_results_dict = load_run_data(result_dir_to_load)
		args = loaded_args
		args.result_dir = result_dir_to_load  # update the path, in case the result folder moved
		hyper_grid_vals = loaded_results_dict['hyper_grid_vals']
		loss = loaded_results_dict['loss']
		reg_grids = loaded_results_dict['reg_grids']
		results_dict = run_simulation(args, hyper_grid_vals, loss, reg_grids, local_mode)
		save_run_data(args, results_dict)
	# *********************************
	else:
		raise AssertionError('Unrecognized run_mode')
	# *********************************

	hyper_grid_vals = results_dict['hyper_grid_vals']
	loss = results_dict['loss']
	reg_grids = results_dict['reg_grids']
	n_reps_finished = results_dict['n_reps_finished']
	n_hyper_grid = len(hyper_grid_vals)
	reg_types = args.reg_types

	# get results statistics
	best_reg_param = {reg_type: np.full((n_hyper_grid), np.nan) for reg_type in reg_types}
	best_loss_mean = {reg_type: np.full((n_hyper_grid), np.nan) for reg_type in reg_types}
	best_loss_std = {reg_type: np.full((n_hyper_grid), np.nan) for reg_type in reg_types}

	# get results statistics
	for reg_type in reg_types:

		for i_hyper_grid, hyper_grid_val in enumerate(hyper_grid_vals):
			# average results over reps:
			loss_curr = loss[reg_type][i_hyper_grid, :, :n_reps_finished]
			loss_m = np.mean(loss_curr, axis=1)

			# Mark the best reg pram for current uniformity val:
			i_best = loss_m.argmin(axis=0)

			best_reg_param[reg_type][i_hyper_grid] = reg_grids[reg_type][i_best]
			best_loss_mean[reg_type][i_hyper_grid] = loss_curr[i_best].mean()
			best_loss_std[reg_type][i_hyper_grid] = loss_curr[i_best].std()

	# end or i_hyper_grid
	# end for reg_type

	SetMrpArgs(args)

	reg_labels = OrderedDict([('NoReg', 'No Regularization'),
	                          ('gamma_guidance', 'Best Discount Regularization'),
	                          ('l2_factor', 'Best L2 Regularization')])

	grid_type = args.hyper_grid_def['type']
	xscale = 1.
	is_int_grid = False
	title_prefix = args.mdp_def['type'] + ' - ' + grid_type
	grid_label = 'Hyper-parameter'
	if grid_type in {'states_entropy', 'states_actions_entropy'}:
		grid_label = r'Entropy [normalized]'
	elif grid_type in {'states_TV_dist_from_uniform', 'states_actions_TV_dist_from_uniform'}:
		grid_label = r'Total-Variation from uniform [normalized]'
	elif grid_type == 'Exploration_epsilon':
		grid_label = r'Exploration parameter $epsilon$'
	elif grid_type == 'Connectivity_k':
		grid_label = r'Connectivity parameter $k$'
		hyper_grid_vals = [int(k) for k in hyper_grid_vals]
		is_int_grid = True

	# plot best reg params
	fig, ax = plt.subplots(2, 1)
	ax[0].errorbar(xscale * hyper_grid_vals, best_reg_param['gamma_guidance'])
	ax[0].grid(True)
	ax[0].set_xlabel(grid_label, fontsize=10)
	ax[0].set_ylabel('Best discount parameter', fontsize=10)
	ax[1].errorbar(xscale * hyper_grid_vals, 1e3 * best_reg_param['l2_factor'])
	ax[1].grid(True)
	ax[1].set_xlabel(grid_label, fontsize=10)
	ax[1].set_ylabel('Best L2 factor [1e3]', fontsize=10)
	if is_int_grid:
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))

	# plot loss
	ax = plt.figure().gca()
	ci_factor = 1.96 / np.sqrt(n_reps_finished)  # 95% confidence interval factor
	for reg_type in reg_labels.keys():
		plt.errorbar(xscale * hyper_grid_vals, best_loss_mean[reg_type], best_loss_std[reg_type] * ci_factor, label=reg_labels[reg_type])
	plt.grid(True)
	plt.legend(fontsize=13)
	plt.xlabel(grid_label)
	plt.ylabel('Loss')
	if is_int_grid:
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))

	if save_PDF:
		save_fig(args.run_name, base_path=args.result_dir)
		#
		# # plot loss gain
		# ax = plt.figure().gca()
		# ci_factor = 1.96 / np.sqrt(n_reps_finished)  # 95% confidence interval factor
		# for reg_type in reg_labels.keys():
		# 	if reg_type != 'NoReg':
		# 		plt.errorbar(xscale * hyper_grid_vals, best_loss_mean['NoReg'] - best_loss_mean[reg_type], (best_loss_std[reg_type] + best_loss_std['NoReg'])* ci_factor,
		# 					 label=reg_labels[reg_type])
		# 		# TODO:  more accurate  std
		# plt.grid(True)
		# plt.legend()
		# plt.xlabel(grid_label)
		# plt.ylabel('Loss Gain')
		# if  is_int_grid:
		# 	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		# if save_PDF:
		# 	save_fig(args.run_name)
	else:
		plt.title(title_prefix + ' \n ' + args.result_dir + ', reps_finished: ' + str(n_reps_finished), fontsize=8)
	# + 'Episode Reward Mean +- 95% CI, ' + ' \n '

	plt.show()
	print('done')
