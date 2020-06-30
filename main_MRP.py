

import numpy as np
import matplotlib.pyplot as plt
import argparse
import timeit
import time
import ray
from copy import deepcopy

from utils.mrp_utils import MRP, SetMrpArgs, calc_typical_mixing_time
from utils.planing_utils import evaluate_value_estimation
from utils.learning_utils import run_value_estimation_method
from utils.common_utils import set_random_seed, create_result_dir, save_run_data, load_run_data, write_to_log, get_grid, start_ray, set_default_plot_params, save_fig, pretty_print_args

set_default_plot_params()

# -------------------------------------------------------------------------------------------
#  Run mode
# -------------------------------------------------------------------------------------------

local_mode = False  # True/False - run non-parallel to get error messages and debugging
# Option to load previous run results:
load_run = False  # False/True If true just load results from dir, o.w run simulationz
result_dir_to_load = './saved/main_MRP/2020_06_16_15_31_31'
# result_dir_to_load = './saved/Multi_Exp/2020_05_30_07_40_26/Exp_6_LSTD_Discount_MixTime'
# result_dir_to_load = './saved/Tabular/2020_06_16_00_52_40_PolEval_LSTD_L2Loss_L2Reg'

# Plot options:
save_PDF = False  # False/True - save figures as PDF file in the result folder
y_lim = [6,16]  # [75, 85] | None
legend_loc = 'best' # | 'best'| 'upper left'
show_stars = True
# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

args = argparse.Namespace()

# ----- Run Parameters ---------------------------------------------#
args.run_name = ''   # 'Name of dir to save results in (if empty, name by time)'
args.seed = 1  # random seed
args.n_reps = 1000  # default 5000  # number of experiment repetitions

#  how to create parameter grid:
# args.param_grid_def = {'type': 'gamma_guidance', 'spacing': 'linspace', 'start': 0.4, 'stop': 0.99, 'num': 50, 'decimals': 10}
# args.param_grid_def = {'type': 'l2_factor', 'spacing': 'linspace', 'start': 0., 'stop': 0.9, 'num': 50, 'decimals': 10}
# args.param_grid_def = {'type': 'l2_fp', 'spacing': 'linspace', 'start': 0.0001, 'stop': 0.1, 'num': 50, 'decimals': 10} # $L_2$ Regularization Factor  - Fixed-Point Phase
# args.param_grid_ def = {'type': 'l2_proj', 'spacing': 'linspace', 'start': 0.0001, 'stop': 0.001, 'num': 20, 'decimals': 10} $L_2$ Regularization Factor  - Projection Phase
# args.param_grid_def = {'type': 'n_trajectories', 'spacing': 'arange', 'start': 1, 'stop': 21}
# args.param_grid_def = {'type': 'depth', 'spacing': 'arange', 'start': 1, 'stop': 10}
args.param_grid_def = {'type': 'states_TV_dist_from_uniform', 'spacing': 'list', 'list': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
#
# ----- Problem Parameters ---------------------------------------------#


# args.mrp_def = {'type': 'ToyMRP', 'p01': 0.5, 'p10': 0.5,  'reward_std': 0.1}
# args.mrp_def = {'type': 'Chain', 'p_left': 0.5, 'length': 9,  'mean_reward_range': (0, 1), 'reward_std': 0.1}
# args.mrp_def = {'type': 'RandomMDP', 'S': 100, 'A': 2, 'k': 5, 'reward_std': 0.1, 'policy': 'uniform'}
args.mrp_def = {'type': 'GridWorld', 'N0': 4, 'N1': 4, 'reward_std': 0.5, 'forward_prob_distrb': 'uniform', 'goal_reward': 1, 'R_low': -0.5, 'R_high': 0.5, 'policy': 'uniform'}

# args.config_grid_def = {'type': 'p_left', 'spacing': 'list', 'list': [0.3, 0.5, 0.7, 0.9]} # for mrp_def['type'] == 'Chain':
# args.config_grid_def = {'type': 'forced_mix_time', 'spacing': 'list', 'list': [1.5, 3, 6, 12]} # must be above  1.0
# args.config_grid_def = {'type': 'n_trajectories', 'spacing': 'list', 'list': [1, 2, 4, 8]}
# args.config_grid_def = {'type': 'trajectory_len', 'spacing': 'list', 'list': [10, 20, 30]}
# args.config_grid_def = {'type': 'states_TV_dist_from_uniform', 'spacing': 'list', 'list': [0, 0.2, 0.4, 0.6, 0.8]}   # note: above 0.9 is very slow
# args.config_grid_def = {'type': 'None', 'spacing': 'list', 'list': [None]}
args.config_grid_def = {'type': 'RegMethod', 'spacing': 'list_tuples',  'list':[ ('None', None), ('Discount Reg.', 0.98), ('L2 Reg.', 0.17)]} #

args.depth = 50  # default: 10 for 'chain', 100 for 'GridWorld'  # Length of trajectory
args.gammaEval = 0.99   # default: 0.99  # gammaEval

args.train_sampling_def = {'type': 'Trajectories'}
# args.train_sampling_def = {'type': 'Generative_uniform'}
# args.train_sampling_def = {'type': 'sample_all_s'}



args.evaluation_loss_type = 'L2_uni_weight' #  'rankings_kendalltau' | 'L2_uni_weight | 'L2' | 'one_pol_iter_l2_loss'

args.initial_state_distrb_type = 'uniform'  # 'uniform' | 'middle'
args.n_trajectories = 8  #

# ----- Algorithm Parameters ---------------------------------------------#
args.default_gamma = None  # default: None  # The default guidance discount factor (if None use gammaEval)

args.alg_type = 'LSTD'  # 'LSTD' | 'LSTD_Nested' | 'batch_TD_value_evaluation' | 'LSTD_Nested_Standard' | 'model_based_pol_eval' | 'model_based_known_P'
args.use_reward_scaling = False  # False | True.  set False for LSTD

# args.base_lstd_l2_fp = 1e-5
args.base_lstd_l2_proj = 1e-4

# if batch_TD_value_evaluation is used:
args.default_l2_TD = None  # default: None  # The default L2 factor for TD (if using discount regularization)
# args.TD_Init_type = 'zero'  # How to initialize V # Options: 'Vmax' | 'zero' | 'random_0_1' |  'random_0_Vmax' | '0.5_'Vmax' |
# args.n_TD_iter = 5000  # Default: 500 for RandomMDP, 5000 for GridWorld  # number of TD iterations
# args.learning_rate_def = {'type': 'a/(b+i_iter)', 'a': 500, 'b': 1000, 'scale': False}
# -------------------------------------------------------------------------------------------

def set_config(args, config_val):
	config_type = args.config_grid_def['type']
	if config_type == 'n_trajectories':
		args.n_trajectories = config_val

	elif config_type == 'trajectory_len':
		args.depth = config_val

	elif config_type == 'p_left':
		args.mrp_def['p_left'] = config_val

	elif config_type == 'forced_mix_time':
		args.forced_mix_time = config_val

	elif config_type == 'states_TV_dist_from_uniform':
		args.train_sampling_def = {'type': 'Generative', 'states_TV_dist_from_uniform': config_val}

	elif config_type == 'None':
		pass

	elif config_type == 'RegMethod':
		reg_str = config_val[0]
		reg_val = config_val[1]
		if reg_str == 'Discount Reg.':
			args.default_gamma = reg_val
		elif reg_str == 'L2 Reg.':
			args.default_l2_TD = reg_val
		elif reg_str == 'None':
			pass
		else:
			raise AssertionError
		# end if
	else:
		raise AssertionError('Unrecognized config_grid_def type')
	# end if

	return args
# # -------------------------------------------------------------------------------------------


def set_params(args_r, param_val):
	gammaEval = args_r.gammaEval
	n_trajectories = args_r.n_trajectories
	if args_r.default_gamma is None:
		gamma_guidance = gammaEval
	else:
		gamma_guidance = args_r.default_gamma
	l2_TD = args_r.default_l2_TD

	alg_type = args_r.alg_type
	l2_proj = 0
	l2_fp = 0
	if alg_type in {'LSTD'}:
		l2_proj = args_r.base_lstd_l2_proj
	elif alg_type in {'LSTD_Nested', 'LSTD_Nested_Standard'}:
		l2_fp = args_r.base_lstd_l2_fp
		l2_proj = args_r.base_lstd_l2_proj
	elif alg_type in {'batch_TD_value_evaluation'}:
		l2_TD = args_r.default_l2_TD
	else:
		raise AssertionError
	regenerate_mrp = False

	grid_type = args_r.param_grid_def['type']

	if grid_type == 'gamma_guidance':
		gamma_guidance = param_val

	elif grid_type == 'l2_factor':
		l2_proj += param_val
		l2_TD = param_val
		l2_fp += param_val

	elif grid_type == 'l2_proj':
		l2_proj = args_r.base_lstd_l2_proj + param_val

	elif grid_type == 'l2_fp':
		l2_fp = args_r.base_lstd_l2_fp + param_val

	elif grid_type == 'n_trajectories':
		args_r.n_trajectories = param_val

	elif grid_type == 'states_TV_dist_from_uniform':
		regenerate_mrp = True
		args_r.train_sampling_def = {'type': 'Generative', 'states_TV_dist_from_uniform': param_val}

	elif grid_type == 'NoReg':
		pass
	# elif args.param_grid_def['type'] == 'depth':
	# 	depth = param_val
	else:
		raise AssertionError
	return regenerate_mrp, gamma_guidance, l2_TD, l2_fp, l2_proj
# # -------------------------------------------------------------------------------------------


# A Ray remote function.
# runs a single repetition of the experiment
@ray.remote  # (num_cpus=0.2)  # specify how much resources the process needs
def run_rep(i_rep, param_val_grid, config_grid, args):
	nS = args.nS

	n_grid = param_val_grid.shape[0]
	n_configs = args.n_configs
	loss_rep = np.zeros((n_configs, n_grid))

	# default values
	gammaEval = args.gammaEval

	for i_config, config_val in enumerate(config_grid):  # grid of n_configs
		args = set_config(args, config_val)

		# Generate MRP:
		M = MRP(args)

		for i_grid, param_val in enumerate(param_val_grid):
			# grid values:
			regenerate_mrp, gamma_guidance, l2_TD, l2_fp, l2_proj  = set_params(args, param_val)

			if regenerate_mrp:
				M = MRP(args)

			# Generate data:
			data = M.SampleDataMrp(args)

			V_est, V_true = run_value_estimation_method(data, M, args, gamma_guidance, l2_proj, l2_fp, l2_TD)

			loss_type = args.evaluation_loss_type
			pi = None
			eval_loss = evaluate_value_estimation(loss_type, V_true, V_est, M, pi, gammaEval, gamma_guidance)
			loss_rep[i_config, i_grid] = eval_loss
		# end for i_grid
	#  end for i_config
	return loss_rep
# end run_rep
# -------------------------------------------------------------------------------------------


def run_simulations(args, save_result, local_mode):
	args_def = deepcopy(args)
	start_ray(local_mode)
	if save_result:
		create_result_dir(args)
		write_to_log('local_mode == {}'.format(local_mode), args)

	start_time = timeit.default_timer()
	set_random_seed(args.seed)

	n_reps = args.n_reps
	param_val_grid = get_grid(args.param_grid_def)
	n_grid = param_val_grid.shape[0]

	config_grid = get_grid(args.config_grid_def)
	n_configs = len(config_grid)
	args.n_configs = n_configs

	loss_mat = np.zeros((n_reps, n_configs, n_grid))

	# ----- Run simulation in parrnell process---------------------------------------------#
	loss_rep_id_lst = []
	for i_rep in range(n_reps):
		# returns objects ids:
		loss_mat_rep_id = run_rep.remote(i_rep, param_val_grid, config_grid, args)
		loss_rep_id_lst.append(loss_mat_rep_id)
	# -----  get the results --------------------------------------------#
	for i_rep in range(n_reps):
		loss_rep = ray.get(loss_rep_id_lst[i_rep])
		write_to_log('Finished: {} out of {} reps'.format(i_rep + 1, n_reps), args)
		loss_mat[i_rep] = loss_rep
	# end for i_rep
	info_dict = {'loss_avg': loss_mat.mean(axis=0), 'loss_std': loss_mat.std(axis=0),
				 'param_val_grid': param_val_grid, 'config_grid': config_grid}
	if save_result:
		save_run_data(args, info_dict)
	stop_time = timeit.default_timer()
	write_to_log('Total runtime: ' +
				 time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)), args)
	write_to_log(['-'*10 +'Defined args: ', pretty_print_args(args_def), '-'*20], args)
	return info_dict
# end  run_simulations
# -------------------------------------------------------------------------------------------


def run_main_mrp(args, save_result=True,  local_mode=local_mode, load_run_data_flag=False, result_dir_to_load='',
                 save_PDF=False, plot=True, show_stars=True, y_lim=None, legend_loc='best'):
	SetMrpArgs(args)
	if load_run_data_flag:
		args, info_dict = load_run_data(result_dir_to_load)
	else:
		info_dict = run_simulations(args, save_result, local_mode)
	if 'loss_avg' in info_dict:
		loss_avg = info_dict['loss_avg']
		loss_std = info_dict['loss_std']
	else:
		loss_avg = info_dict['planing_loss_avg']
		loss_std = info_dict['planing_loss_std']
	if 'param_val_grid' in info_dict:
		param_val_grid = info_dict['param_val_grid']
	else:
		param_val_grid = info_dict['alg_param_grid']
	config_grid =  info_dict['config_grid']
	n_reps = args.n_reps

	# ----- Plot figures  ---------------------------------------------#
	if plot or save_PDF:
		ax = plt.figure().gca()
		if args.train_sampling_def['type'] in {'Generative', 'Generative_uniform', 'Generative_Stationary'}:
			data_size_per_traj = args.depth
		elif args.train_sampling_def['type'] == 'Trajectories':
			data_size_per_traj = args.depth
		elif args.train_sampling_def['type'] == 'sample_all_s':
			data_size_per_traj = args.nS
		else:
			raise AssertionError
		# end if
		xscale = 1.
		x_label = ''
		legend_title = ''
		grid_type = args.param_grid_def['type']
		if grid_type == 'gamma_guidance':
			x_label = r'Guidance Discount Factor $\gamma$'
		elif grid_type == 'l2_TD':
			x_label = r'$L_2$ Regularization Factor [1e-3]'
			xscale = 1e3
		elif grid_type == 'l2_proj':
			x_label = r'$L_2$ Regularization Factor  - Projection Phase'
		elif grid_type == 'l2_fp':
			x_label = r'$L_2$ Regularization Factor  - Fixed-Point Phase'
		elif grid_type == 'l2_TD':
			x_label = r'$L_2$ Regularization Factor'
		elif grid_type == 'l2_factor':
			x_label = r'$L_2$ Regularization Factor'
		elif grid_type == 'n_trajectories':
			if args.train_sampling_def['type'] in {'Generative', 'Generative_uniform'} \
					or args.config_grid_def['type'] == 'states_TV_dist_from_uniform':
				xscale = args.depth
				x_label = 'Num. Samples'
			else:
				x_label = 'Num. Trajectories'
			# end if
		elif grid_type == 'states_TV_dist_from_uniform':
			x_label = 'Total-Variation from\n uniform [normalized]'
		# end if
		ci_factor = 1.96 / np.sqrt(n_reps)  # 95% confidence interval factor
		# mixing_time_labels = ['Fast mixing', 'Moderate mixing',  'Slow mixing']
		for i_config, config_val in enumerate(config_grid): # for of plots in the figure

			config_type = args.config_grid_def['type']

			if config_type == 'p_left':
				args.mrp_def['p_left'] = config_val
				M = MRP(args)
				mixing_time = np.around(calc_typical_mixing_time(M.P), decimals=3)
				label_str = r'$\tau$={}'.format(mixing_time)
				print(label_str + r': $p_l=${} , $1/(1-|\lambda_2|)${}'.format(config_val, mixing_time))

			elif config_type == 'forced_mix_time':
				label_str = '{}'.format(config_val)
				legend_title = 'Mixing-time'

			elif config_type == 'n_trajectories':
				if args.train_sampling_def['type'] in {'Generative_uniform', 'Generative', 'Generative_Stationary'}:
					legend_title = 'Num. Samples'
					label_str = '{} '.format(config_val * data_size_per_traj)
				else:
					legend_title = 'Num. Trajectories'
					label_str = '{} '.format(config_val)

			elif config_type == 'states_TV_dist_from_uniform':
				legend_title = 'Total-Variation from\n uniform [normalized]'
				label_str = '{} '.format(config_val)

			elif config_type == 'None':
				legend_title = None
				label_str = ''

			elif config_type == 'RegMethod':
				legend_title = 'Regularizer'
				label_str = str(config_val[0]) + ': '+ str(config_val[1])


			else:
				raise AssertionError


			plt.errorbar(xscale * param_val_grid, loss_avg[i_config], yerr=loss_std[i_config] * ci_factor,
						 marker='.', label=label_str)

			if show_stars:
				# Mark the lowest point:
				i_best = np.argmin(loss_avg[i_config])
				plt.scatter(xscale * param_val_grid[i_best], loss_avg[i_config][i_best], marker='*', s=400)
				print(label_str + ' Best x-coord: ' + str(xscale *  param_val_grid[i_best]) )
			# end if
		# for i_config

		plt.grid(True)
		plt.ylabel(r'Loss')
		plt.xlabel(x_label)
		if args.evaluation_loss_type == 'L2':
			plt.ylabel(r'$L_2$ Loss')
		if args.evaluation_loss_type == 'L2_uni_weight':
			plt.ylabel(r'Avg. $L_{2}$ Loss')
		elif args.evaluation_loss_type == 'rankings_kendalltau':
			plt.ylabel(r'Ranking Loss')
		if legend_title is not None:
			plt.legend(title=legend_title, loc=legend_loc, fontsize=12, title_fontsize=12)  # loc='upper right'
		# plt.xlim([0.95,0.99])

		if y_lim:
			plt.ylim(y_lim)
		# ax.set_yticks(np.arange(0., 9., step=1.))

		if save_PDF:
			save_fig(args.run_name)
		else:
			# plt.title('Loss +- 95% CI \n ' + str(args.args))
			plt.title(args.mdp_def['type'] + '  ' + args.run_name + ' \n ' + args.result_dir, fontsize=6)
		# end if save_PDF
	# end if
	pretty_print_args(args)
	if plot:
		plt.show()
	info_dict['result_dir'] = args.result_dir
	return info_dict
# end  run_main
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
	info_dict = run_main_mrp(args, save_result=True, local_mode=local_mode, load_run_data_flag=load_run,
	                         result_dir_to_load=result_dir_to_load, save_PDF=save_PDF, y_lim=y_lim, legend_loc=legend_loc)
	# write_to_log(['Defined args: ', pretty_print_args(args_def)], info_dict['args'])
	print('done')




