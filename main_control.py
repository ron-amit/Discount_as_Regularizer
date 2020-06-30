"""
In  tabular MDP setting, evaluates the learning of optimal policy using different guidance discount factors
On-policy means we run episodes,
 in each episode we generate roll-outs/trajectories of current policy and run algorithm to improve the policy.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import timeit
import time
import ray
from copy import deepcopy
from utils.mdp_utils import MDP, SetMdpArgs
from utils.planing_utils import PolicyEvaluation, PolicyIteration, get_stationary_distrb, GetUniformPolicy
from utils.learning_utils import run_learning_method
from utils.common_utils import set_random_seed, create_result_dir, save_run_data, load_run_data, write_to_log, get_grid, \
	start_ray, set_default_plot_params, save_fig, pretty_print_args

set_default_plot_params()

# -------------------------------------------------------------------------------------------
#  Run mode
# -------------------------------------------------------------------------------------------

local_mode = False  # True/False - run non-parallel to get error messages and debugging
# Option to load previous run results:
load_run = False  # False/True If true just load results from dir, o.w run simulation
# result_dir_to_load = './saved/Multi_Exp/2020_05_30_07_40_26/Exp_5_LSTDQ_Discount_TV_dist'
result_dir_to_load = './saved/main_control/2020_06_28_11_59_58/'


# Plot options:
save_PDF = False  # False/True - save figures as PDF file in the result folder
y_lim = None # [75, 85] | None
show_stars = True
# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

args = argparse.Namespace()

# -----  Run Parameters ---------------------------------------------#

args.run_name = ''  # 'Name of dir to save results in (if empty, name by time)'
args.seed = 1  # random seed

args.n_reps = 1000  # default 1000  # number of experiment repetitions

#  how to create parameter grid:
# args.param_grid_def = {'type': 'gamma_guidance', 'spacing': 'linspace', 'start': 0.5, 'stop': 0.99, 'num': 50, 'decimals': 10}
args.param_grid_def = {'type': 'L2_factor', 'spacing': 'linspace', 'start': 1e-4, 'stop': 0.01, 'num': 50, 'decimals': 10}


args.config_grid_def = {'type': 'n_trajectories', 'spacing': 'list', 'list': [4, 8, 16, 32]}  # grid of number of trajectories to generate per episode  Default:  [1, 2, 4, 8]
# args.config_grid_def = {'type': 'states_actions_TV_dist_from_uniform', 'spacing': 'list', 'list': [0, 0.3, 0.6, 0.9]}   # note: above 0.9 is very slow
# args.config_grid_def = {'type': 'chain_mix_time', 'spacing': 'list', 'list': [5, 10, 20, 40, 80]}  # must be above  1.0,  generates states and actions from a Markov chain with some mixing time
# args.config_grid_def = {'type': 'n_episodes', 'spacing': 'list', 'list': [1, 2, 3, 4]}
# args.config_grid_def = {'type': 'None', 'spacing': 'list', 'list': [None]}

# ----- Problem Parameters ---------------------------------------------#
# MDP definition ( see data_utils.SetMdpArgs)
# args.mdp_def = {'type': 'RandomMDP', 'S': 10, 'A': 5, 'k': 2, 'reward_std': 0.1}
args.mdp_def = {'type': 'GridWorld', 'N0': 4, 'N1': 4, 'reward_std': 0.1, 'forward_prob_distrb': 'uniform',  'goal_reward': 1, 'R_low': -0.5, 'R_high': 0.5}
# args.mdp_def = {'type': 'GridWorld', 'N0': 4, 'N1': 4, 'reward_std': 0.1, 'forward_prob_distrb': {'alpha': 3, 'beta': 1}, 'goal_reward': 1}


args.depth = 10  # default: 10  # Length of trajectory
args.n_trajectories = 4  # default value
args.gammaEval = 0.99  # default: 0.99  # gammaEval
args.n_episodes = 5  # Number of episodes

args.train_sampling_def = {'type': 'Trajectories'}  # n_traj  trajectories of length args.depth
# args.train_sampling_def = {'type': 'Generative_uniform'}  # n_samples =  args.depth X n_traj
# args.train_sampling_def = {'type': 'Generative_Stationary'}  #  n_samples =  args.depth X n_traj
# args.train_sampling_def = {'type': 'sample_all_s_a'}  # n_samples per (s,a) =  n_traj
# args.train_sampling_def = {'type': 'Generative', 'states_TV_dist_from_uniform': 0, 'actions_TV_dist_from_uniform': 0}  # n_samples =  args.depth X n_traj
# args.train_sampling_def = {'type': 'Generative', 'states_dist_from_uniform': 0.7, 'actions_dist_from_uniform': 0.7}  # n_samples =  args.depth X n_traj
# args.train_sampling_def = {'type': 'Generative', 'states_entropy': 1.5, 'actions_entropy': 1.0}  # n_samples =  args.depth X n_traj

# ----- Algorithm Parameters ---------------------------------------------#
args.initial_policy = 'uniform'  # 'uniform' (default) | 'generated_random'
args.epsilon = 0.1  # for epsilon-greedy
args.default_gamma = None  # default: None  # The default guidance discount factor (if None use gammaEval)
args.default_l2_factor = 1e-4  # default: None  # The default L2 factor (if using discount regularization) - note: it is necessary for LSTD
args.method = 'SARSA'  # 'Policy-Evaluation Algorithm'  # Options: 'Expected_SARSA'  | 'Model_Based' | 'SARSA' | 'LSTDQ' | 'ELSTDQ' | 'ELSTDQ_nested' | 'LSTDQ_nested'
args.use_reward_scaling = False

# Hyper-parameters for iterative methods ('SARSA', 'Expected_SARSA' )
args.TD_Init_type = 'zero'  # How to initialize V # Options: 'Vmax' | 'zero' | 'random_0_1' |  'random_0_Vmax' | '0.5_'Vmax' |
args.n_TD_iter = 5000    # Default: 500 for RandomMDP, 5000 for GridWorld  # number of TD iterations
args.learning_rate_def = {'type': 'a/(b+i_iter)', 'a': 500, 'b': 1000, 'scale': True}

# -------------------------------------------------------------------------------------------


# A Ray remote function.
# Runs a single repetition of the experiment
@ray.remote
def run_rep(i_rep, alg_param_grid, config_grid_def, args_r):

	set_random_seed(args_r.seed + i_rep)
	config_grid_vals = get_grid(config_grid_def)
	n_config_grid = len(config_grid_vals)
	n_grid = len(alg_param_grid)
	# runs a single repetition of the experiment
	loss_rep = np.zeros((n_config_grid, n_grid))
	gammaEval = args_r.gammaEval
	config_type = config_grid_def['type']

	# grid of number of trajectories to generate
	for i_config, config_val in enumerate(config_grid_vals):
		n_traj = args_r.n_trajectories
		if config_type == 'n_trajectories':
			n_traj = config_val
		elif config_type == 'states_actions_TV_dist_from_uniform':
			args_r.train_sampling_def = {'type': 'Generative', 'states_TV_dist_from_uniform': config_val, 'actions_TV_dist_from_uniform': config_val}
		elif config_type == 'chain_mix_time':
			args_r.train_sampling_def = {'type': 'chain_mix_time', 'mix_time': config_val}
		elif config_type == 'n_episodes':
			args_r.n_episodes = config_val
		elif config_type == 'None':
			pass
		else:
			raise AssertionError

		# Generate MDP:
		M = MDP(args_r)

		# Optimal policy for the MDP:
		pi_opt, V_opt, Q_opt = PolicyIteration(M, gammaEval)

		# grid of regularization param
		for i_grid, alg_param in enumerate(alg_param_grid):
				gamma_guidance, l1_factor, l2_factor = get_regularization_params(args_r, alg_param,
				                                                                 args_r.param_grid_def['type'])

				# run the learning episodes:
				pi_t = run_learning_method(args_r, M, n_traj, gamma_guidance, l2_factor, l1_factor)

				# Evaluate performance of learned policy:
				V_t, _ = PolicyEvaluation(M, pi_t, gammaEval)

				loss_rep[i_config, i_grid] = (np.abs(V_opt - V_t)).mean()
		# end for grid
	#  end for i_config
	return loss_rep


# end run_rep
# -------------------------------------------------------------------------------------------


def get_regularization_params(args_r, alg_param, reg_type):
	gammaEval = args_r.gammaEval
	if args_r.default_gamma is None:
		gamma_guidance = gammaEval
	else:
		gamma_guidance = args_r.default_gamma
	l2_factor = args.default_l2_factor
	l1_factor = None

	if reg_type in 'L2_factor':
		l2_factor = alg_param
	elif reg_type == 'L1_factor':
		l1_factor = alg_param
	elif reg_type == 'gamma_guidance':
		gamma_guidance = alg_param
	elif reg_type == 'NoReg':
		pass
	else:
		raise AssertionError('Unrecognized regularization type: ' + str(reg_type))
	return gamma_guidance, l1_factor, l2_factor


# -------------------------------------------------------------------------------------------


def run_simulations(args, save_result, local_mode, init_ray=True):
	if init_ray:
		start_ray(local_mode)
	if save_result:
		create_result_dir(args)
		write_to_log('local_mode == {}'.format(local_mode), args)

	start_time = timeit.default_timer()
	set_random_seed(args.seed)

	n_reps = args.n_reps
	alg_param_grid = get_grid(args.param_grid_def)
	n_grid = alg_param_grid.shape[0]
	config_grid_vals = get_grid(args.config_grid_def)
	n_config_grid = len(config_grid_vals)
	planing_loss = np.zeros((n_reps, n_config_grid, n_grid))
	info_dict = {}
	# ----- Run simulation in parrnell process---------------------------------------------#
	loss_rep_id_lst = []
	for i_rep in range(n_reps):
		# returns objects ids:
		args_r = deepcopy(args)
		planing_loss_rep_id = run_rep.remote(i_rep, alg_param_grid, args_r.config_grid_def, args_r)
		loss_rep_id_lst.append(planing_loss_rep_id)
	# end for i_rep
	# -----  get the results --------------------------------------------#
	for i_rep in range(n_reps):
		loss_rep = ray.get(loss_rep_id_lst[i_rep])
		if i_rep % max(n_reps // 100, 1) == 0:
			time_str = time.strftime("%H hours, %M minutes and %S seconds",
			                         time.gmtime(timeit.default_timer() - start_time))
			write_to_log('Finished: {} out of {} reps, time: {}'.format(i_rep + 1, n_reps, time_str), args)
		# end if
		planing_loss[i_rep] = loss_rep
		info_dict = {'planing_loss_avg': planing_loss.mean(axis=0), 'planing_loss_std': planing_loss.std(axis=0),
		             'alg_param_grid': alg_param_grid, 'n_reps_finished': i_rep + 1}
		if save_result:
			save_run_data(args, info_dict, verbose=0)
		# end if
	# end for i_rep
	if save_result:
		save_run_data(args, info_dict)
	stop_time = timeit.default_timer()
	write_to_log('Total runtime: ' +
	             time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)), args,
	             save_result)
	return info_dict


# end  run_simulations
# -------------------------------------------------------------------------------------------


def run_main_control(args, save_result=True, load_run_data_flag=False, result_dir_to_load='', save_PDF=False, plot=True,
                     local_mode=False, init_ray=True):
	SetMdpArgs(args)
	if load_run_data_flag:
		args, info_dict = load_run_data(result_dir_to_load)
	else:
		info_dict = run_simulations(args, save_result, local_mode, init_ray=init_ray)
	planing_loss_avg = info_dict['planing_loss_avg']
	planing_loss_std = info_dict['planing_loss_std']
	alg_param_grid = info_dict['alg_param_grid']
	if 'n_reps_finished' in info_dict.keys():
		n_reps_finished = info_dict['n_reps_finished']
	else:
		n_reps_finished = args.n_reps
	# end if
	# ----- Plot figures  ---------------------------------------------#

	if plot or save_PDF:
		ax = plt.figure().gca()
		if args.train_sampling_def['type'] in {'Generative', 'Generative_uniform', 'Generative_Stationary'}:
			data_size_per_traj = args.depth * args.n_episodes
		elif args.train_sampling_def['type'] == 'Trajectories':
			data_size_per_traj = args.depth * args.n_episodes
		elif args.train_sampling_def['type'] == 'sample_all_s_a':
			data_size_per_traj = args.nS * args.nA * args.n_episodes
		else:
			raise AssertionError
		# end if
		xscale = 1.
		legend_title = ''
		if args.param_grid_def['type'] == 'L2_factor':
			plt.xlabel(r'$L_2$ Regularization Factor [1e-2]')
			xscale = 1e2
		elif args.param_grid_def['type'] == 'L1_factor':
			plt.xlabel(r'$L_1$ Regularization Factor ')
		elif args.param_grid_def['type'] == 'gamma_guidance':
			plt.xlabel(r'Guidance Discount Factor $\gamma$')
		else:
			raise AssertionError('Unrecognized args.grid_type')
		# end if
		ci_factor = 1.96 / np.sqrt(n_reps_finished)  # 95% confidence interval factor

		config_grid_vals = get_grid(args.config_grid_def)
		for i_config, config_val in enumerate(config_grid_vals): # for of plots in the figure

			if args.config_grid_def['type'] == 'n_trajectories':
				if args.train_sampling_def['type'] in {'Generative_uniform', 'Generative', 'Generative_Stationary'}:
					legend_title = 'Num. Samples'
					label_str = '{} '.format(config_val * data_size_per_traj)
				else:
					legend_title = 'Num. Trajectories'
					label_str = '{} '.format(config_val)

			elif args.config_grid_def['type'] == 'states_actions_TV_dist_from_uniform':
				legend_title = 'Total-Variation from\n uniform [normalized]'
				label_str = '{} '.format(config_val)

			elif args.config_grid_def['type'] == 'chain_mix_time':
				legend_title = 'Mixing time'
				label_str = '{} '.format(config_val)

			elif args.config_grid_def['type'] == 'n_episodes':
				legend_title = 'Num. Episodes'
				label_str = '{} '.format(config_val)

			else:
				raise AssertionError
			# end if
			plt.errorbar(alg_param_grid * xscale, planing_loss_avg[i_config], yerr=planing_loss_std[i_config] * ci_factor,
			             marker='.', label=label_str)
			if show_stars:
				# Mark the lowest point:
				i_best = np.argmin(planing_loss_avg[i_config])
				plt.scatter(alg_param_grid[i_best] * xscale, planing_loss_avg[i_config][i_best], marker='*', s=400)
			# end if
		# for i_config
		plt.grid(True)
		plt.ylabel('Loss')
		plt.legend(title=legend_title, loc='best', fontsize=12)  # loc='upper right'
		if y_lim:
			plt.ylim(y_lim)
		# plt.xlim([0.5,1])
		# ax.set_yticks(np.arange(0., 9., step=1.))
		# plt.figure(figsize=(5.8, 3.0))  # set up figure size

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
	print('done')
	info_dict['result_dir'] = args.result_dir
	return info_dict


# end run_main


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
	info_dict = run_main_control(args, save_result=True, load_run_data_flag=load_run,
	                             result_dir_to_load=result_dir_to_load, save_PDF=save_PDF, local_mode=local_mode)
