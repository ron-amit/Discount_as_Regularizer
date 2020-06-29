"""
In  tabular MDP setting, evaluates the learning of optimal policy using different guidance discount factors
On-policy means we run episodes,
 in each episode we generate roll-outs/trajectories of current policy and run algorithm to improve the policy.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from copy import deepcopy
import timeit
import time

from main_control import run_main_control
from utils.common_utils import set_random_seed, create_result_dir, save_run_data, load_run_data, write_to_log, get_grid, start_ray, set_default_plot_params, save_fig

set_default_plot_params()


# -------------------------------------------------------------------------------------------
#  Run mode
# -------------------------------------------------------------------------------------------

load_run_data_flag = False  # False/True If true just load results from dir, o.w run simulation
result_dir_to_load = './saved/run_2d_reg_grid_PolOpt/2020_06_28_11_59_58'  #
save_PDF = False  # False/True - save figures as PDF file
local_mode = False  # True/False - run non-parallel to get error messages and debugging

# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

args = argparse.Namespace()

# ----- Run Parameters ---------------------------------------------#
args.run_name = ''   # 'Name of dir to save results in (if empty, name by time)'
args.seed = 1  # random seed
args.n_reps = 1000  # default 1000  # number of experiment repetitions

#  how to create parameter grid:
args.l2_grid_def =  {'type': 'L2_factor', 'spacing': 'linspace', 'start': 0, 'stop': 0.005, 'num': 11, 'decimals': 10}
args.gam_grid_def = {'type': 'gamma_guidance', 'spacing': 'linspace', 'start': 0.79, 'stop': 0.99, 'num': 11, 'decimals': 10}

# ----- Problem Parameters ---------------------------------------------#

# ----- Problem Parameters ---------------------------------------------#
# MDP definition ( see data_utils.SetMdpArgs)
# args.mdp_def = {'type': 'RandomMDP', 'S': 10, 'A': 5, 'k': 2, 'reward_std': 0.1}
args.mdp_def = {'type': 'GridWorld', 'N0': 4, 'N1': 4, 'reward_std': 0.1, 'forward_prob_distrb': 'uniform',  'goal_reward': 1, 'R_low': -0.5, 'R_high': 0.5}
# args.mdp_def = {'type': 'GridWorld', 'N0': 4, 'N1': 4, 'reward_std': 0.1, 'forward_prob_distrb': {'alpha': 3, 'beta': 1}, 'goal_reward': 1}


args.depth = 10  # default: 10  # Length of trajectory
args.gammaEval = 0.99   # default: 0.99  # gammaEval
args.n_episodes = 5  # Number of episodes
args.n_trajectories = 16   # default number of trajectories to generate per episode
args.train_sampling_def = {'type': 'Trajectories'}
args.config_grid_def = {'type': 'None', 'spacing': 'list', 'list': [None]}

# ----- Algorithm Parameters ---------------------------------------------#
args.method = 'SARSA'  # default: 'Expected_SARSA' # 'RL Algorithm'  # Options: 'Model_Based' | 'SARSA' | Expected_SARSA
args.TD_Init_type = 'zero'   # How to initialize V # Options: 'Vmax' (default) | 'zero' | 'random_0_1' |  'random_0_'Vmax'' | '0.5_'Vmax' |
args.use_reward_scaling = True
args.n_TD_iter = 5000  # Default: 500 for RandomMDP, 5000 for GridWorld  # number of TD iterations
args.epsilon = 0.1  # for epsilon-greedy
args.learning_rate_def = {'type': 'a/(b+i_iter)', 'a': 500, 'b': 1000, 'scale': True}
args.default_gamma = None  # default: None  # The default guidance discount factor (if None use gammaEval)
args.default_l2_factor = 1e-4    # default: None  # The default L2 factor (if using discount regularization) - note: it is necessary for LSTD

# -------------------------------------------------------------------------------------------
def run_simulations(args, local_mode):
	start_ray(local_mode)
	create_result_dir(args)
	write_to_log('local_mode == {}'.format(local_mode), args)
	start_time = timeit.default_timer()
	create_result_dir(args)
	set_random_seed(args.seed)

	l2_grid = get_grid(args.l2_grid_def)
	gam_grid = get_grid(args.gam_grid_def)
	write_to_log('gamma_grid == {}'.format(gam_grid), args)
	write_to_log('l2_grid == {}'.format(l2_grid), args)
	grid_shape = (len(l2_grid), len(gam_grid))
	loss_avg = np.zeros(grid_shape)
	loss_std = np.zeros(grid_shape)

	run_idx = 0
	for i0 in range(grid_shape[0]):
		for i1 in range(grid_shape[1]):
			args_run = deepcopy(args)
			args_run.param_grid_def = {'type': 'L2_factor', 'spacing': 'list', 'list': [l2_grid[i0]]}
			args_run.default_gamma = gam_grid[i1]

			info_dict = run_main_control(args_run, save_result=False, plot=False, init_ray=False)
			loss_avg[i0, i1] = info_dict['planing_loss_avg'][0]
			loss_std[i0, i1] = info_dict['planing_loss_std'][0]
			run_idx += 1
			print("Finished {}/{}".format(run_idx, loss_avg.size))
		# end for
	# end for
	grid_results_dict = {'l2_grid': l2_grid, 'gam_grid': gam_grid, 'loss_avg': loss_avg,
						 'loss_std': loss_std}
	save_run_data(args, grid_results_dict)
	stop_time = timeit.default_timer()
	write_to_log('Total runtime: ' +
				 time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)), args)
	return grid_results_dict
# -------------------------------------------------------------------------------------------


if __name__ == "__main__":
	if load_run_data_flag:
		args, grid_results_dict = load_run_data(result_dir_to_load)
	else:
		grid_results_dict = run_simulations(args, local_mode)
	l2_grid = grid_results_dict['l2_grid']
	gam_grid = grid_results_dict['gam_grid']
	loss_avg = grid_results_dict['loss_avg']
	loss_std = grid_results_dict['loss_std']

	ci_factor = 1.96 / np.sqrt(args.n_reps)  # 95% confidence interval factor
	max_deviate = 100. * np.max(loss_std * ci_factor /  loss_avg)
	print('Max 95% CI relative to mean: ', max_deviate, '%')

	# fig, ax = plt.subplots(figsize=(7, 7))
	with sns.axes_style("white"):

		yticklabels = np.around(l2_grid * 1e5, decimals=3)
		yticklabels = np.round(yticklabels).astype(int)
		xticklabels = np.around(gam_grid, decimals=3)
		ax = sns.heatmap(loss_avg,  cmap="YlGnBu", xticklabels=xticklabels, yticklabels = yticklabels,  annot=True, annot_kws={"size": 8})
		ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
		plt.xlabel(r'Guidance Discount Factor $\gamma$')
		plt.ylabel(r'$L_2$ Regularization Factor [1e-5')
		if save_PDF:
			save_fig(args.run_name)
		else:
			plt.title('Loss avg. Max 95% CI relative to mean: {}%\n {}'.format(np.around(max_deviate, decimals=1), args.run_name))
		plt.show()
		print('done')
