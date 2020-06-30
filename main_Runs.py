# Uses code from:
# https://github.com/sfujim/TD3
# https://github.com/pranz24/pytorch-soft-actor-critic

import sys
import os
import argparse
import timeit
import time
import glob
import ray
import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import ray
from utils.common_utils import create_result_dir, save_run_data, load_run_data, write_to_log, time_now, get_grid, \
	create_results_backup, start_ray, save_fig, pretty_print_args
from utils.jobs_utils import get_num_finished, print_status

params = {'font.size': 10, 'lines.linewidth': 2, 'legend.fontsize': 10, 'legend.handlelength': 2, 'pdf.fonttype': 42,
		  'ps.fonttype': 42}
plt.rcParams.update(params)

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, project_dir)

# --------------------------------------------------------------------------------------------------------------------#
#  Set parameters
# --------------------------------------------------------------------------------------------------------------------#

parser = argparse.ArgumentParser()
parser.add_argument("--alg", default="TD3")  # algorithm name 'TD3' 'SAC' |  'OurDDPG')
parser.add_argument("--env", default="Hopper-v2")  # OpenAI gym environment name
parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=1e4, type=int)  # Time steps initial random policy is used
parser.add_argument("--max_timesteps", default=2e5, type=int)  # Max time steps to run environment
parser.add_argument("--eval_freq", default=int(25e3), type=int)  # How often (time steps) we evaluate  and save 'snapshot'
parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)  # 256  # Batch size for both actor and critic
parser.add_argument("--default_discount", default=0.999)  # Default Discount factor
parser.add_argument("--tau", default=0.005)  # Target network update rate
parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
parser.add_argument("--run_name",
					default="")  # ='Name of dir to save results  (of the grid search) in (if empty, name by time)',
parser.add_argument("--iter_per_sample", default=1)  #
args = parser.parse_args()

# hardware resources required for each worker:
num_gpus = 0.3  # how much of the GPU is required for each worker

# run config
smoke_test = False  # True/False - short run for debug
local_mode = False  # True/False - run non-parallel to get error messages and debugging

save_PDF = False  # False/True - save figures as PDF file
y_lim = None  #  None | [100, 900]
x_lim = None   #  None | [0.4, 1]


# Option to load previous run results or continue unfinished run or start a new run:
run_mode = 'New'  # 'New' / 'Load' / 'Continue' / 'ContinueNewGrid' / 'ContinueAddGrid' / 'LoadSnapshot'
increase_n_reps_in_loaded_grid = False  # True/False - if true and run_mode is a continue mode, then increase the number
# of reps im each point to be at least  args.n_reps  o.w, loaded grid point will stay with
# the number of reps they have


# If run_mode ==  'Load' / 'Continue' use this results dir:
result_dir_to_load = './saved/main_Runs/2020_06_14_17_32_16'

timesteps_snapshot_to_load = int(25e3)  # Used if run_mode=='LoadSnapshot'

args.n_reps = 20  # 100 # number of experiment repetitions for each point in grid
args.evaluation_num_episodes = 1000
args.save_model = False
#  how to create parameter grid:

args.param_grid_def = {'type': 'gamma_guidance', 'spacing': 'list', 'decimals': 4, 'list': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 1.]}
# args.param_grid_def = {'type': 'gamma_guidance', 'spacing': 'linspace', 'start': 0.2, 'stop': 0.999, 'num': 10, 'decimals': 4}
# args.param_grid_def = {'type': 'L2_factor', 'spacing': 'linspace', 'start': 0.0, 'stop': 0.05, 'num': 11, 'decimals': 4}
# args.param_grid_def = {'type': 'L2_factor', 'spacing': 'list', 'list': [0.055, 0.06, 0.065, 0.07, 0.075, 0.08]}

args.default_critic_l2_reg = 0  # default L2 regularization factor for the Q-networks (critic)
args.actor_l2_reg = 0  # L2 regularization factor for the policy-networks (actor)

args.actor_hiddens = [400, 300]
args.critic_hiddens = [400, 300]

# ----------------------------------------------------------------------
if smoke_test:
	print('Smoke Test !!!!!!! \n' * 10)
	args.start_timesteps = 1e0
	args.max_timesteps = 1e1
	args.eval_freq = 2e0
	args.n_reps = 2
	args.evaluation_num_episodes = 2


# ----------------------------------------------------------------------
# define a remote function
@ray.remote(num_gpus=num_gpus, max_calls=1)
def run_simulation_remote(args_run, job_name, job_info):
	if args_run.alg in {'TD3', 'OurDDPG'}:
		from TD3_Code.mainTD3 import run_simulation_TD3
		write_to_log('Starting: {}, time: {}'.format(job_name, time_now()), args_run)
		args_run.run_name += ':' + job_name
		return run_simulation_TD3(args_run, job_info), args_run
	# elif  args_run.alg == 'SAC':
	# 	from SAC_Code.main_SAC import run_simulation_SAC
	# 	write_to_log('Starting: {}, time: {}'.format(job_name, time_now()), args_run)
	# 	args_run.run_name += ':' + job_name
	# 	return run_simulation_SAC(args_run, job_info), args_run
	else:
		raise AssertionError


# --------------------------------------------------------------------------------------------------------------------#

#  Get results
# --------------------------------------------------------------------------------------------------------------------#

# --------------------------------------------------------------------------------------------------------------------#

if run_mode in {'Load', 'Continue'}:
	print_status(result_dir_to_load)
	#  Load previous run
	args, info_dict = load_run_data(result_dir_to_load)
	args.result_dir = result_dir_to_load  # update the path, in case the result folder moved
	if 'result_reward_mat' not in info_dict.keys():
		raise AssertionError('run_mode=={}, but the path result_dir_to_load does not contain any finished results (try run_mode="LoadSnapshot")'.format(run_mode))
	alg_param_grid = info_dict['alg_param_grid']
	result_reward_mat = info_dict['result_reward_mat']
	run_time = info_dict['run_time']
	n_grid = len(alg_param_grid)
	n_reps_per_point = np.full(shape=n_grid, fill_value=args.n_reps)
	print('Loaded parameters: \n', args, '\n', '-' * 20)

# --------------------------------------------------------------------------------------------------------------------#

elif run_mode == 'LoadSnapshot':
	print_status(result_dir_to_load)
	#  Load previous run
	args, info_dict = load_run_data(result_dir_to_load)
	print('Loaded parameters: \n', args, '\n', '-' * 20)
	args.result_dir = result_dir_to_load  # update the path, in case the result folder moved
	alg_param_grid = get_grid(args.param_grid_def)
	run_time = info_dict['run_time']
	path = os.path.join(result_dir_to_load, 'jobs')
	all_files = glob.glob(os.path.join(path, "*.p"))
	n_grid = len(alg_param_grid)
	n_reps = args.n_reps
	n_reps_per_point = np.full(shape=n_grid, fill_value=0, dtype=np.int)
	result_reward_mat = np.full(shape=(n_grid, n_reps), fill_value=np.nan)

	for f_path in all_files:
		save_dict = pickle.load(open(f_path, "rb"))
		job_info = save_dict['job_info']
		timesteps_snapshots = save_dict['timesteps_snapshots']
		evaluations = save_dict['evaluations']
		load_idx = np.searchsorted(timesteps_snapshots, int(timesteps_snapshot_to_load), 'right')
		if load_idx == len(timesteps_snapshots):
			continue  # snapshot not found
		i_grid = np.searchsorted(alg_param_grid, job_info['grid_param'], 'right')
		if i_grid == len(alg_param_grid):
			# the loaded param is not in our defined alg_param_grid
			continue
			# TODO: add option to load all files, even if not in the defined alg_param_grid (show_all_saved_results flag)
		i_rep = job_info['i_rep']
		result_reward_mat[i_grid, n_reps_per_point[i_grid]] = evaluations[load_idx]
		n_reps_per_point[i_grid] += 1


# --------------------------------------------------------------------------------------------------------------------#

elif run_mode in {'ContinueNewGrid', 'ContinueAddGrid'}:
	print_status(result_dir_to_load)
	# Create a new gird according to param_grid_def defined above, and use the loaded results if compatible.
	# all the other run  args (besides param_grid_def) are according to the loaded file
	loaded_args, info_dict = load_run_data(result_dir_to_load)
	if 'alg' not in loaded_args: # legacy naming
		loaded_args.alg = loaded_args.policy
	assert loaded_args.param_grid_def['type'] == args.param_grid_def['type']
	create_results_backup(result_dir_to_load)
	loaded_alg_param_grid = info_dict['alg_param_grid']
	loaded_result_mat = info_dict['result_reward_mat']
	run_time = info_dict['run_time']
	args.result_dir = result_dir_to_load  # update the path, in case the result folder moved

	new_param_grid_def = args.param_grid_def
	desired_alg_param_grid = get_grid(new_param_grid_def)
	args = deepcopy(loaded_args)

	if run_mode == 'ContinueAddGrid':
		new_alg_param_grid = np.union1d(loaded_alg_param_grid, desired_alg_param_grid)
		new_alg_param_grid.sort()
		args.param_grid_def['spacing'] = 'list',
		args.param_grid_def['list'] = new_alg_param_grid

	else:  # run_mode == 'ContinueNewGrid'
		# in this case we omit loaded grid point that are not in desired_alg_param_grid
		args.param_grid_def = new_param_grid_def
		new_alg_param_grid = desired_alg_param_grid

	n_grid = len(new_alg_param_grid)
	desired_n_reps = args.n_reps

	# check how many completed results we have in loaded data:
	found_points = False
	finished_reps_per_point = np.zeros(n_grid, dtype=np.int)
	for i_grid, grid_param in enumerate(new_alg_param_grid):
		if grid_param in loaded_alg_param_grid:
			load_idx = np.nonzero(loaded_alg_param_grid == grid_param)
			found_points = True
			finished_reps_per_point[i_grid] = get_num_finished(loaded_result_mat[load_idx])
	# end for i_grid
	if not found_points:
		raise Warning('Loaded file  {} did not complete any of the desired grid points'.format(result_dir_to_load))

	# determine number of reps required in each grid point
	n_reps_per_point = np.zeros(n_grid, dtype=np.int)
	for i_grid, grid_param in enumerate(new_alg_param_grid):
		if grid_param in desired_alg_param_grid or increase_n_reps_in_loaded_grid:
			n_reps_per_point[i_grid] = max(desired_n_reps, finished_reps_per_point[i_grid])
		else:
			n_reps_per_point[i_grid] = finished_reps_per_point[i_grid]

	# now take completed results from loaded data:
	result_reward_mat = np.full((n_grid, np.max(n_reps_per_point)), np.nan)
	for i_grid, grid_param in enumerate(new_alg_param_grid):
		if grid_param in loaded_alg_param_grid:
			load_idx = np.nonzero(loaded_alg_param_grid == grid_param)
			for i_rep in range(finished_reps_per_point[i_grid]):
				result_reward_mat[i_grid, i_rep] = loaded_result_mat[load_idx, i_rep]
			# end for i_rep
		# end if
	# end for i_grid
	write_to_log('Continue run with new grid def {}, {}'.format(new_param_grid_def, time_now()), args)
	write_to_log('Run parameters: \n' + str(args) + '\n' + '-' * 20, args)
	pretty_print_args(args)
	alg_param_grid = new_alg_param_grid
# --------------------------------------------------------------------------------------------------------------------#

elif run_mode == 'New':
	# Start from scratch
	run_time = 0
	create_result_dir(args)
	os.makedirs(os.path.join(args.result_dir, 'jobs'))
	alg_param_grid = get_grid(args.param_grid_def)
	n_grid = len(alg_param_grid)
	n_reps = args.n_reps
	n_reps_per_point = np.full(shape=n_grid, fill_value=n_reps, dtype=np.int)
	result_reward_mat = np.full(shape=(n_grid, n_reps), fill_value=np.nan)
else:
	raise AssertionError('Unrecognized run_mode')
# --------------------------------------------------------------------------------------------------------------------#

if run_mode in {'New', 'Continue', 'ContinueNewGrid', 'ContinueAddGrid'}:

	# Run grid
	start_time = timeit.default_timer()
	start_ray(local_mode)
	write_to_log('Run grid == {}'.format(alg_param_grid), args)
	write_to_log('local_mode == {}'.format(local_mode), args)
	out_id = [[None for _ in range(n_reps_per_point[i_grid])] for i_grid in range(n_grid)]
	# check previously finished reps
	n_finished_g = np.zeros(n_grid, dtype=np.int)
	for i_grid, grid_param in enumerate(alg_param_grid):
		n_finished_g[i_grid] = get_num_finished(result_reward_mat[i_grid])
		write_to_log(
			f'Grid point {1 + i_grid}/{len(alg_param_grid)}, val: {grid_param},'
			f' Number of finished reps loaded: {n_finished_g[i_grid]}', args)
	# Run several repetitions of training:
	# note: the for i_rep is th outer loop so we will get initial results from all grid point, even if not accurate
	for i_rep in range(np.max(n_reps_per_point)):
		for i_grid, grid_param in enumerate(alg_param_grid):
			n_finished = n_finished_g[i_grid]  # number of reps already finished
			if n_finished <= i_rep < n_reps_per_point[i_grid]:
				args_run = deepcopy(args)
				# Set seed (unique for each repetition)
				args_run.seed = args.seed + i_rep
				# Set args with the grid value
				if args.param_grid_def['type'] == 'L2_factor':
					args_run.discount = args.default_discount
					args_run.critic_l2_reg = grid_param
					job_name = 'L2_' + str(grid_param)
				elif args.param_grid_def['type'] == 'gamma_guidance':
					args_run.discount = grid_param
					args_run.critic_l2_reg = args.default_critic_l2_reg
					job_name = 'Gamma_' + str(grid_param)
				else:
					raise AssertionError('Unrecognized args.grid_type')
				job_name += '_Rep_' + str(i_rep)
				args_run.job_name = job_name
				# Start the job and get the outputs objects id's:
				job_info = {'grid_param': grid_param, 'i_rep': i_rep}
				out_id[i_grid][i_rep] = run_simulation_remote.remote(args_run, job_name, job_info)
	# end for i_grid
	# end if
	# end for i_rep
	# collect the outputs of the finished jobs:
	for i_rep in range(np.max(n_reps_per_point)):
		for i_grid, grid_param in enumerate(alg_param_grid):
			# check if job is already finished (or not sent at all):
			if i_rep >= n_reps_per_point[i_grid] or not np.isnan(result_reward_mat[i_grid, i_rep]):
				continue  # skip
			output, args_run = ray.get(out_id[i_grid][i_rep])
			result_reward_mat[i_grid, i_rep] = output
			# note: the final reward is an average performance on eval_episodes=10 of final policy
			write_to_log(
				f'Finished Rep: {i_rep + 1}/{n_reps_per_point[i_grid]} of Grid point {i_grid}/{len(alg_param_grid)}'
				f' ({args_run.job_name}), Reward : {output}, Time now: {time_now()}', args)
			# Save results so far:
			stop_time = timeit.default_timer()
			run_time += stop_time - start_time
			start_time = timeit.default_timer()
			save_run_data(args, {'alg_param_grid': alg_param_grid, 'result_reward_mat': result_reward_mat,
								 'run_time': run_time}, verbose=1)
	# end for i_grid
	# end for i_rep

	write_to_log('Total runtime: ' + time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(run_time)), args)

# --------------------------------------------------------------------------------------------------------------------#
#  Plot results
# --------------------------------------------------------------------------------------------------------------------#

mean_reward = []
std_reward = []
n_finished_g = []
grid_vals = []
for i_grid, val in enumerate(alg_param_grid):
	n_finished = get_num_finished(result_reward_mat[i_grid])
	if n_finished > 0:
		mean_r = np.mean(result_reward_mat[i_grid, :n_finished])
		mean_reward.append(mean_r)
		std_r = np.std(result_reward_mat[i_grid, :n_finished])
		std_reward.append(std_r)
		n_finished_g.append(n_finished)
		grid_vals.append(val)
		print(
			f'Grid point {i_grid + 1}/{len(alg_param_grid)}, Val {val}, #Reps Finished {n_finished},'
			f' Mean Reward {mean_r}, STD Reward {std_r}')

mean_reward = np.array(mean_reward)
std_reward = np.array(std_reward)
grid_vals = np.array(grid_vals)
n_finished_g = np.array(n_finished_g)
xscale = 1.
if args.param_grid_def['type'] == 'L2_factor':
	xscale = 1e2
	xlabel = r'$L_2$ Factor [1e-2]'
	title_prefix = args.env + r', $L_2$ Regularization'
elif args.param_grid_def['type'] == 'gamma_guidance':
	xlabel = r'Guidance Discount Factor $\gamma$'
	title_prefix = args.env + r', Discount Regularization'
else:
	raise AssertionError('Unrecognized args.grid_type')

if run_mode == 'LoadSnapshot':
	title_prefix += ', TimeSteps: {}'.format(int(timesteps_snapshot_to_load))
else:
	title_prefix += ', TimeSteps: {}'.format(int(args.max_timesteps))

# Plot number of finished reps
plt.figure()
plt.plot(xscale * grid_vals, n_finished_g, marker='o')
plt.grid(True)
plt.xlabel(xlabel)
plt.xlabel("# finished reps")

# Plot reward
ci_factor = 1.96 / np.sqrt(n_finished_g)  # 95% confidence interval factor
plt.figure()
plt.plot(xscale * grid_vals, mean_reward, marker='o')

plt.fill_between(xscale * grid_vals, mean_reward - std_reward * ci_factor, mean_reward + std_reward * ci_factor,
				 color='blue', alpha=0.2)
plt.grid(True)
plt.xlabel(xlabel)
if y_lim:
	plt.ylim(y_lim)
if x_lim:
	plt.xlim(x_lim)
plt.ylabel('Average Episode Return')
if save_PDF:
	plt.title(title_prefix)
	save_fig(args.run_name)
else:
	plt.title(f'{title_prefix}, reps_finished: {min(n_finished_g)} - {max(n_finished_g)}\n {args.result_dir}',
			  fontsize=8)
# + 'Episode Reward Mean +- 95% CI, ' + ' \n '
plt.show()
