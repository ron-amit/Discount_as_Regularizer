import sys, os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, project_dir)


import glob, pickle
import numpy as np

from utils.common_utils import load_run_data, write_to_log, get_grid


#------------------------------------------------------------------------------------------------------------~

def get_num_finished(result_arr):
	if result_arr.ndim == 2:
		result_arr = result_arr[0]
	for i, x in enumerate(result_arr):
		if np.isnan(x):
			return i
	return result_arr.size


#------------------------------------------------------------------------------------------------------------~


def print_status(result_dir_to_load):

	args, info_dict = load_run_data(result_dir_to_load)
	alg_param_grid = get_grid(args.param_grid_def)
	n_grid = len(alg_param_grid)
	n_reps = args.n_reps
	print('Loaded parameters: \n', args, '\n', '-' * 20)
	if 'result_reward_mat' in info_dict.keys():
		result_reward_mat = info_dict['result_reward_mat']
	else:
		result_reward_mat = np.full(shape=(n_grid, n_reps), fill_value=np.nan)
	path = os.path.join(result_dir_to_load, 'jobs')
	all_files = glob.glob(os.path.join(path, "*.p"))
	n_rep_finish_per_point = np.full(shape=n_grid, fill_value=0, dtype=np.int)
	n_steps_finished = np.full(shape=(n_grid, n_reps), fill_value=-1, dtype=np.int)
	for f_path in all_files:
		save_dict = pickle.load(open(f_path, "rb"))
		job_info = save_dict['job_info']
		timesteps_snapshots = save_dict['timesteps_snapshots']
		i_grid = np.searchsorted(alg_param_grid, job_info['grid_param'], 'right')
		if i_grid == len(alg_param_grid):
			# the loaded param is not in our defined alg_param_grid
			continue
			# TODO: add option to load all files, even if not in the defined alg_param_grid (show_all_saved_results flag)
		i_rep = job_info['i_rep']
		if not np.isnan(result_reward_mat[i_grid, i_rep]):
			n_steps_finished[i_grid, i_rep] = args.max_timesteps
			n_rep_finish_per_point[i_grid] += 1
		else:
			n_steps_finished[i_grid, i_rep] = timesteps_snapshots[-1]

	for i_grid, grid_param in enumerate(alg_param_grid):
		write_to_log('Grid point {}/{}, val: {}, Number of finished reps loaded: {}'.
					 format(1 + i_grid, len(alg_param_grid), grid_param, n_rep_finish_per_point[i_grid]), args)
		for i_rep in range(n_reps):
			if n_steps_finished[i_grid, i_rep] != -1:
				print('Rep: {}, Finished Time-Steps: {}'.format(i_rep, n_steps_finished[i_grid, i_rep]))