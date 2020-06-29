

import argparse
# from main_control import run_main_control
from main_MRP import run_main_mrp
import os
from utils.common_utils import create_result_dir, convert_args, time_now, write_to_file, set_default_plot_params
from copy import deepcopy

set_default_plot_params()

# -------------------------------------------------------------------------------------------
#  Run mode
# -------------------------------------------------------------------------------------------

local_mode = False  # True/False - run non-parallel to get error messages and debugging

plot = False  # False/True - show plots
save_PDF = True  # False/True - save figures as PDF file
# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

args_shared = argparse.Namespace()

# -----  Run Parameters ---------------------------------------------#

args_shared.run_name = ''  # 'Name of dir to save results in (if empty, name by time)'
args_shared.seed = 1  # random seed

args_shared.n_reps = 5000   # default 5000  # number of experiment repetitions


# ------------------------------------------------------------------------------------------
create_result_dir(args_shared, run_type='Multi_Exp')
log_file_path = os.path.join(args_shared.result_dir, 'log') + '.out'
# -------------------------------------------------------------------------------------------
i_exp = 1


# -------------------------------------------------------------------------------------------
args = deepcopy(args_shared)
args.run_name = 'TD_sample_all_s_discount_reg'
args.mrp_def = {'type': 'GridWorld', 'N0': 4, 'N1': 4, 'reward_std': 0.1, 'forward_prob_distrb': 'uniform',  'goal_reward': 1, 'R_low': -0.5, 'R_high': 0.5, 'policy': 'uniform'}
args.param_grid_def = {'type': 'gamma_guidance', 'spacing': 'linspace', 'start': 0.9, 'stop': 0.99, 'num': 50, 'decimals': 10}
args.config_grid_def = {'type': 'n_trajectories', 'spacing': 'list', 'list': [1, 2, 4, 8]}
args.depth = 50  # default: 10 for 'chain', 100 for 'GridWorld'  # Length of trajectory
args.gammaEval = 0.99   # default: 0.99  # gammaEval
args.train_sampling_def = {'type': 'sample_all_s'}
args.evaluation_loss_type = 'L2_uni_weight' #  'rankings_kendalltau' | 'L2_uni_weight | 'L2'
args.initial_state_distrb_type = 'uniform'  # 'uniform' | 'middle'
args.default_n_trajectories = 2  #
args.default_gamma = None  # default: None  # The default guidance discount factor (if None use gammaEval)
args.alg_type = 'batch_TD_value_evaluation'  # 'LSTD' | 'LSTD_Nested' | batch_TD_value_evaluation | LSTD_Nested_Standard
args.use_reward_scaling = False  # False | True.  set False for LSTD
args.base_lstd_l2_fp = 1e-5
args.base_lstd_l2_proj = 1e-4
# if batch_TD_value_evaluation is used:
args.default_l2_TD = None  # default: None  # The default L2 factor for TD (if using discount regularization)
args.TD_Init_type = 'zero'  # How to initialize V # Options: 'Vmax' | 'zero' | 'random_0_1' |  'random_0_Vmax' | '0.5_'Vmax' |
args.n_TD_iter = 5000  # Default: 500 for RandomMDP, 5000 for GridWorld  # number of TD iterations
args.learning_rate_def = {'type': 'a/(b+i_iter)', 'a': 500, 'b': 1000, 'scale': False}

#####
args.result_dir = os.path.join(args_shared.result_dir, 'Exp_' + str(i_exp) + '_' + args.run_name)
write_to_file('Starting Exp {} ({})'.format(i_exp, args.run_name), log_file_path)
info_dict = run_main_mrp(args, save_result=True, local_mode=local_mode, plot=plot, save_PDF=save_PDF)
write_to_file('Exp {}. ({}) saved in {},  time: {}'.
              format(i_exp, args.run_name, info_dict['result_dir'], time_now()), log_file_path)
i_exp += 1
####
# -------------------------------------------------------------------------------------------
args = deepcopy(args_shared)
args.run_name = 'TD_sample_all_s_L2_reg'
args.mrp_def = {'type': 'GridWorld', 'N0': 4, 'N1': 4, 'reward_std': 0.1, 'forward_prob_distrb': 'uniform',  'goal_reward': 1, 'R_low': -0.5, 'R_high': 0.5, 'policy': 'uniform'}
args.param_grid_def = {'type': 'l2_TD', 'spacing': 'linspace', 'start': 0., 'stop': 0.01, 'num': 50, 'decimals': 10}
args.config_grid_def = {'type': 'n_trajectories', 'spacing': 'list', 'list': [1, 2, 4, 8]}
args.depth = 50  # default: 10 for 'chain', 100 for 'GridWorld'  # Length of trajectory
args.gammaEval = 0.99   # default: 0.99  # gammaEval
args.train_sampling_def = {'type': 'sample_all_s'}
args.evaluation_loss_type = 'L2_uni_weight' #  'rankings_kendalltau' | 'L2_uni_weight | 'L2'
args.initial_state_distrb_type = 'uniform'  # 'uniform' | 'middle'
args.default_n_trajectories = 2  #
args.default_gamma = None  # default: None  # The default guidance discount factor (if None use gammaEval)
args.alg_type = 'batch_TD_value_evaluation'  # 'LSTD' | 'LSTD_Nested' | batch_TD_value_evaluation | LSTD_Nested_Standard
args.use_reward_scaling = False  # False | True.  set False for LSTD
args.base_lstd_l2_fp = 1e-5
args.base_lstd_l2_proj = 1e-4
# if batch_TD_value_evaluation is used:
args.default_l2_TD = None  # default: None  # The default L2 factor for TD (if using discount regularization)
args.TD_Init_type = 'zero'  # How to initialize V # Options: 'Vmax' | 'zero' | 'random_0_1' |  'random_0_Vmax' | '0.5_'Vmax' |
args.n_TD_iter = 5000  # Default: 500 for RandomMDP, 5000 for GridWorld  # number of TD iterations
args.learning_rate_def = {'type': 'a/(b+i_iter)', 'a': 500, 'b': 1000, 'scale': False}

#####
args.result_dir = os.path.join(args_shared.result_dir, 'Exp_' + str(i_exp) + '_' + args.run_name)
write_to_file('Starting Exp {} ({})'.format(i_exp, args.run_name), log_file_path)
info_dict = run_main_mrp(args, save_result=True, local_mode=local_mode, plot=plot, save_PDF=save_PDF)
write_to_file('Exp {}. ({}) saved in {},  time: {}'.
              format(i_exp, args.run_name, info_dict['result_dir'], time_now()), log_file_path)
i_exp += 1
####
# -------------------------------------------------------------------------------------------
