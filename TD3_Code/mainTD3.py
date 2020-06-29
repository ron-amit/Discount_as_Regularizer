import numpy as np
import torch
import gym
import argparse
import os
import pickle

from TD3_Code import utils, TD3, OurDDPG, DDPG
import json
# import warnings

# ---------------------------------------------------------------------------------------------------------------------------------#

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


# ---------------------------------------------------------------------------------------------------------------------------------#

def run_simulation_TD3(args, job_info=None):
    # warnings.filterwarnings("ignore", message="Box bound precision lowered by casting to float32")
    # warnings.filterwarnings("ignore", message="WARN: gym.spaces.Box autodetected dtype")

    if "alg" not in args:
        args.alg = args.policy

    file_name = f"{args.alg}_{args.env}_{args.seed}_{args.run_name}"
    print("---------------------------------------")
    print(f"Policy: {args.alg}, Env: {args.env}, Seed: {args.seed}, SimName: {args.run_name}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "actor_hiddens": args.actor_hiddens,
        "critic_hiddens": args.critic_hiddens,
        "actor_l2_reg": args.actor_l2_reg,
        "critic_l2_reg": args.critic_l2_reg,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.alg == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.alg == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.alg == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = []
    timesteps_snapshots = []

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    eval_reward = None

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            for i_iter in range(args.iter_per_sample):
                policy.train(replay_buffer, args.batch_size)
            # end for i_iter
        # end if

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        # end if

        # Evaluate
        if (t + 1) % args.eval_freq == 0:
            eval_reward = eval_policy(policy, args.env, args.seed, args.evaluation_num_episodes)
            evaluations.append(eval_reward)
            np.save(f"./results/{file_name}", evaluations)
            timesteps_snapshots.append(t+1)
            if args.save_model:
                policy.save(f"./models/{file_name}")
            # end if
            f_path = os.path.join(args.result_dir, 'jobs', args.job_name) + ".p"
            save_dict = {'job_info': job_info,
                         'timesteps_snapshots': timesteps_snapshots,
                         'evaluations': evaluations}
            pickle.dump(save_dict, open(f_path, "wb"))
        # end if
    #  end for t
    print("final_eval_reward: ", eval_reward)
    return eval_reward
# end run_simulation

# ---------------------------------------------------------------------------------------------------------------------------------#


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--run_name", default="")  #
    parser.add_argument("--iter_per_sample", default=1)  #
    parser.add_argument("--actor_hiddens", default='[400, 300]', type=json.loads)
    parser.add_argument("--critic_hiddens", default='[400, 300]', type=json.loads)
    parser.add_argument("--critic_l2_reg", default=0)  #  L2 regularization factor for the Q-networks (critic)
    parser.add_argument("--actor_l2_reg", default=0)  # L2 regularization factor for the policy-networks (actor)

    args = parser.parse_args()
    run_simulation_TD3(args)
