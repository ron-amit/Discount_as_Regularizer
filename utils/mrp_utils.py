import numpy as np
from utils.prob_utils import sample_discrete
from utils.mdp_utils import SetMdpArgs, MDP
from utils.planing_utils import GetUniformPolicy, GetPolicyDynamics
from utils.prob_utils import augment_mix_time, generate_discrete_prob_TV, generate_prob_given_entropy

# Markov Reward Process

# ------------------------------------------------------------------------------------------------------------~


def SetMrpArgs(args):
    mrp_type = args.mrp_def['type']
    if mrp_type == 'ToyMRP':
        args.nS = 2
        args.reward_std = args.mrp_def['reward_std']
    elif mrp_type == 'Chain':
        args.nS = args.mrp_def['length']
        args.reward_std = args.mrp_def['reward_std']
    elif mrp_type in {'RandomMDP', 'GridWorld'}:
        SetMdpArgs(args)
    else:
        raise AssertionError('Invalid mrp_type')
# ------------------------------------------------------------------------------------------------------------~

# ------------------------------------------------------------------------------------------------------------~


class MRP(): # Markov Reward Process
    def __init__(self, args):
        """
          Parameters:

          Returns:

          """
        nS = args.nS  # number of states
        P = np.zeros((nS, nS))
        mrp_type = args.mrp_def['type']

        if mrp_type == 'ToyMRP':
            p01 = args.mrp_def['p01']
            p00 = 1 - p01
            p10 = args.mrp_def['p10']
            p11 = 1 - p10

            P[0, 0] = p00
            P[0, 1] = p01
            P[1, 0] = p10
            P[1, 1] = p11

            R = np.random.rand(nS)

        elif mrp_type == 'Chain':
            p_left = args.mrp_def['p_left']
            p_right = 1 - p_left
            nS = args.mrp_def['length']
            P = np.zeros((nS, nS))
            for i in range(nS):
                if i < nS - 1:
                    P[i, i+1] = p_right
                else:
                    P[i,i] = p_right
                if i > 0:
                    P[i, i - 1] = p_left
                else:
                    P[i, i] = p_left

            mean_reward_range =  args.mrp_def['mean_reward_range']
            R = mean_reward_range[0] + (mean_reward_range[1] - mean_reward_range[0]) * np.random.rand(nS)
            # R = np.random.rand(nS)

        elif mrp_type in {'RandomMDP', 'GridWorld'}:
            mdp = MDP(args, init_sampling=False)
            nS = mdp.nS
            nA = mdp.nA
            if args.mrp_def['policy'] == 'uniform':
                pi = GetUniformPolicy(nS, nA)
            else:
                raise AssertionError('Unrecognized policy def')
            P, R = GetPolicyDynamics(mdp.P, mdp.R, pi)

            # Modify P so we have the desired mixing time
            if 'forced_mix_time' in args:
                P = augment_mix_time(P, args.forced_mix_time)


        else:
            raise AssertionError('Invalid mrp_type')

        self.R = R
        self.P = P
        self.nS = nS
        self.type = args.mrp_def['type']
        self.define_sampling_method(args)
        self.reward_std = args.reward_std
        if args.initial_state_distrb_type == 'middle':
            self.initial_state_distrb = np.zeros(nS)
            self.initial_state_distrb[nS // 2] = 1.
        elif args.initial_state_distrb_type == 'uniform':
            self.initial_state_distrb = np.ones(nS) / nS
        else:
            raise AssertionError("Unrecognized initial_state_distrb_type")

    # ------------------------------------------------------------------------------------------------------------~


    def define_sampling_method(self, args):
        sampling_def = args.train_sampling_def
        nS = self.nS
        sampling_type = sampling_def['type']
        if sampling_type == 'Trajectories':
            pass  # in this case we will later sample a sequence trajectories according to some given policy
        elif sampling_type == 'sample_all_s':
            # in this case we will later sample all (s) X number of trajectories
            pass
        elif sampling_type == 'Generative':
            # Create a fixed sampling distribution
            # Create sampling distribution for states
            if 'states_TV_dist_from_uniform' in sampling_def.keys():
                tv_dist = sampling_def['states_TV_dist_from_uniform']
                probs_s = generate_discrete_prob_TV(args, nS, tv_dist_nrml=tv_dist)
            elif 'states_entropy' in sampling_def.keys():
                ent_S = sampling_def['states_entropy']
                probs_s = generate_prob_given_entropy(args, nS, ent_S)
            else:
                raise AssertionError('Unrecognized sampling_def')
            # end if
            self.probs_s = probs_s

        elif sampling_type == 'Generative_uniform':
            # Create fixed uniform sampling distribution
            probs_s = generate_discrete_prob_TV(args, nS, tv_dist_nrml=0)
            self.probs_s = probs_s

        else:
            raise AssertionError('Unrecognized sampling_def')
        # end if
        self.sampling_type = sampling_type
    # end def
    # ------------------------------------------------------------------------------------------------------------~


    def SampleDataMrp(self, args, p0=None):
        """
        # generate n trajectories

        Parameters:
        """
        n_traj = args.n_trajectories
        R = self.R
        P = self.P
        nS = self.nS
        reward_std = self.reward_std
        sampling_type = self.sampling_type
        depth = args.depth  # trajectory length
        if p0 is None:
            p0 = self.initial_state_distrb
        data = []
        if sampling_type == 'Trajectories':
            for i_traj in range(n_traj):
                data.append([])
                # sample initial state:
                s = sample_discrete(p0)
                for t in range(depth):
                    # Until t==depth, sample a~pi(.|s), s'~P(.|s,a), r~R(s,a)
                    s_next = sample_discrete(P[s, :])
                    r = R[s] + np.random.randn(1)[0] * reward_std
                    data[i_traj].append((s, r, s_next))
                    s = s_next
                # end for t
            # end for i_traj

        elif sampling_type in {'Generative', 'Generative_uniform'}:
            #  iid sampling according to distribution that was set in the init of current MDP
            probs_s = self.probs_s
            # Sample data
            for i_traj in range(n_traj):
                data.append([])
                for t in range(depth):
                    s = sample_discrete(probs_s)
                    r = R[s] + np.random.randn(1)[0] * reward_std
                    s_next = sample_discrete(P[s, :])
                    data[i_traj].append((s, r, s_next))
            # end for t
        # end for i_traj
        elif sampling_type == 'sample_all_s':
            for i_traj in range(n_traj):
                data.append([])
                for s in range(nS):
                    r = R[s] + np.random.randn(1)[0] * reward_std
                    s_next = sample_discrete(P[s, :])
                    data[i_traj].append((s, r, s_next))
                # end for s
            # end for i_traj
        else:
            raise AssertionError('Unrecognized data_type')
        return data


# ------------------------------------------------------------------------------------------------------------~


def calc_typical_mixing_time(P):
    evals, evecs = np.linalg.eig(P)
    evals_abs = np.abs(evals)
    evals_abs.sort()
    mixing_time = 1 / (1-evals_abs[-2]) # inverse spectral gap , note: evals_abs[-1] always == 1
    return mixing_time
# ------------------------------------------------------------------------------------------------------------~

