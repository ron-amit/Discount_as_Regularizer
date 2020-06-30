import numpy as np
from utils.prob_utils import sample_discrete, generate_prob_given_entropy, generate_discrete_prob_TV
from utils.planing_utils import get_stationary_distrb, GetUniformPolicy, GetPolicyDynamics
from utils.prob_utils import augment_mix_time

# Markov Decision Process
# ------------------------------------------------------------------------------------------------------------~


def SetMdpArgs(args):

	if 'mdp_def' in args:
		mdp_def = args.mdp_def
	else:
		mdp_def = args.mrp_def
	mdp_type = mdp_def['type']
	args.mdp_def = mdp_def
	if mdp_type == 'RandomMDP':
		args.nS = mdp_def['S']  # number of states
		args.nA = mdp_def['A']  # number of actions
		args.k = mdp_def['k']  # Number of non-zero entries in each row  of transition-matrix
		args.reward_std = mdp_def['reward_std']
	elif mdp_type == 'GridWorld':
		args.nS = mdp_def['N0'] * mdp_def['N1']
		args.nA = 5
		args.reward_std = mdp_def['reward_std']
	elif mdp_type == 'GridWorld2':
		args.nS = mdp_def['N0'] * mdp_def['N1']
		args.nA = 4
		args.reward_std = mdp_def['reward_std']
	else:
		raise AssertionError('Invalid mdp_type')


#------------------------------------------------------------------------------------------------------------~


class MDP(): # Markov Desicion Process
	def __init__(self, args, init_sampling=True):
		"""
		  Randomly generate an MDP


		  Parameters:

		  Returns:
		  P: [nS x nA x nS] transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
		  R: [nS x nA] mean rewards matrix R
		  """
		nS = args.nS  # number of states
		nA = args.nA  # number of actions
		self.nA = nA
		self.nS = nS

		####### Create the MDP
		mdp_type = args.mdp_def['type']
		if mdp_type == 'RandomMDP':
			P, R = self.generate_RandomMDP(args)
		elif mdp_type == 'GridWorld':
			P, R = self.generate_GridWorld(args)
		else:
			raise AssertionError('Invalid mdp_type')

		# Save MDP parameters
		assert np.max(R) <= 1   # we assume Rmax <= 1
		self.R = R
		self.reward_std = args.reward_std
		self.P = P
		self.type = args.mdp_def['type']
		self.define_initial_distrb(args)
		if init_sampling:
			self.define_sampling_method(args)
	# end function
	# ------------------------------------------------------------------------------------------------------------~

	def define_initial_distrb(self, args):
		nS = self.nS
		nA = self.nA
		if 'initial_state_distrb' in args.mdp_def.keys():
			initial_state_distrb = args.mdp_def['initial_state_distrb']
			if initial_state_distrb == 'uniform':
				self.initial_state_distrb = np.ones(nS) / nS  # uniform
			elif initial_state_distrb == 'state_0':
				self.initial_state_distrb = np.zeros(nS)
				self.initial_state_distrb[0] = 1.
			else:
				raise AssertionError('Invalid initial_state_distrb')
		else:
			self.initial_state_distrb = np.ones(nS) / nS  # uniform
	# ------------------------------------------------------------------------------------------------------------~

	def define_sampling_method(self, args):
		sampling_def = args.train_sampling_def
		nS = self.nS
		nA = self.nA

		if sampling_def['type'] == 'Trajectories':
			pass  # in this case we will later sample a sequence trajectories according to some given policy

		elif sampling_def['type'] == 'Generative_Stationary':
			pass  # in this case we will later sample a iid draws from the stationary distribution according to some given policy

		elif sampling_def['type'] == 'sample_all_s_a':
			# in this case we will later sample all (s,a) X number of trajectories
			pass

		elif sampling_def['type'] == 'chain_mix_time':
			# in this case the data generation is done by a chain we set to have some mix time (the policy is ignored)
			pi_unif = GetUniformPolicy(nS, nA)
			Pss, Rs = GetPolicyDynamics(self.P, self.R, pi_unif)
			# Modify P so we have the desired mixing time
			mix_time = args.train_sampling_def['mix_time']
			Pss = augment_mix_time(Pss, mix_time)
			self.Pss_fixed = Pss

		elif sampling_def['type'] == 'Generative':
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
			# Create sampling distribution for actions
			if 'actions_TV_dist_from_uniform' in sampling_def.keys():
				tv_dist = sampling_def['actions_TV_dist_from_uniform']
				probs_a = generate_discrete_prob_TV(args, nA, tv_dist_nrml=tv_dist)
			elif 'actions_entropy' in sampling_def.keys():
				ent_A = sampling_def['actions_entropy']
				probs_a = generate_prob_given_entropy(args, nA, ent_A)
			else:
				raise AssertionError('Unrecognized sampling_def')
			self.probs_s = probs_s
			self.probs_a = probs_a

		elif sampling_def['type'] == 'Generative_uniform':
			# Create fixed uniform sampling distribution
			probs_s = generate_discrete_prob_TV(args, nS, tv_dist_nrml=0)
			probs_a = generate_discrete_prob_TV(args, nA, tv_dist_nrml=0)
			self.probs_s = probs_s
			self.probs_a = probs_a

		else:
			raise AssertionError('Unrecognized sampling_def')
		# end if
		self.sampling_type =  sampling_def['type']
	# end def
	# ------------------------------------------------------------------------------------------------------------~


	def SampleData(self, args, pi, n_traj, p0=None):
		"""
		# generate n trajectories

		Parameters:
		P: [nS x nA x nS] transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
		R: [nS x nA] mean rewards matrix R
		pi: [nS x nA]  matrix representing  pi(a|s)
		n: number of trajectories to generate
		depth: Length of trajectory
		p0 (optional) [nS] matrix of initial state distribution
		Returns:
		data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
		"""
		R = self.R
		P = self.P
		nS = self.nS
		nA = self.nA
		reward_std = self.reward_std
		depth = args.depth
		if p0 is None:
			p0 = self.initial_state_distrb
		data = []
		sampling_type = self.sampling_type

		if sampling_type == 'Trajectories':
			# generate sampled trajectories
			for i_traj in range(n_traj):
				data.append([])
				# sample initial state:
				s = sample_discrete(p0)
				a = sample_discrete(pi[s, :])
				# Until t==depth, sample a~pi(.|s), s'~P(.|s,a), r~R(s,a)
				for t in range(depth):
					if self.is_transitions_deterministic == 'True':
						# deterministic transition
						s_next = self.next_state_table[s, a]
					else:
						s_next = sample_discrete(P[s, a, :])
					a_next = sample_discrete(pi[s_next, :])
					r = R[s, a] + np.random.randn(1)[0] * reward_std
					data[i_traj].append((s, a, r, s_next, a_next))
					s = s_next
					a = a_next
				# end for t
			# end for i_traj

		elif sampling_type in {'Generative', 'Generative_uniform'}:
			#  iid sampling according to distribution that was set in the init of current MDP
			probs_s = self.probs_s
			probs_a = self.probs_a
			# Sample data
			for i_traj in range(n_traj):
				data.append([])
				for t in range(depth):
					s = sample_discrete(probs_s)
					a = sample_discrete(probs_a)
					r = R[s, a] + np.random.randn(1)[0] * reward_std
					s_next = sample_discrete(P[s, a, :])
					a_next = sample_discrete(pi[s_next, :])
					# s_next and  a_next is not used in the new iteration since we generate iid samples
					data[i_traj].append((s, a, r, s_next, a_next))
				# end for t
			# end for i_traj

		elif sampling_type == 'Generative_Stationary':
			# iid sampling distribution accordign to the stationary distribution of pi
			probs_s = get_stationary_distrb(self, pi)
			# Sample data
			for i_traj in range(n_traj):
				data.append([])
				for t in range(depth):
					s = sample_discrete(probs_s)
					probs_a = pi[s]
					a = sample_discrete(probs_a)
					r = R[s, a] + np.random.randn(1)[0] * reward_std
					s_next = sample_discrete(P[s, a, :])
					a_next = sample_discrete(pi[s_next, :])
					data[i_traj].append((s, a, r, s_next, a_next))
				# end for t
			# end for i_traj

		elif sampling_type == 'sample_all_s_a':
			for i_traj in range(n_traj):
				data.append([])
				for s in range(nS):
					for a in range(nA):
						r = R[s, a] + np.random.randn(1)[0] * reward_std
						s_next = sample_discrete(P[s, a, :])
						a_next = sample_discrete(pi[s_next, :])
						data[i_traj].append((s, a, r, s_next, a_next))
					# end for a
				# end for s
			# end for i_traj

		elif sampling_type == 'chain_mix_time':
			Pss = self.Pss_fixed
			# generate sampled trajectories
			for i_traj in range(n_traj):
				data.append([])
				# sample initial state:
				s = sample_discrete(p0)
				a = sample_discrete(pi[s, :])
				# Until t==depth, sample a~pi(.|s), s'~Pss(.|s), r~R(s,a)
				# policy is ignored in the dynamics
				for t in range(depth):
					s_next = sample_discrete(Pss[s, :])
					a_next = sample_discrete(pi[s_next, :])
					r = R[s, a] + np.random.randn(1)[0] * reward_std
					data[i_traj].append((s, a, r, s_next, a_next))
					s = s_next
					a = a_next
				# end for t
			# end for i_traj
		else:
			raise AssertionError('Unrecognized data_type')
		return data
# ------------------------------------------------------------------------------------------------------------~


	def generate_RandomMDP(self, args):

		nS = self.nS
		nA = self.nA
		P = np.zeros((nS, nA, nS))
		# For each state-action pair (s; a), the distribution over the next state,  P_{s,a,s'}=P(s'|s,a), is determined by choosing k  non-zero entries uniformly from
		#  all nS states, filling these k entries with values uniformly drawn from [0; 1], and finally normalizing
		k = args.k  # Number of non-zero entries in each row  of transition-matrix
		self.is_transitions_deterministic = False
		for a in range(nA):
			for i in range(nS):
				nonzero_idx = np.random.choice(nS, k, replace=False)
				for j in nonzero_idx:
					P[i, a, j] = np.random.rand(1)
				P[i, a, :] /= P[i, a, :].sum()
		R = np.random.rand(nS, nA)  # rewards means
		return P, R
# ------------------------------------------------------------------------------------------------------------~
	def generate_GridWorld(self, args):

		nS = self.nS
		nA = self.nA
		forward_prob_distrb = args.mdp_def['forward_prob_distrb']
		self.is_transitions_deterministic = forward_prob_distrb == 1
		N0 = args.mdp_def['N0']
		N1 = args.mdp_def['N1']
		action_set = [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]]
		assert args.nA == len(action_set)
		P = np.zeros((nS, nA, nS))
		next_state_table = np.empty((nS, nA), dtype=np.int)

		##### Create reward means: ####
		R_image = (args.mdp_def['R_high'] - args.mdp_def['R_low']) * np.random.rand(nS) + args.mdp_def['R_low']  # draw reward means
		# add goal state
		if args.mdp_def['goal_reward'] is not None:
			goal_reward = args.mdp_def['goal_reward']
			s_goal = np.random.randint(nS)
			R_image[s_goal] = goal_reward
		# translate to R(s,a) matrix
		R = np.zeros((nS, nA))
		for s in range(nS):
			# set same reward mean for all actions in each state
			R[s, :] = R_image[s]

		##### set state transition probs #####
		for s0 in range(N0):
			for s1 in range(N1):
				s = s0 + s1 * N0
				for a, shift in enumerate(action_set):
					s_next0 = s0 + shift[0]
					s_next1 = s1 + shift[1]
					if 0 <= s_next0 < N0 and 0 <= s_next1 < N1 and np.any(action_set[a] != [0, 0]):
						# in case the move is legal and we may change state
						s_next = s_next0 + s_next1 * N0

						if forward_prob_distrb == 1:  # deterministic
							p_forward = 1.
						elif isinstance(forward_prob_distrb, float):  # deterministic
							p_forward = forward_prob_distrb
						elif forward_prob_distrb == 'uniform':  # uniform distribution
							p_forward = np.random.rand(1)
						elif 'beta' in forward_prob_distrb:  # beta distribution
							p_forward = np.random.beta(forward_prob_distrb['alpha'], forward_prob_distrb['beta'],
							                           size=1)
						else:
							raise AssertionError("Invalid args.mdp_def['forward_prob_distrb']")
						# end if

						P[s, a, s_next] = p_forward  # move state probability
						P[s, a, s] = (1. - p_forward)  # stay in place probability
					else:
						# otherwise - stay in place
						s_next = s
						P[s, a, s_next] = 1.
					# end if
					if self.is_transitions_deterministic:
						next_state_table[s, a] = s_next
					# end if
				# end for a
			# end for s1
			# assert np.abs(P[s, a].sum() - 1) < 1e-6
		# end for s0
		if self.is_transitions_deterministic:
			self.next_state_table = next_state_table
		return P, R