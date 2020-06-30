import numpy as np
from numpy import matlib
from random import randrange
from utils.planing_utils import GetUniformPolicy, generalized_greedy, PolicyIteration_GivenRP, draw_policy_at_random


# ------------------------------------------------------------------------------------------------------------~
def run_learning_method(args_r, M, n_traj, gamma_guidance, l2_factor, l1_factor):
    if args_r.method in {'Expected_SARSA','SARSA', 'LSTDQ', 'ELSTDQ', 'LSTDQ_nested', 'ELSTDQ_nested'}:
        pi_t = model_free_approx_policy_iteration(args_r, M, n_traj, gamma_guidance, l2_factor, l1_factor)

    elif args_r.method == 'Model_Based':
        pi_t = model_based_learning(args_r, M, n_traj, gamma_guidance, l2_factor, l1_factor)
    else:
        raise AssertionError('unrecognized method')
    return pi_t



# -------------------------------------------------------------------------------------------

def model_based_learning(args, M, n_traj, gamma_guidance, l2_factor=None, l1_factor=None):
    if l2_factor is not None or l1_factor is not None:
        raise AssertionError('Not supported')
    nS = args.nS
    nA = args.nA
    epsilon = args.epsilon
    # Initial  behaviour policy - this is the policy used for collecting data
    if 'initial_policy' not in args or args.initial_policy == 'uniform':
        pi_b = GetUniformPolicy(nS, nA)
    elif args.initial_policy == 'generated_random':
        pi_b = draw_policy_at_random(nS, nA)
    else:
        raise AssertionError('Unrecognized args.initial_policy')

    pi_t = pi_b.copy()  # Initial  target policy  - the is the policy maintained by policy iteration
    data_history = []  # data from all previous episodes
    # Run episodes
    for i_episode in range(args.n_episodes):
        # Generate data:
        data = M.SampleData(args, pi_b, n_traj, p0=None)
        data_history += data
        # Improve policy:
        # 1. Estimate model:
        P_est, R_est = ModelEstimation(data_history, nS, nA)
        # 2. Certainty-Equivalence policy w.r.t model-estimation and gamma_guidance:
        pi_t, _, _ = PolicyIteration_GivenRP(R_est, P_est, gamma_guidance, args)
        pi_b = (1 - epsilon) * pi_t + (epsilon / nA)
    # end for i_episode
    return pi_t

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
def model_free_approx_policy_iteration(args, M, n_traj, gamma_guidance, l2_factor=None, l1_factor=None):
    nS = args.nS
    nA = args.nA
    data_history = []  # data from all previous episodes

    # Initial  behaviour policy - this is the policy used for collecting data
    if 'initial_policy' not in args or args.initial_policy == 'uniform':
        pi_b = GetUniformPolicy(nS, nA)
    elif args.initial_policy == 'generated_random':
        pi_b = draw_policy_at_random(nS, nA)
    else:
        raise AssertionError('Unrecognized args.initial_policy')

    pi_t = pi_b.copy()  # Initial  target policy  - the is the policy maintained by policy iteration
    Q_est = None  # the Q-function will be initialized in the first evaluation step

    # Run episodes:
    for i_episode in range(args.n_episodes):
        # Generate data:
        data = M.SampleData(args, pi_b,  n_traj, p0=None)
        data_history += data

        # Improve value estimation:
        if args.method in {'Expected_SARSA'}:
            # since this is off-policy evaluation - use all data
            Q_est = run_expected_sarsa(data_history, pi_t, gamma_guidance, args,
                                       initial_Q=Q_est, l2_factor=l2_factor, l1_factor=l1_factor)
        elif args.method in {'LSTDQ', 'ELSTDQ'}:
            # since this is off-policy evaluation - use all data
            Q_est = LSTDQ(data_history, pi_t, gamma_guidance, args, l2_factor=l2_factor, l1_factor=l1_factor)

        elif args.method in {'LSTDQ_nested', 'ELSTDQ_nested'}:
            # since this is off-policy evaluation - use all data
            Q_est = LSTDQ_nested(data_history, pi_t, gamma_guidance, args, l2_factor=l2_factor, l1_factor=l1_factor)

        elif args.method in {'SARSA'}:
            # since this is on-policy evaluation - use data only from current policy
            Q_est = run_sarsa(data, nS, nA, gamma_guidance, args, initial_Q=Q_est, l2_factor=l2_factor, l1_factor=l1_factor)
        else:
            raise AssertionError('unrecognized method')

        # Improve policy:
        pi_t = generalized_greedy(Q_est)

        #  behaviour policy : For exploration set an epsilon-greedy policy:
        epsilon = args.epsilon
        pi_b = (1 - epsilon) * pi_t + (epsilon / nA)
    return pi_t



# ------------------------------------------------------------------------------------------------------------~
# value estimation:
def run_value_estimation_method(data, M, args, gamma_guidance, l2_proj, l2_fp, l2_TD):
    gammaEval = args.gammaEval
    nS = M.nS

    V_true = np.linalg.solve((np.eye(nS) - gammaEval * M.P), M.R)

    if args.alg_type == 'LSTD':
        V_est = LSTD(data, gamma_guidance, args, l2_factor=l2_proj)

    elif args.alg_type == 'LSTD_Nested':
        V_est = LSTD_Nested(data, gamma_guidance, args, l2_proj, l2_fp)

    elif args.alg_type == 'LSTD_Nested_Standard':
        V_est = LSTD_Nested_Standard(data, gamma_guidance, args, l2_proj, l2_fp)

    elif args.alg_type == 'batch_TD_value_evaluation':
        V_est = batch_TD_value_evaluation(data, gamma_guidance, args, l2_factor=l2_TD)

    elif args.alg_type == 'model_based_pol_eval':
        V_est = model_based_pol_eval(data, gamma_guidance, args)

    elif args.alg_type == 'model_based_known_P':
        V_est = model_based_known_P(data, gamma_guidance, args, M)
    else:
        raise AssertionError('Unrecognized args.grid_type')

    return V_est, V_true
    # ------------------------------------------------------------------------------------------------------------~
def set_learning_rate(i_iter, args, gamma_guidance):
    learning_rate_def = args.learning_rate_def
    lr_type = learning_rate_def['type']
    if lr_type == 'const':
        alpha = learning_rate_def['alpha']
    elif lr_type == 'a/(b+i_iter)':
        a = learning_rate_def['a']
        b = learning_rate_def['b']
        alpha = a / (b + i_iter)
    elif args.learning_rate_def == 'a/(b+sqrt(i_iter))':
        a = learning_rate_def['a']
        b = learning_rate_def['b']
        alpha = a / (b + np.sqrt(i_iter))
    else:
        raise AssertionError('Invalid learning_rate_def')
    if learning_rate_def['scale']:
        # explanation to scaling:  according to equivalency proposition, this scaling would make discount reg (using gamma) to behave like using gammaEval but with added reg term.
        gammaEval = args.gammaEval
        alpha *= gammaEval / gamma_guidance
    return alpha

# ------------------------------------------------------------------------------------------------------------~
def set_initial_value(args, shape, gamma_guidance):
    TD_Init_type = args.TD_Init_type
    gammaEval = args.gammaEval
    Rmax = 1  # assumption in mdp creation
    Vmax = Rmax / (1-gammaEval)
    # explanation to scaling:  according to equivalency proposition, this scaling would make discount reg (using gamma) to behave like using gammaEval but with added reg term.
    VmaxGamma = Rmax / (1-gamma_guidance)
    if isinstance(shape, int):
        shape = [shape]
    if TD_Init_type == 'random_0_1':
        x = np.random.rand(*shape)
    elif TD_Init_type == 'random_0_Vmax':
        x = np.random.rand(*shape) * Vmax
    elif TD_Init_type == 'zero':
        x = np.zeros(shape)
    elif TD_Init_type == 'Vmax':
        x = np.ones(shape) * Vmax
    elif TD_Init_type == '0.5_Vmax':
        x = 0.5 * np.ones(shape) / (1 - gammaEval)
    elif TD_Init_type == 'VmaxGamma':
        x = np.ones(shape) * VmaxGamma
    else:
        raise AssertionError('Invalid TD_Init_type')
    return x
# ------------------------------------------------------------------------------------------------------------~
def ModelEstimation(data, nS, nA):
    """
    Maximum-Likelihood estimation of model based on data

    Parameters:
    data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
    S: number of states
    A: number of actions

    Returns:
    P_est: [S x A x S] estimated transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
    R_est: [S x A] estimated mean rewards matrix R
    """

    counts_sas = np.zeros((nS, nA, nS))
    counts_sa = np.zeros((nS, nA))
    R_est = np.zeros((nS, nA))
    P_est = np.zeros((nS, nA, nS))
    for traj in data:
        for sample in traj:
            (s, a, r, s_next, a_next) = sample
            counts_sa[s, a] += 1
            counts_sas[s, a, s_next] += 1
            R_est[s, a] += r

    for s in range(nS):
        for a in range(nA):
            if counts_sa[s, a] == 0:
                # if this state-action doesn't exist in data
                # Use default values:
                R_est[s, a] = 0.5
                P_est[s, a, :] = 1 / nS
            else:
                R_est[s, a] /= counts_sa[s, a]
                P_est[s, a, :] = counts_sas[s, a, :] / counts_sa[s, a]
    if np.any(np.abs(P_est.sum(axis=2) - 1) > 1e-5):
        raise RuntimeError('Transition Probability matrix not normalized!!')
    return P_est, R_est


# ------------------------------------------------------------------------------------------------------------~
def TD_value_evaluation(data, nS, nA, gamma, args):
    """
    Runs TD iterations on data set of samples from unknown policy to estimate the value function of this policy

    Parameters:
    data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
    nS: number of states
    nA: number of actions
    gamma: Discount factor

    Returns:
    V_est: [S] The estimated value-function for a fixed policy pi, i,e. the the expected discounted return when following pi starting from some state
    """

    gammaEval = args.gammaEval
    # Initialization:
    V_est = set_initial_value(args, nS, gamma)

    # prev_V = V_pi.copy()
    # stop_diff = 1e-5  # stopping condition

    # Join list of data tuples from all trajectories:
    data_tuples = sum(data, [])
    n_samples = len(data_tuples)
    for i_iter in range(args.n_TD_iter):
        alpha = set_learning_rate(i_iter, args, gamma)
        # Choose random sample:
        i_sample = randrange(n_samples)
        (s, a, r, s_next, a_next) = data_tuples[i_sample]
        if args.use_reward_scaling:
            r *= gamma / gammaEval
        delta = r + gamma * V_est[s_next] - V_est[s]
        V_est[s] += alpha * delta
    # end for i_iter
        # if i_iter > 0 and (i_iter % 10000 == 0):
        #     if np.linalg.norm(V_pi - prev_V) < stop_diff:
        #         break
        #     prev_V = V_pi
    return V_est



# ------------------------------------------------------------------------------------------------------------~
def run_sarsa(data, nS, nA, gamma, args, initial_Q=None, l2_factor=None, l1_factor=None):
    """
      Runs TD iterations on data set of samples from unknown policy to estimate the Q-function of this policy
      using SARSA algorithm

    Parameters:
    data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
    nS: number of states
    nA: number of actions
    gamma: Discount factor

    Returns:
    Q_est [S x A] The estimated Q-function for a fixed policy pi, i,e. the the expected discounted return when following pi starting from some state and action
    """
    gammaEval = args.gammaEval
    # Initialization:
    if initial_Q is None:
        Q_est = set_initial_value(args, (nS, nA), gamma)
    else:
        Q_est = initial_Q

    # prev_V = V_pi.copy()
    # stop_diff = 1e-5  # stopping condition

    # Join list of data tuples from all trajectories:
    data_tuples = sum(data, [])
    n_samples = len(data_tuples)
    for i_iter in range(args.n_TD_iter):
        alpha = set_learning_rate(i_iter, args, gamma)
        # Choose random sample:
        i_sample = randrange(n_samples)
        (s, a, r, s_next, a_next) = data_tuples[i_sample]
        if args.use_reward_scaling:
            r *= gamma / gammaEval
        delta = r + gamma * Q_est[s_next, a_next] - Q_est[s, a]
        Q_est[s, a] += alpha * delta

        # Add the gradient of the added regularization term:
        if l2_factor is not None:
            reg_grad = 2 * l2_factor * Q_est  # gradient of the L2 regularizer [tabular case]
            Q_est -= alpha * reg_grad

        if l1_factor is not None:
            reg_grad = 1 * l1_factor * np.sign(Q_est)  # gradient of the L2 regularizer [tabular case]
            Q_est -= alpha * reg_grad
    # end for i_iter

    return Q_est


# ------------------------------------------------------------------------------------------------------------~
def run_expected_sarsa(data, pi, gamma, args, initial_Q=None, l2_factor=None, l1_factor=None):
    """
      Runs TD iterations on data set of samples from given  policy to estimate the Q-function of this policy
       using Expected-SARSA algorithm

    Parameters:
    data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
    pi: the policy that generated that generated the data
    S: number of states
    A: number of actions
    gamma: Discount factor

    Returns:
    Q_est [S x A] The estimated Q-function for the fixed policy pi, i,e. the the expected discounted return when following pi starting from some state and action
    """
    if pi.ndim != 2:
        raise AssertionError('Invalid input')
    S = pi.shape[0]
    A = pi.shape[1]
    gammaEval = args.gammaEval
    # Initialization:
    if initial_Q is None:
        Q_est = set_initial_value(args, (S, A), gamma)
    else:
        Q_est = initial_Q

    # prev_V = V_pi.copy()
    # stop_diff = 1e-5  # stopping condition

    # Join list of data tuples from all trajectories:
    data_tuples = sum(data, [])
    n_samples = len(data_tuples)
    for i_iter in range(args.n_TD_iter):
        alpha = set_learning_rate(i_iter, args, gamma)
        # Choose random sample:
        i_sample = randrange(n_samples)
        (s, a, r, s_next, a_next) = data_tuples[i_sample]
        if args.use_reward_scaling:
            r *= gamma / gammaEval
        V_next = np.dot(Q_est[s_next, :], pi[s_next, :])
        delta = r + gamma * V_next - Q_est[s, a]
        Q_est[s, a] += alpha * delta

        # Add the gradient of the added regularization term:
        if l2_factor is not None:
            reg_grad = 2 * l2_factor * Q_est  # gradient of the L2 regularizer [tabular case]
            Q_est -= alpha * reg_grad

        if l1_factor is not None:
            reg_grad = 1 * l1_factor * np.sign(Q_est)  # gradient of the L2 regularizer [tabular case]
            Q_est -= alpha * reg_grad
    # end for i_iter
    return Q_est
# ------------------------------------------------------------------------------------------------------------~


def LSTDQ(data, pi, gamma, args, l2_factor=None, l1_factor=None):
    """
       given  policy to estimate the Q-function of this policy
       using LSTDQ algorithm
       "Least-squares policy iteration, MG Lagoudakis, R Parr - Journal of machine learning research, 2003"

    Parameters:
    data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
    pi: the policy that generated that generated the data
    S: number of states
    A: number of actions
    gamma: Discount factor

    Returns:
    Q_est [S x A] The estimated Q-function for the fixed policy pi, i,e. the the expected discounted return when following pi starting from some state and action
    """
    if l1_factor is not None:
        raise AssertionError('Not supported yet')

    if l2_factor is None:
        l2_factor = args.default_l2_factor  # to prevent  Singular matrix
    if pi.ndim != 2:
        raise AssertionError('Invalid input')
    nS = pi.shape[0]
    nA = pi.shape[1]
    gammaEval = args.gammaEval
    # Join list of data tuples from all trajectories:
    data_tuples = sum(data, [])

    n_samples = len(data_tuples)
    n_feat = nS * nA
    Amat = np.zeros((n_feat, n_feat))
    bmat = np.zeros((n_feat, 1))

    for i_samp in range(n_samples):
        (s, a, r, s_next, a_next) = data_tuples[i_samp]
        if args.use_reward_scaling:
            r *= gamma / gammaEval
            raise AssertionError('use_reward_scaling should be False with LSTDQ')
        ind1 = s_a_to_ind(s, a, nS, nA)
        Amat[ind1, ind1] += 1
        bmat[ind1] += r
        if args.method == 'LSTDQ':
            # SARSA style update
            ind2 = s_a_to_ind(s_next, a_next, nS, nA)
            Amat[ind1, ind2] -= gamma
        elif args.method == 'ELSTDQ':
            # Expected SARSA style update
            for a_prime in range(nA):
                ind2 = s_a_to_ind(s_next, a_prime, nS, nA)
                Amat[ind1, ind2] -= gamma * pi[s, a_prime]
        else:
            raise AssertionError

    Qest_vec = np.linalg.solve(Amat + l2_factor * np.eye(n_feat), bmat)
    Q_est = np.reshape(Qest_vec, (nS, nA))
    return Q_est
# ------------------------------------------------------------------------------------------------------------~


def s_a_to_ind(s, a, nS, nA):
    ind = s * nA + a
    return ind
# ------------------------------------------------------------------------------------------------------------~


def ind_to_s_a(ind, nS, nA):
    s = ind // nA
    a = ind % nA
    return s, a
# ------------------------------------------------------------------------------------------------------------~


def LSTDQ_nested(data, pi, gamma, args, l2_factor=None, l1_factor=None):
    """
       given  policy to estimate the Q-function of this policy
       using LSTDQ algorithm
       "Least-squares policy iteration, MG Lagoudakis, R Parr - Journal of machine learning research, 2003"
            based on:
     regularized least-squares temporal difference learning with nested ℓ 2 and ℓ 1 penalizationMW Hoffman, A Lazaric, M Ghavamzadeh, R Munos - European Workshop on …, 2011


    Parameters:
    data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
    pi: the policy that generated that generated the data
    S: number of states
    A: number of actions
    gamma: Discount factor

    Returns:
    Q_est [S x A] The estimated Q-function for the fixed policy pi, i,e. the the expected discounted return when following pi starting from some state and action
    """
    if l1_factor is not None:
        raise AssertionError('Not supported yet')

    if l2_factor is None:
        l2_factor = args.default_l2_factor  # to prevent  Singular matrix
    if pi.ndim != 2:
        raise AssertionError('Invalid input')
    nS = pi.shape[0]
    nA = pi.shape[1]
    gammaEval = args.gammaEval
    # Join list of data tuples from all trajectories:
    data_tuples = sum(data, [])

    n_samples = len(data_tuples)
    n_feat = nS * nA + 1  # +1 for bias

    I = np.eye((n_feat))
    Phi = np.zeros((n_samples, n_feat))  # For each sample: feature of current state
    PhiPrime = np.zeros((n_samples, n_feat))   # For each sample: feature of next state
    R = np.zeros((n_samples, 1))  # For each sample: reward

    for i_samp in range(n_samples):
        (s, a, r, s_next, a_next) = data_tuples[i_samp]
        if args.use_reward_scaling:
            r *= gamma / gammaEval
        ind1 = s_a_to_ind(s, a, nS, nA)
        Phi[i_samp, ind1] = 1.
        Phi[i_samp, -1] = 1.  # for bias
        PhiPrime[i_samp, -1] = 1.  # for bias
        R[i_samp] = r

        if args.method == 'LSTDQ_nested':
            # SARSA style update
            ind2 = s_a_to_ind(s_next, a_next, nS, nA)
            PhiPrime[i_samp, ind2] = 1.
        elif args.method == 'ELSTDQ_nested':
            # Expected SARSA style update
            for a_prime in range(nA):
                ind2 = s_a_to_ind(s_next, a_prime, nS, nA)
                PhiPrime[i_samp, ind2] = pi[s, a_prime]
        else:
            raise AssertionError

    l2_proj = l2_fp = l2_factor
    PhiBar = Phi.mean(axis=0)  # features means
    PhiPrimeBar = PhiPrime.mean(axis=0)  # features means
    RBar = R.mean(axis=0)
    PhiTilde = Phi - PhiBar
    PhiPrimeTilde = PhiPrime - PhiPrimeBar
    Rtilde = R - RBar
    sigmaPhi = PhiTilde.std(axis=0)
    sigmaPhi[sigmaPhi == 0] = 1.
    PhiHat = PhiTilde / sigmaPhi
    SigmaMat = PhiHat @ np.linalg.inv(PhiHat.T @ PhiHat + l2_proj * np.eye(n_feat)) @ PhiHat.T
    Xmat = Phi - gamma * SigmaMat @ PhiPrimeTilde - gamma * matlib.repmat(PhiPrimeBar, n_samples, 1)
    yMat = SigmaMat @ Rtilde + matlib.repmat(RBar, n_samples, 1)
    Amat = Xmat.T @ Xmat + l2_fp * np.eye(n_feat)
    bmat = Xmat.T @ yMat
    theta_vec = np.linalg.solve(Amat, bmat)
    Qest_vec = theta_vec[:-1] + theta_vec[-1]  #  add bias term ... Q(s,a) = (phi.T @ theta)_{s,a} =  theta[s,a] + theta[-1]
    Q_est = np.reshape(Qest_vec, (nS, nA))
    return Q_est
# ------------------------------------------------------------------------------------------------------------~


def LSTD(data, gamma, args, l2_factor):
    """
       given  policy to estimate the Q-function of this policy
       using LSTDQ algorithm

    Parameters:
    data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
    pi: the policy that generated that generated the data
    S: number of states
    A: number of actions
    gamma: Discount factor

    Returns:
    V_est [nS x 1] The estimated Q-function for the fixed policy pi, i,e. the the expected discounted return when following pi starting from some state and action
    """

    # Join list of data tuples from all trajectories:
    data_tuples = sum(data, [])

    nS = args.nS
    gammaEval = args.gammaEval
    n_samples = len(data_tuples)
    n_feat = nS
    Amat = np.zeros((n_feat, n_feat))
    bmat = np.zeros((n_feat, 1))

    for i_samp in range(n_samples):
        (s, r, s_next) = data_tuples[i_samp]
        if args.use_reward_scaling:
            r *= gamma / gammaEval
        Amat[s, s] += 1
        Amat[s, s_next] -= gamma
        bmat[s] += r

    Vest_vec = np.linalg.solve(Amat + l2_factor * np.eye(n_feat), bmat)
    V_est = np.reshape(Vest_vec, (nS, 1))
    return V_est

# ------------------------------------------------------------------------------------------------------------~
def LSTD_Nested(data, gamma, args, l2_proj=0., l2_fp=0.):
    """
     based on:
     regularized least-squares temporal difference learning with nested ℓ 2 and ℓ 1 penalizationMW Hoffman, A Lazaric, M Ghavamzadeh, R Munos - European Workshop on …, 2011

    Parameters:
    data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
    pi: the policy that generated that generated the data
    S: number of states
    A: number of actions
    gamma: Discount factor

    Returns:
    V_est [nS x 1] The estimated Q-function for the fixed policy pi, i,e. the the expected discounted return when following pi starting from some state and action
    """

    # Join list of data tuples from all trajectories:
    data_tuples = sum(data, [])

    nS = args.nS
    gammaEval = args.gammaEval
    n_samples = len(data_tuples)
    n_feat = nS
    Amat = np.zeros((n_feat, n_feat))
    bmat = np.zeros((n_feat, 1))
    I = np.eye((n_feat))
    Phi = I # tabular case

    PhiTPhi = np.zeros((n_feat, n_feat))
    PhiTPhiPrime = np.zeros((n_feat, n_feat))

    for i_samp in range(n_samples):
        (s, r, s_next) = data_tuples[i_samp]
        if args.use_reward_scaling:
            r *= gamma / gammaEval
        PhiTPhi[s, s] += 1
        PhiTPhiPrime[s, s_next] += 1
        Amat[s, s] += 1
        Amat[s, s_next] -= gamma
        bmat[s] += r

    # Amat = PhiTPhi - gamma * PhiTPhiPrime
    Cmat = Phi @ np.linalg.inv(PhiTPhi + l2_proj * I)
    Xmat = Cmat @ (Amat + l2_proj * I)
    yMat = Cmat @ bmat
    theta_vec =  np.linalg.solve(Xmat.T @ Xmat + l2_fp * I,  Xmat.T @ yMat)

    V_est = np.reshape(theta_vec, (nS, 1))
    return V_est

# ------------------------------------------------------------------------------------------------------------~
def LSTD_Nested_Standard(data, gamma, args, l2_proj=0., l2_fp=0.):
    """
     based on:
     regularized least-squares temporal difference learning with nested ℓ 2 and ℓ 1 penalizationMW Hoffman, A Lazaric, M Ghavamzadeh, R Munos - European Workshop on …, 2011

    Parameters:
    data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
    pi: the policy that generated that generated the data
    S: number of states
    A: number of actions
    gamma: Discount factor

    Returns:
    V_est [nS x 1] The estimated Q-function for the fixed policy pi, i,e. the the expected discounted return when following pi starting from some state and action
    """

    # Join list of data tuples from all trajectories:
    data_tuples = sum(data, [])

    nS = args.nS
    gammaEval = args.gammaEval
    n_samples = len(data_tuples)
    n_feat = nS + 1  # +1 for bias

    I = np.eye((n_feat))

    Phi = np.zeros((n_samples, n_feat))  # For each sample: feature of current state
    PhiPrime = np.zeros((n_samples, n_feat))   # For each sample: feature of next state
    R = np.zeros((n_samples, 1))  # For each sample: reward

    for i_samp in range(n_samples):
        (s, r, s_next) = data_tuples[i_samp]
        if args.use_reward_scaling:
            r *= gamma / gammaEval
        Phi[i_samp, s] = 1.
        Phi[i_samp, -1] = 1.  # for bias
        PhiPrime[i_samp, s_next] = 1.
        PhiPrime[i_samp, -1] = 1.  # for bias
        R[i_samp] = r

    PhiBar = Phi.mean(axis=0)  # features means
    PhiPrimeBar = PhiPrime.mean(axis=0)  # features means
    RBar = R.mean(axis=0)


    PhiTilde = Phi - PhiBar
    PhiPrimeTilde = PhiPrime - PhiPrimeBar
    Rtilde = R - RBar

    sigmaPhi = PhiTilde.std(axis=0)
    sigmaPhi[sigmaPhi == 0] = 1.

    PhiHat = PhiTilde / sigmaPhi

    SigmaMat = PhiHat @ np.linalg.inv(PhiHat.T @ PhiHat + l2_proj * np.eye(n_feat)) @ PhiHat.T
    Xmat = Phi - gamma * SigmaMat @ PhiPrimeTilde - gamma * matlib.repmat(PhiPrimeBar, n_samples, 1)
    yMat = SigmaMat @ Rtilde + matlib.repmat(RBar, n_samples, 1)
    Amat = Xmat.T @ Xmat + l2_fp * np.eye(n_feat)
    bmat = Xmat.T @ yMat
    theta_vec = np.linalg.solve(Amat, bmat)
    V_est = theta_vec[:-1] + theta_vec[-1]  #  add bias term ... V(s) = (phi.T @ theta)_s =  theta[s] + theta[-1]

    return V_est

# ------------------------------------------------------------------------------------------------------------~

# # ------------------------------------------------------------------------------------------------------------~
def batch_TD_value_evaluation(data, gamma, args, l2_factor=None):
    """
    Runs TD iterations on data set of samples from unknown policy to estimate the value function of this policy

    Parameters:
    data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
    gamma: Discount factor

    Returns:
    V_est: [S] The estimated value-function for a fixed policy pi, i,e. the the expected discounted return when following pi starting from some state
    """
    nS = args.nS
    gammaEval = args.gammaEval
    # Initialization:
    V_est = set_initial_value(args, nS, gamma)

    # Join list of data tuples from all trajectories:
    data_tuples = sum(data, [])

    n_samples = len(data_tuples)

    for i_iter in range(args.n_TD_iter):
        alpha = set_learning_rate(i_iter, args, gamma)
        # Choose random sample:
        i_sample = randrange(n_samples)
        (s, r, s_next) = data_tuples[i_sample]
        if args.use_reward_scaling:
            r *= gamma / gammaEval
        delta = r + gamma * V_est[s_next] - V_est[s]
        V_est[s] += alpha * delta
        # Add the gradient of the added regularization term:
        if l2_factor is not None:
            reg_grad = 2 * l2_factor * V_est  # gradient of the L2 regularizer [tabular case]
            V_est -= alpha * reg_grad
        # end if
    # end for i_iter
    return V_est
# # ------------------------------------------------------------------------------------------------------------~
# ------------------------------------------------------------------------------------------------------------~


def ModelEstimation_MRP(data, args):
    """
    Maximum-Likelihood estimation of model based on data

    Parameters:
    data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
    S: number of states
    A: number of actions

    Returns:
    P_est: [S x A x S] estimated transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
    R_est: [S x A] estimated mean rewards matrix R
    """
    nS = args.nS
    counts_ss = np.zeros((nS, nS))
    counts_s = np.zeros((nS))
    R_est = np.zeros((nS))
    P_est = np.zeros((nS, nS))
    for traj in data:
        for sample in traj:
            (s, r, s_next) = sample
            counts_s[s] += 1
            counts_ss[s, s_next] += 1
            R_est[s] += r
        # end for sample
    # end for sample

    for s in range(nS):
        if counts_s[s] == 0:
            # if this state-action doesn't exist in data
            # Use default values:
            R_est[s] = 0.5
            P_est[s, :] = 1 / nS
        else:
            R_est[s] /= counts_s[s]
            P_est[s, :] = counts_ss[s, :] / counts_s[s]
        # end if
    # end for s
    if np.any(np.abs(P_est.sum(axis=1) - 1) > 1e-5):
        raise RuntimeError('Transition Probability matrix not normalized!!')
    return P_est, R_est
# # ------------------------------------------------------------------------------------------------------------~


def model_based_pol_eval(data, gamma, args, l2_factor=None):
    if l2_factor is not None:
        raise AssertionError('Not supported yet')
    nS = args.nS
    P_est, R_est = ModelEstimation_MRP(data, args)
    V_est = np.linalg.solve((np.eye(nS) - gamma * P_est), R_est)
    return V_est
# # ------------------------------------------------------------------------------------------------------------~


def model_based_known_P(data, gamma, args, M, l2_factor=None):
    if l2_factor is not None:
        raise AssertionError('Not supported yet')
    nS = args.nS
    P = M.P #  get known P
    _, R_est = ModelEstimation_MRP(data, args)
    V_est = np.linalg.solve((np.eye(nS) - gamma * P), R_est)
    return V_est
# # ------------------------------------------------------------------------------------------------------------~
