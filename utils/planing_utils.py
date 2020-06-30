import numpy as np
from scipy import stats
from utils.prob_utils import sample_simplex

#------------------------------------------------------------------------------------------------------------~


def generalized_argmax_indicator(x):
    argmax_size = np.sum(x == x.max())
    indc = np.zeros_like(x)
    indc[x == x.max()] = 1 / argmax_size
    return indc


#------------------------------------------------------------------------------------------------------------~


def generalized_greedy(Q):
    """
    Calculates a greedy policy w.r.t Q
    if all Q values are distinct then we derive a deterministic policy.
    If several Q values are equal, the probability is divided among them

    Parameters:
     Q [S x A]  Q-function

    Returns:
    pi: [S x A]  matrix representing  pi(a|s)
    """
    if Q.ndim != 2:
        raise AssertionError('Invalid input')
    S = Q.shape[0]
    A = Q.shape[1]
    pi = np.zeros((S,A))
    for s in range(S):
        pi[s] = generalized_argmax_indicator(Q[s])
    return pi

#------------------------------------------------------------------------------------------------------------~


def GetUniformPolicy(nS, nA):
    """
    Create a Markov stochastic policy which chooses actions randomly uniform from each state

    Parameters:
    nS: number of states
    nA: number of actions

    Returns:
    pi: [S x A]  matrix representing  pi(a|s)
    """
    pi = np.ones((nS, nA))
    for i in range(nS):
        pi[i] /= pi[i].sum()
    return pi
#-------------------------------------------------------------------------------------------


def draw_policy_at_random(nS, nA):
    pi = np.zeros((nS, nA))
    for i in range(nS):
        pi[i] = sample_simplex(nA)
    return pi
#------------------------------------------------------------------------------------------------------------~


def GetPolicyDynamics(P, R, pi):
    """
    Calculate the dynamics when following the policy pi

    Parameters:
    P: [nS x nA x nS] transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
    R: [nS x nA] mean rewards matrix R
    pi: [nS x nA]  matrix representing  pi(a|s)

    Returns:
    P_pi: [nS x nS] transitions matrix  when following the policy pi      (P_pi)_{s,s'} P^pi(s'|s)
    R_pi: [nS] mean rewards at each state when following the policy pi    (R_pi)_{s} = R^pi(s)
    """
    if P.ndim != 3 or R.ndim != 2 or pi.ndim != 2:
        raise AssertionError('Invalid input')
    nS = P.shape[0]
    nA = P.shape[1]
    P_pi = np.zeros((nS, nS))
    R_pi = np.zeros((nS))
    for i in range(nS):  # current state
        for a in range(nA):
            for j in range(nS): # next state
                # Explanation: P(s'|s) = sum_a pi(a|s)P(s'|s,a)
                P_pi[i, j] += pi[i, a] * P[i,a,j]
                is_row_updated = True
            R_pi[i] += pi[i, a] * R[i,a]

    if np.any(np.abs(P_pi.sum(axis=1) - 1) > 1e-5):
        raise RuntimeError('Probabilty matrix not normalized!!')
    return P_pi, R_pi

#------------------------------------------------------------------------------------------------------------~


def PolicyEvaluation(M, pi, gamma, P_pi=None, R_pi=None):
    """
    Calculates the value-function for a given policy pi and a known model

    Parameters:
    P: [nS x nA x nS] transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
    R: [nS x nA] mean rewards matrix R
    pi: [nS x nA]  matrix representing  pi(a|s)
    gamma: Discount factor

    Returns:
    V_pi: [nS] The value-function for a fixed policy pi, i,e. the the expected discounted return when following pi starting from some state
    Q_pi [nS x nA] The Q-function for a fixed policy pi, i,e. the the expected discounted return when following pi starting from some state and action
    """
    # (1) Use PolicyDynamics to get P and R, (2) V = (I-gamma*P)^-1 * R
    P = M.P
    R = M.R
    if P.ndim != 3 or R.ndim != 2 or pi.ndim != 2:
        raise AssertionError('Invalid input')
    nS = P.shape[0]
    nA = P.shape[1]
    if P_pi is None or R_pi is None:
        P_pi, R_pi = GetPolicyDynamics(P, R, pi)
    V_pi = np.linalg.solve((np.eye(nS) - gamma * P_pi), R_pi)
    # Verify that R_pi + gamma * np.matmul(P_pi,  V_pi) == V_pi
    Q_pi = np.zeros((nS, nA))
    for a in range(nA):
        for i in range(nS):
            Q_pi[i, a] = R[i, a] + gamma * np.matmul(P[i,a,:], V_pi)
    # Verify that V_pi(s) = sum_a pi(a|s) * Q_pi(s,a)
    return V_pi, Q_pi
#------------------------------------------------------------------------------------------------------------~


def PolicyIteration(M, gamma):
    """
       Finds the optimal policy given a known model using policy-iteration algorithm

       Parameters:
       P: [nS x nA x nS] transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
       R: [nS x nA] mean rewards matrix R
       gamma: Discount factor

       Returns
       pi_opt [nS x nA]: An optimal policy (assuming given model and gamma)
       V_opt: [nS] The optimal value-function , i,e. the the expected discounted return when following optimal policy  starting from some state
       Q_opt [nS x nA] The optimal Q-function, i,e. the the expected discounted return when following optimal policy starting from some state and action
       """

    # The algorithm: until policy not changes: (1) run policy-evaluation to get Q_pi  (2) new_policy = argmax Q
    nS = M.nS
    nA = M.nA
    Q_pi = np.zeros((nS, nA))
    # initial point of the algorithm: uniform policy
    pi = np.ones((nS, nA)) / nA
    pi_prev = nA - pi # arbitrary different policy than pi
    max_iter = nS*nA
    iter = 0
    while np.any(pi != pi_prev):
        pi_prev = pi
        _, Q_pi = PolicyEvaluation(M, pi, gamma)
        # Policy improvement:
        # pi = np.zeros((nS, nA))
        # pi[np.arange(nS), np.argmax(Q_pi, axis=1)] = 1  #  set 1 for the optimal action w.r.t Q, and 0 for the other actions
        pi = generalized_greedy(Q_pi)
        if iter > max_iter:
            raise RuntimeError('Policy Iteration should have stopped by now!')
        iter += 1

    pi_opt = pi
    Q_opt = Q_pi
    V_opt = np.max(Q_opt, axis=1)
    return pi_opt,  V_opt, Q_opt
# ------------------------------------------------------------------------------------------------------------~


def PolicyIteration_GivenRP(R, P, gamma, args):
    from utils.mdp_utils import MDP
    M = MDP(args)
    M.R = R
    M.P = P
    return PolicyIteration(M, gamma)
# ------------------------------------------------------------------------------------------------------------~


def ValueOfGreedyPolicyForV(V, M, gammaEval):
    P = M.P
    R = M.R
    nS = M.nS
    nA = M.nA
    Q_pi = np.zeros((nS, nA))
    for a in range(nA):
        for i in range(nS):
            Q_pi[i, a] = R[i, a] + gammaEval * np.matmul(P[i, a, :], V)
    # pi = np.zeros((nS, nA))
    # pi[np.arange(nS), np.argmax(Q_pi, axis=1)] = 1  # set 1 for the optimal action w.r.t Q, and 0 for the other actions
    pi = generalized_greedy(Q_pi)
    Vgreedy, _ = PolicyEvaluation(M, pi, gammaEval)
    return Vgreedy
# ------------------------------------------------------------------------------------------------------------~


def BellmanErr(V, M, pi, gammaEval):
    P = M.P
    R = M.R
    P_pi, R_pi = GetPolicyDynamics(P, R, pi)
    errVec = V - (R_pi + gammaEval * np.matmul(P_pi, V))
    return errVec
# ------------------------------------------------------------------------------------------------------------~


def evaluate_value_estimation(loss_type, V_pi, V_est, M, pi, gammaEval, gamma_guidance):
    if loss_type == 'correction_scaling':
        # Correction factor:
        V_est = V_est * (1 / (1 - gammaEval)) / (1 / (1 - gamma_guidance))
        eval_loss = np.abs(V_pi - V_est).mean()

    elif loss_type =='L1_normalized':
        V_est_norm = np.sum(np.abs(V_est))
        V_pi_norm = np.sum(np.abs(V_pi))
        eval_loss = np.abs(V_pi / V_pi_norm - V_est / V_est_norm).mean()

    # elif loss_type =='Lmax_normalized':
    #     V_est_norm = np.max(np.abs(V_est))
    #     V_pi_norm = np.max(np.abs(V_pi))
    #     eval_loss = np.abs(V_pi / V_pi_norm - V_est / V_est_norm).mean()

    elif loss_type =='Lmax':
        eval_loss = np.abs(V_pi - V_est).max()


    elif loss_type =='L2_normalized':
        V_est_norm = np.linalg.norm(V_est)
        V_pi_norm = np.linalg.norm(V_pi)
        eval_loss = np.sqrt(np.square(V_pi / V_pi_norm - V_est / V_est_norm).mean())

    elif loss_type =='L2':
        eval_loss = np.sqrt(np.square(V_pi - V_est).sum())

    elif loss_type =='L2_uni_weight':
        eval_loss = np.sqrt(np.mean(np.square(V_pi - V_est)))


    elif loss_type == 'one_pol_iter_l2_loss':
        # Optimal policy for the MDP:
        pi_opt, V_opt, Q_opt = PolicyIteration(M, gammaEval)
        V_est_g = ValueOfGreedyPolicyForV(V_est, M, gammaEval)
        eval_loss = (np.square(V_opt - V_est_g)).mean()

    elif loss_type == 'greedy_V_L1':
        V_pi_g = ValueOfGreedyPolicyForV(V_pi, M, gammaEval)
        V_est_g = ValueOfGreedyPolicyForV(V_est, M, gammaEval)
        eval_loss = np.abs(V_pi_g - V_est_g).mean()

    elif loss_type == 'greedy_V_L_infty':
        V_pi_g = ValueOfGreedyPolicyForV(V_pi, M, gammaEval)
        V_est_g = ValueOfGreedyPolicyForV(V_est, M, gammaEval)
        eval_loss = np.abs(V_pi_g - V_est_g).max()

    elif loss_type =='Bellman_Err':
        # V_pi_err = BellmanErr(V_pi, P, R, pi, gammaEval) # should be very close to 0
        V_est_err = BellmanErr(V_est, M, pi, gammaEval)
        eval_loss = np.abs(V_est_err).mean()

    elif loss_type == 'Values_SameGamma':
        V_pi_gamma, _ = PolicyEvaluation(M, pi, gamma_guidance)
        eval_loss = np.abs(V_est - V_pi_gamma).mean()

    elif loss_type == 'rankings_kendalltau':
        tau, p_value = stats.kendalltau(V_est, V_pi)
        eval_loss = -tau
    elif loss_type == 'rankings_spearmanr':
        rho, pval = stats.spearmanr(V_est, V_pi)
        eval_loss = -rho
    # elif loss_type == 'greedy_V_L_infty_MRP':
    #     V_pi_g = ValueOfGreedyPolicyForV(V_pi, M, gammaEval)
    #     V_est_g = ValueOfGreedyPolicyForV(V_est, M, gammaEval)
    #     eval_loss = np.abs(V_pi_g - V_est_g).max()

    else:
        raise AssertionError('unrecognized evaluation_loss_type')
    return eval_loss


# ------------------------------------------------------------------------------------------------------------~
def get_stationary_distrb(M, pi):
    P_pi, R_pi = GetPolicyDynamics(M.P, M.R, pi)
    evals, evecs = np.linalg.eig(P_pi.T)  # det left eigenvectors of P_pi
    evals_abs = np.abs(evals)
    sort_inds = np.argsort(evals_abs)
    ind_of_largest = sort_inds[-1]
    dVec = evecs[:, ind_of_largest].copy()
    dVec = np.real_if_close(dVec)
    dVec /= dVec.sum()

    ### DEBUG : verify that p -> dVec ######
    # p = np.ones(M.nS) / M.nS
    # for i in range(1000):
    #     p = p @ P_pi
    # print(p)
    ######
    return dVec
