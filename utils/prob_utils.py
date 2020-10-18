import numpy as np
import scipy.stats as sps

import random
from utils.common_utils import write_to_log

def sample_discrete(probs):
    """
    Samples a discrete distribution
     Parameters:
        probs - probabilities over {0,...K-1}
    Returns:
        drawn random integer from  {0,...K-1} according to probs
    """
    K = probs.size
    return np.random.choice(K, size=1, p=probs)[0]


# ------------------------------------------------------------------------------------------------------------~
def draw_prob_at_dist(n, d=0.):
    """
    Randomly generates a discrete distribution of dimension n with L1 distance of d from the uniform distribution.
    """
    if d == 0.:
        return np.ones(n) / n
    diffs = np.random.uniform(size=n)
    diffs = diffs - diffs.mean() # center to make sum(diffs)==0
    diffs = diffs * d / np.sum(np.abs(diffs))  # scale distance
    probs = diffs + (1/n)  # add diffs to a uniform prob
    probs = simplex_projection(probs)
    assert np.all(0 <= probs)
    return probs

# ------------------------------------------------------------------------------------------------------------~
def generate_discrete_prob_TV(args, n, tv_dist_nrml=0., tol=1e-1, max_iter=5e6):
    """
    Randomly generates a discrete distribution of dimension n with L1 distance of d from the uniform distribution.
    assumption tv_dist is normalized in [0,1]
    """
    assert 0 <= tv_dist_nrml <= 1

    if tv_dist_nrml == 0.:
        return np.ones(n) / n

    tv_dist_max = 0.5 * 2 * (n - 1) / n  # note: tv_dist = 0.5 * L1 dist
    tv_dist = tv_dist_nrml * tv_dist_max  # de-normalize

    done = False
    i = 0
    probs = None
    best_probs = None
    best_err = float('inf')
    l1_dist = 2 * tv_dist # the desired L1 dist
    best_tv_dist_nrml = None
    while not done:
        i += 1
        # probs = sample_simplex(n)   # unifromly
        probs = draw_prob_at_dist(n, l1_dist)
        #  Check that we indeed got the correct L1 distance from a uniform distrib
        curr_l1_dist = np.sum(np.abs(probs-1/n))
        curr_tv_dist_nrml = 0.5 * curr_l1_dist / tv_dist_max
        curr_err = np.abs(curr_tv_dist_nrml - tv_dist_nrml)
        if curr_err < tol:
            done = True
        if curr_err < best_err:
            best_err = curr_err
            best_probs = probs
            best_tv_dist_nrml = curr_tv_dist_nrml
        if i >= max_iter:
            write_to_log(['rejection sampling failed -', 'desired tv_dist [normalized]: ', tv_dist_nrml,
                          ', best_tv_dist_nrml: ', best_tv_dist_nrml, 'best_err: ', best_err], args)
            probs = best_probs
            done = True
        # if i >= max_iter:
        #     write_to_log(['n ', n, 'tv_dist ', tv_dist, 'tol ', tol, 'i ', i,
        #     'desired l1_dist', l1_dist, 'last l1_dist', curr_l1_dist], args)
        #     raise AssertionError('rejection sampling failed')
    # print(i)
    return probs

# ------------------------------------------------------------------------------------------------------------~

# ------------------------------------------------------------------------------------------------------------~
def generate_prob_given_entropy(args, n, ent_nrml, tol=1e-1, max_iter=5e6):
    """
    Randomly generates a discrete distribution of dimension n with given entropy ent.
    assumption ent is normalized in [0,1]
    """
    assert 0. <= ent_nrml <= 1.

    if ent_nrml == 1.:
        return np.ones(n) / n

    ent_max = np.log2(n)
    ent = ent_nrml * ent_max  # de-normalize

    # use rejection sampling
    done = False
    i = 0
    probs = None
    best_probs = None
    best_err = float('inf')
    best_ent_nrml = None
    while not done:
        i += 1
        probs = sample_simplex(n)
        curr_ent = sps.entropy(probs, base=2)
        curr_ent_nrml = curr_ent / ent_max
        cur_err = np.abs(ent_nrml - curr_ent_nrml)
        if cur_err < best_err:
            best_err = cur_err
            best_probs = probs
            best_ent_nrml= curr_ent_nrml
        if cur_err < tol:
            done = True
        if i >= max_iter:
            write_to_log(['rejection sampling failed -', 'desired ent: ', ent, 'best_err: ', best_err, ', best_ent_nrml: ', best_ent_nrml], args)
            probs = best_probs
            done = True
            # write_to_log(['n ', n, 'ent ', ent, 'tol ', tol, 'i ', i, 'last ent', curr_ent], args)
            # raise AssertionError('rejection sampling failed')
    return probs
# ------------------------------------------------------------------------------------------------------------~

def simplex_projection(s):
    """Projection onto the unit simplex.
    source: https://ee227c.github.io/code/lecture5.html"""
    if np.sum(s) <=1 and np.alltrue(s >= 0):
        return s
    # Code taken from https://gist.github.com/daien/1272551
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(s)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - 1) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    probs = np.maximum(s-theta, 0)
    probs /= probs.sum()  #  correct numerical errors
    return probs

# ------------------------------------------------------------------------------------------------------------~
def sample_simplex(n):
    """
    Randomly (uniform) draw vector from n-simplex
    """
    v = np.random.uniform(size=n)
    v = -np.log(v)
    v /= np.sum(v)
    return v
# ------------------------------------------------------------------------------------------------------------~
# @jit(nopython=True)
# def sample_simplex(n):
#     """
#     Randomly (uniform) draw vector from n-simplex
#     use Kraemer Algorithm
#     https://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
#     """
#     M = 1000 # the numbers are up to  precision 1/M
#     x = np.random.randint(low=0, high=M+1, size=n-1)
#     x[0] = 0
#     x[-1] = M
#     x.sort()
#     y = x[1:] - x[:-1]
#     probs = y/M
#     return probs




def augment_mix_time(Pss, forced_mix_time):
    nS = Pss.shape[0]
    # Modify P so we have the desired mixing time
    evals, evecs = np.linalg.eig(Pss)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    lambda2_old = evals[1]
    lambda2_old_abs = np.abs(lambda2_old)
    spect_gap_old = 1 - lambda2_old_abs
    mix_time_old = 1 / spect_gap_old
    mix_time = forced_mix_time
    forced_lamb2_abs = (mix_time - 1.) / mix_time
    forced_spect_gap = 1 - forced_lamb2_abs
    evals[1] = lambda2_old * forced_lamb2_abs / lambda2_old_abs  # update lambda_2
    # for any e.val (besides the first) with abs larger then lamb2_abs - fix it
    for iev in range(1, nS):
        eval_abs = np.abs(evals[iev])
        if eval_abs > forced_lamb2_abs:
            evals[iev] *= forced_lamb2_abs / eval_abs

    # reconstruct P:
    PssNew = np.real(evecs @ np.diag(evals) @ np.linalg.inv(evecs))
    # correct numerical errors:
    PssNew[PssNew < 1e-15] = 0
    row_sums = PssNew.sum(axis=1)
    PssNew = PssNew / row_sums[:, np.newaxis]
    return PssNew
