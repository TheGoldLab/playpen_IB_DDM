"""Exact machinery for the *absorbing*-bound analogue of ib_core.py.

ib_core.py solves the bead task: a *volatile* source with a stabilizing
(non-absorbing) bound, where the optimal capacity-limited recursive machine
is a bounded ladder that holds a category forever (Glaze's leaky
accumulator, discretized).

This module covers the complementary case that the standard DDM lives in:
a *stationary* source (no hazard, no leak) with an *absorbing* bound, where
crossing the bound is the decision itself. The bridge is a classical one
(Feller; Ratcliff 1978's original random-walk formulation): a discrete-time,
discrete-state random walk with absorbing barriers converges to the
drift-diffusion model as the number of interior states k -> infinity, with
step size and step duration shrinking together. Here k is the natural stand-in
for representational *capacity* (an observer who can only hold k
distinguishable states of accumulated evidence between the bounds), so this
module lets you ask exactly what a capacity cut looks like in DDM terms: hold
the physical decision bound `a`, drift `v`, and noise `s` fixed, shrink `k`,
and see what happens to choice accuracy and the RT distribution.

Everything that admits a closed form is computed exactly (fundamental-matrix
absorbing-Markov-chain algebra, or the known ODE solutions for the
continuous DDM's hitting-time moments); Monte Carlo simulation is provided
only to validate the exact results, mirroring the validation style of
ib_core.py / IB_memo_v6.

Conventions
-----------
* The decision variable lives on states 0..k. States 0 and k are absorbing
  ("choice B" / "choice A"). States 1..k-1 are transient.
* Positive drift v favors the upper bound (state k = "choice A" = correct
  when v > 0).
* z_frac in (0, 1) is the starting point as a fraction of the bound
  separation; z_frac = 0.5 is an unbiased start.
"""
import numpy as np

# --------------------------------------------------------------- the bridge


def walk_params_from_ddm(v, s, a, k):
    """Map continuous DDM parameters (drift v, noise s, bound separation a)
    and a capacity k (number of interior rungs) onto the matched discrete
    random walk: step size eps, step duration tau, and up-step probability p.

    Derivation (Feller / Ratcliff 1978): a symmetric random walk with step
    +-eps every tau seconds, up-probability p = 0.5*(1 + (v/s)*sqrt(tau)),
    converges as tau -> 0 to a Wiener process with drift v and diffusion
    coefficient s^2, provided eps = s*sqrt(tau). Fixing eps = a/k (so k steps
    span the bound) and solving for tau gives the two lines below; larger k
    means smaller, faster, less informative steps, i.e. finer capacity.
    """
    eps = a / k
    tau = (eps / s) ** 2
    p = 0.5 * (1.0 + (v / s) * np.sqrt(tau))
    return dict(eps=eps, tau=tau, p=p, k=k)


def k_min_for_validity(v, s, a):
    """Minimum k for which p in (0, 1), i.e. the random-walk approximation
    is still a valid probability. Below this, a single step would carry more
    evidence than the whole bound separation allows: p = 0.5*(1 + v*a/(k*s^2))
    requires k > |v|*a/s^2. This is the absorbing-bound analogue of the
    memo's beta_c: a capacity floor below which the discretization breaks."""
    return abs(v) * a / s ** 2


# ------------------------------------------------------- the discrete chain


def build_absorbing_chain(k, p):
    """Transient-transient block Q and transient-absorbing block R for the
    k+1-state chain (0..k, absorbing at the ends), symmetric +-1 steps with
    up-probability p. Returns Q (k-1, k-1), R (k-1, 2) with columns
    [lower, upper], and the transient state labels (1..k-1)."""
    n = k - 1
    if n < 1:
        raise ValueError("need k >= 2 for at least one transient state")
    Q = np.zeros((n, n))
    R = np.zeros((n, 2))
    for i, state in enumerate(range(1, k)):
        if state - 1 == 0:
            R[i, 0] += 1 - p
        else:
            Q[i, i - 1] += 1 - p
        if state + 1 == k:
            R[i, 1] += p
        else:
            Q[i, i + 1] += p
    return Q, R, np.arange(1, k)


def exact_absorption_stats(Q, R, z0_index):
    """Fundamental-matrix solution: expected steps to absorption and
    absorption probabilities [lower, upper], starting from the transient
    state at position z0_index (0-based index into the transient block)."""
    n = Q.shape[0]
    N = np.linalg.inv(np.eye(n) - Q)
    expected_steps = N @ np.ones(n)
    absorb_probs = N @ R
    return dict(expected_steps=expected_steps[z0_index],
                p_lower=absorb_probs[z0_index, 0],
                p_upper=absorb_probs[z0_index, 1],
                N=N)


def exact_rt_distribution(Q, R, z0_index, n_max):
    """Exact P(absorbed at step n, into lower/upper) for n = 1..n_max, by
    propagating the transient-state distribution one step at a time
    (v_n = v_{n-1} @ Q) and reading off the absorption mass at each step
    (v_{n-1} @ R). Returns array of shape (n_max, 2); the leftover
    probability mass still transient after n_max steps is also returned so
    truncation error is visible rather than silent."""
    n = Q.shape[0]
    v = np.zeros(n)
    v[z0_index] = 1.0
    hits = np.zeros((n_max, 2))
    for step in range(n_max):
        hits[step] = v @ R
        v = v @ Q
    return hits, v.sum()


def simulate_discrete_chain(k, p, z0_index, n_trials, n_max, seed=0):
    """Monte Carlo cross-check for exact_absorption_stats / exact_rt_distribution."""
    rng = np.random.default_rng(seed)
    state = np.full(n_trials, z0_index + 1)  # transient labels are 1..k-1
    steps = np.full(n_trials, np.nan)
    choice = np.full(n_trials, np.nan)  # 0 = lower, 1 = upper
    active = np.ones(n_trials, dtype=bool)
    for t in range(1, n_max + 1):
        if not active.any():
            break
        moves = rng.random(active.sum()) < p
        state[active] += np.where(moves, 1, -1)
        just_hit_upper = active.copy()
        just_hit_upper[active] &= (state[active] == k)
        just_hit_lower = active.copy()
        just_hit_lower[active] &= (state[active] == 0)
        steps[just_hit_upper | just_hit_lower] = t
        choice[just_hit_upper] = 1
        choice[just_hit_lower] = 0
        active &= ~(just_hit_upper | just_hit_lower)
    return steps, choice


# ------------------------------------------------------- the continuous DDM


def simulate_ddm(v, s, a, z_frac, dt, n_trials, t_max, seed=0):
    """Euler-Maruyama simulation of the continuous DDM (dX = v dt + s dW,
    absorbing at 0 and a) as the k -> infinity ground truth. Returns RT
    (decision time only, no non-decision time) and choice (0 = lower,
    1 = upper); RT is NaN for trials that fail to absorb by t_max."""
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(t_max / dt))
    x = np.full(n_trials, z_frac * a)
    rt = np.full(n_trials, np.nan)
    choice = np.full(n_trials, np.nan)
    active = np.ones(n_trials, dtype=bool)
    sqrt_dt = np.sqrt(dt)
    for step in range(1, n_steps + 1):
        if not active.any():
            break
        idx = np.where(active)[0]
        x[idx] += v * dt + s * sqrt_dt * rng.standard_normal(idx.size)
        hit_upper = active.copy(); hit_upper[idx] &= (x[idx] >= a)
        hit_lower = active.copy(); hit_lower[idx] &= (x[idx] <= 0)
        done = hit_upper | hit_lower
        rt[done] = step * dt
        choice[hit_upper] = 1
        choice[hit_lower] = 0
        active &= ~done
    return rt, choice


def ddm_analytic_accuracy(v, s, a, z_frac=0.5):
    """P(hit upper before lower) for BM with drift v, noise s, absorbing at
    0 and a, starting at z = z_frac*a. Closed form from the scale-function /
    reflection-principle solution of the exit-probability ODE; v = 0 handled
    by the driftless limit (linear interpolation, i.e. the martingale X_t)."""
    z = z_frac * a
    if abs(v) < 1e-12:
        return z / a
    x = 2 * v / s ** 2
    return (1 - np.exp(-x * z)) / (1 - np.exp(-x * a))


def ddm_analytic_mean_rt(v, s, a, z_frac=0.5):
    """E[hitting time] for the same process, from solving
    (s^2/2) m''(z) + v m'(z) = -1, m(0) = m(a) = 0. v = 0 handled by the
    driftless limit m(z) = z*(a-z)/s^2."""
    z = z_frac * a
    if abs(v) < 1e-12:
        return z * (a - z) / s ** 2
    x = 2 * v / s ** 2
    C2 = (a / v) / (np.exp(-x * a) - 1)
    return -z / v + C2 * (np.exp(-x * z) - 1)


# ---------------------------------------------------------- capacity sweep


def capacity_frontier(k_list, v, s, a, z_frac=0.5):
    """Sweep interior-state count k (capacity), returning exact accuracy and
    mean RT (real time, via tau) at each valid k. k values below
    k_min_for_validity are skipped. This is the absorbing-bound analogue of
    the memo's IB frontier / Figure 5 cost-of-recursion curve: capacity on
    one axis, performance (here, accuracy and speed jointly) on the other."""
    kmin = k_min_for_validity(v, s, a)
    out = dict(k=[], bits=[], accuracy=[], mean_rt=[], p=[])
    for k in k_list:
        if k <= kmin or k < 2:
            continue
        wp = walk_params_from_ddm(v, s, a, k)
        z0 = int(np.clip(round(z_frac * k), 1, k - 1)) - 1  # 0-based transient index
        Q, R, _ = build_absorbing_chain(k, wp['p'])
        stats = exact_absorption_stats(Q, R, z0)
        out['k'].append(k)
        out['bits'].append(np.log2(k))
        out['accuracy'].append(stats['p_upper'] if v >= 0 else stats['p_lower'])
        out['mean_rt'].append(stats['expected_steps'] * wp['tau'])
        out['p'].append(wp['p'])
    return {key: np.array(val) for key, val in out.items()}
