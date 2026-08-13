"""Information-optimal placement of discrete states for the *interrogation*
(fixed/variable viewing-duration) accumulator -- companion to absorbing_ib.py
/ ib_optimal_placement.py, which cover the terminating (absorbing-bound, RT)
task.

THE TASK. Gold & Shadlen (2003, J Neurosci 23:632-651) used a version of the
random-dot direction task where the *experimenter*, not the subject, decides
when evidence stops: motion is shown for a duration T drawn independently of
the (unobservable, still-forming) decision, and the choice is read out from
whatever the decision variable's state happens to be at T. This module is
that task's capacity-limited analogue: no absorbing bound (the accumulator
never stops itself), and performance is scored by the *state at a random
stopping time* rather than by which of two boundaries gets hit first.

Two structural differences from the absorbing case, both handled below
------------------------------------------------------------------------
1. **Occupancy has no fundamental matrix to read it off.** There is no
   absorption, so "how much time the representation spends near x" isn't
   well-defined the way it was for a terminating chain. What matters instead
   is the *marginal distribution of the accumulator's position at whichever
   time the trial happens to end* -- i.e. occupancy here means "how much
   mass ends up near x, averaged over the actual distribution of viewing
   durations T the task uses." That's a closed-form Gaussian mixture
   (X_T | drift ~ N(drift*T, s^2*T)), integrated numerically over T's
   density -- no chain, no matrix inversion needed for this piece.
   Relevance q(x) = P(Y=upper | X_t=x) is untouched: the Girsanov LLR
   identity (absorbing_ib / ib_optimal_placement) has no time dependence,
   so relevance() from ib_optimal_placement is reused as-is.
2. **The chain has no absorbing ends -- it reflects.** A capacity-k observer
   here has k *ordinary* states, not k-1 interior states flanked by two
   absorbing labels. The two extreme nodes use a one-sided version of the
   same Kushner & Dupuis (2001) finite-difference generator
   (ib_optimal_placement.nonuniform_ctmc_rates) instead of routing outward
   flux to an absorbing state. Reading out the choice at a specific,
   possibly-random T is then an exact CTMC transient-distribution problem
   (matrix exponential of the generator), not a first-passage one.

A readout subtlety worth flagging explicitly (it cost real debugging time
building this module): accuracy must be scored by a *deterministic*
threshold decode (which side of the start position the state ends up on),
not by a soft, probability-matching readout like E[q(X_T)] (mixing choices
in proportion to q). The latter is a strictly worse decision rule -- e.g. for
the pure continuous DDM, E[q(X_T)] at v=0.6, s=1, T=1.5 is ~0.685, while the
deterministic sign(X_T) accuracy is Phi(v*sqrt(T)/s) ~= 0.769 -- and only the
deterministic version converges to that closed form as k grows. It also
matches how every other accuracy number in this notebook series is scored:
absorbing_ib/ib_optimal_placement's "accuracy" is which *boundary* got hit,
already a deterministic label, not a graded one.
"""
import numpy as np
from scipy.linalg import expm
from scipy.stats import norm

from ib_optimal_placement import (
    relevance, optimal_partition_fixed_start, uniform_partition_boundaries_fixed_start,
    objective_of_boundaries, group_centroids,
)


# ------------------------------------------------------- viewing-duration T

def viewing_duration_density(t, t0, tau, T_max):
    """Near-flat-hazard viewing-duration density used in Gold & Shadlen
    (2003): t0 plus an Exponential(mean tau), censored at T_max and
    renormalized on [t0, T_max] so it integrates to 1 there. Matches the
    paper's actual design (100 msec plus Exponential(mean 300 msec)),
    chosen so the subject can't anticipate when a trial will end (a flat
    hazard on t-t0 up to the censoring point)."""
    Z = 1.0 - np.exp(-(T_max - t0) / tau)
    dens = np.where((t >= t0) & (t <= T_max), np.exp(-(t - t0) / tau) / tau / Z, 0.0)
    return dens


def viewing_duration_grid(t0, tau, T_max, n=400):
    """Quadrature grid and (normalized-to-sum-1) weights for integrating
    against viewing_duration_density on [t0, T_max] via the midpoint rule."""
    edges = np.linspace(t0, T_max, n + 1)
    tgrid = (edges[:-1] + edges[1:]) / 2
    w = viewing_duration_density(tgrid, t0, tau, T_max)
    return tgrid, w / w.sum()


# --------------------------------------------------- 1. occupancy & relevance

def fine_reference_interrogation(k0, v0, s, t0, tau, T_max, x_range, n_tgrid=400):
    """Fine uniform reference grid over accumulated evidence x in
    [-x_range, x_range]: occupancy w(x) is the honest (unknown-sign,
    known-magnitude v0) marginal density of X_T, itself marginalized over
    the task's actual viewing-duration distribution --

        w(x) = integral over t of f_T(t) * 0.5*[N(x; v0 t, s^2 t) + N(x; -v0 t, s^2 t)] dt

    -- the interrogation-task analogue of
    ib_optimal_placement_unknown_drift.mixture_fine_reference's occupancy
    mixture, just built from a Gaussian mixture instead of a fundamental
    matrix (see module docstring, point 1). Relevance q(x) is the same exact
    closed form as the absorbing case (untouched by the extra T-averaging --
    see module docstring). x_range should comfortably cover the task's
    reachable range (several sigma*sqrt(T_max) plus drift*T_max); this is
    checked, not just assumed, by the k0-independent sanity check in the
    notebook."""
    x = np.linspace(-x_range, x_range, k0)
    tgrid, wT = viewing_duration_grid(t0, tau, T_max, n_tgrid)
    w = np.zeros_like(x)
    for t, wt in zip(tgrid, wT):
        sd = s * np.sqrt(t)
        w += wt * 0.5 * (norm.pdf(x, v0 * t, sd) + norm.pdf(x, -v0 * t, sd))
    w = w / w.sum()
    z0_pos = 0.0
    z0_index = int(np.argmin(np.abs(x - z0_pos)))
    q = relevance(x, v0, s, z0_pos)
    return dict(x=x, w=w, q=q, z0_index=z0_index, z0_pos=z0_pos, k0=k0,
                v0=v0, s=s, t0=t0, tau=tau, T_max=T_max)


# ------------------------------------------------- 2. reflecting-boundary CTMC

def reflecting_ctmc_rates(x_nodes, v_signed, s):
    """Up/down jump rates at *every* node of x_nodes (all k of them are real
    states here -- no absorbing end labels, unlike
    ib_optimal_placement.nonuniform_ctmc_rates). Interior nodes use the same
    Kushner & Dupuis (2001) central finite-difference generator; the two
    edge nodes use the matching one-sided (reflecting) version, i.e. only
    the rate pointing back into the domain is kept. Can go negative at very
    coarse, extreme k (the reflecting analogue of
    ib_optimal_placement.nonuniform_ctmc_rates's k_min_for_validity caveat)
    -- returned as-is, not silently clipped, so the caller can check."""
    k = len(x_nodes)
    c_plus, c_minus = np.zeros(k), np.zeros(k)
    for i in range(k):
        if 0 < i < k - 1:
            h_minus, h_plus = x_nodes[i] - x_nodes[i - 1], x_nodes[i + 1] - x_nodes[i]
            c_minus[i] = (s ** 2 - v_signed * h_plus) / (h_minus * (h_minus + h_plus))
            c_plus[i] = (v_signed * h_minus + s ** 2) / (h_plus * (h_minus + h_plus))
        elif i == 0:
            h_plus = x_nodes[1] - x_nodes[0]
            c_plus[i] = v_signed / h_plus + s ** 2 / h_plus ** 2
        else:  # i == k - 1
            h_minus = x_nodes[i] - x_nodes[i - 1]
            c_minus[i] = -v_signed / h_minus + s ** 2 / h_minus ** 2
    return c_plus, c_minus


def is_valid_reflecting_chain(x_nodes, v0, s):
    """True iff reflecting_ctmc_rates stays non-negative at every node under
    BOTH true drifts +-v0 (the honest chain must be valid whichever sign
    turns out to be true) -- the reflecting-task analogue of
    absorbing_ib.k_min_for_validity, needed because a placement's node
    spacing (not just its state *count*) determines validity here: the
    IB-optimal placement concentrates nodes where occupancy is high and
    tends to stay valid down to very few states, while a naively
    evenly-spaced placement can go invalid (negative rates -- not a proper
    CTMC generator, and exp(Q*T) then returns meaningless, sometimes
    negative, 'probabilities') at surprisingly large k if its start-adjacent
    spacing is uneven enough. Always check this before trusting
    interrogation_chain_stats's output."""
    for v_signed in (v0, -v0):
        c_plus, c_minus = reflecting_ctmc_rates(x_nodes, v_signed, s)
        if np.any(c_plus < 0) or np.any(c_minus < 0):
            return False
    return True


def reflecting_generator(x_nodes, v_signed, s):
    """CTMC generator matrix Q (k x k) for the reflecting non-uniform chain:
    off-diagonal entries are the rates from reflecting_ctmc_rates, diagonal
    is minus the row sum (standard generator convention). exp(Q*T) gives the
    exact transient state-distribution propagator (state_distribution_at)."""
    c_plus, c_minus = reflecting_ctmc_rates(x_nodes, v_signed, s)
    k = len(x_nodes)
    Q = np.zeros((k, k))
    for i in range(k):
        if i + 1 < k:
            Q[i, i + 1] = c_plus[i]
        if i - 1 >= 0:
            Q[i, i - 1] = c_minus[i]
        Q[i, i] = -(c_plus[i] + c_minus[i])
    return Q


def state_distribution_at(Q, z0_index, T):
    """Exact P(state=r | started at z0_index, elapsed time T), via the
    matrix exponential of the generator -- the reflecting-chain analogue of
    ib_optimal_placement.nonuniform_chain_stats's fundamental-matrix solve
    (that one solves an absorption problem; this one is a transient-
    distribution problem, so it needs expm instead of a matrix inverse)."""
    return expm(Q * T)[z0_index]


# ------------------------------------------------------------- 3. evaluation

def interrogation_chain_stats(x_nodes, v0, s, z0_index, Tvals):
    """Honest accuracy at each viewing duration in Tvals: evaluate the SAME
    fixed placement under both true drifts +-v0 (the encoder doesn't know
    the sign, but physical trials do have one), decode by a deterministic
    threshold at the start position (state > z0's position -> 'upper'; see
    module docstring for why this, not a soft q(x)-weighted readout, is the
    right accuracy notion), and combine with equal prior -- the
    interrogation-task analogue of
    ib_optimal_placement_unknown_drift.mixture_chain_stats.

    One discretization-only wrinkle handled explicitly: the exact z0 state
    itself (probability of "hasn't moved off the start node yet") is
    genuinely tied under a threshold-at-z0 rule, and is scored as a forced
    coin-flip guess (0.5 credit), not silently dropped and not (as an
    earlier, buggy version of this function did, via a floating-point grid
    offset) accidentally credited whole to one side. As k grows this state's
    probability mass -> 0 and the distinction stops mattering -- checked
    directly in the notebook against the k -> large continuum limit."""
    Qp = reflecting_generator(x_nodes, +v0, s)
    Qm = reflecting_generator(x_nodes, -v0, s)
    above = x_nodes > x_nodes[z0_index]
    below = x_nodes < x_nodes[z0_index]
    accs = np.empty(len(Tvals))
    for i, T in enumerate(Tvals):
        Pp = state_distribution_at(Qp, z0_index, T)
        Pm = state_distribution_at(Qm, z0_index, T)
        correct_plus = Pp[above].sum() + 0.5 * Pp[z0_index]
        correct_minus = Pm[below].sum() + 0.5 * Pm[z0_index]
        accs[i] = 0.5 * correct_plus + 0.5 * correct_minus
    return accs


def continuous_interrogation_accuracy(v, s, Tvals):
    """Exact continuum ceiling: P(sign(X_T) correct) = Phi(v*sqrt(T)/s) for
    X_T ~ N(v*T, s^2*T) starting at 0 (Y=upper <=> drift=+v), no discretization
    anywhere -- the interrogation-task analogue of ddm_analytic_accuracy."""
    return norm.cdf(v * np.sqrt(np.asarray(Tvals, dtype=float)) / s)


def simulate_reflecting_chain_path(x_nodes, v, s, z0_index, T_max, seed=0):
    """Single-trial path (real time, position) of the reflecting non-uniform
    CTMC, run out to T_max (the longest a trial's viewing duration could be)
    -- the reflecting-chain analogue of intuitions.simulate_nonuniform_chain_path,
    minus the absorption bookkeeping (there is none to do here)."""
    rng = np.random.default_rng(seed)
    c_plus, c_minus = reflecting_ctmc_rates(x_nodes, v, s)
    rate_tot = c_plus + c_minus
    p_up = np.divide(c_plus, rate_tot, out=np.full_like(c_plus, 0.5), where=rate_tot > 0)
    idx = z0_index
    t = 0.0
    ts, xs = [0.0], [x_nodes[idx]]
    while t < T_max:
        t += rng.exponential(1.0 / rate_tot[idx])
        idx += 1 if rng.random() < p_up[idx] else -1
        idx = int(np.clip(idx, 0, len(x_nodes) - 1))
        ts.append(min(t, T_max))
        xs.append(x_nodes[idx])
    return np.array(ts), np.array(xs)


# ---------------------------------------------------------------- pipeline

def ib_optimal_chain_interrogation(k0, k_groups, v0, s, t0, tau, T_max, x_range, Tvals, n_tgrid=400):
    """End-to-end: fine occupancy/relevance -> DP-optimal partition (fixed
    start at z0=0) -> node positions -> honest accuracy-vs-T for both the
    optimal and same-method uniform placements. Mirrors
    ib_optimal_placement_unknown_drift.ib_optimal_chain_unknown_drift_fixed_start's
    structure exactly, substituting this module's occupancy/generator/
    evaluation pieces for the absorbing-chain ones."""
    fr = fine_reference_interrogation(k0, v0, s, t0, tau, T_max, x_range, n_tgrid)
    n = len(fr["w"])
    z0_index = fr["z0_index"]

    boundaries_opt, obj_opt, qbar, z0g_opt = optimal_partition_fixed_start(
        fr["w"], fr["q"], k_groups, z0_index)
    boundaries_unif, z0g_unif = uniform_partition_boundaries_fixed_start(
        n, k_groups, z0_index)
    obj_unif = objective_of_boundaries(fr["w"], fr["q"], boundaries_unif, qbar=qbar)

    xn_opt = group_centroids(fr["x"], fr["w"], boundaries_opt)
    xn_unif = group_centroids(fr["x"], fr["w"], boundaries_unif)

    valid_opt = is_valid_reflecting_chain(xn_opt, v0, s)
    valid_unif = is_valid_reflecting_chain(xn_unif, v0, s)
    acc_opt = interrogation_chain_stats(xn_opt, v0, s, z0g_opt, Tvals) if valid_opt else None
    acc_unif = interrogation_chain_stats(xn_unif, v0, s, z0g_unif, Tvals) if valid_unif else None

    return dict(fr=fr, boundaries_opt=boundaries_opt, boundaries_unif=boundaries_unif,
                obj_opt=obj_opt, obj_unif=obj_unif, qbar=qbar,
                x_nodes_opt=xn_opt, x_nodes_unif=xn_unif,
                z0_index_opt=z0g_opt, z0_index_unif=z0g_unif,
                valid_opt=valid_opt, valid_unif=valid_unif,
                acc_opt=acc_opt, acc_unif=acc_unif, Tvals=np.asarray(Tvals, dtype=float))
