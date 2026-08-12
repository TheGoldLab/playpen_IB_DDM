"""Information-optimal placement of discrete states for the absorbing-bound
accumulator, WITHOUT letting the encoder see the trial's answer.

ib_optimal_placement.py's relevance()/fine_reference()/ib_optimal_chain()
build the encoder using the trial's actual, signed drift v -- an oracle
upper bound, not a model of a real decision-maker (see that module's
docstring). This module fixes that: the encoder is built from a *prior*
over which hypothesis is true, with the trial's actual sign hidden, using
only the trial's magnitude (SNR) as known task structure. It reuses
ib_optimal_placement's DP (optimal_partition) and non-uniform chain
construction (nonuniform_ctmc_rates, build_birth_death,
nonuniform_chain_stats, simulate_nonuniform_chain) unchanged -- neither
depends on how the occupancy/relevance inputs were computed.

The two changed ingredients
----------------------------
1. **Relevance**, now prior-aware. The LLR identity from the known-drift
   module (LLR_t = (2v0/s^2)(X_t - z0), path/time-independent by Girsanov)
   still holds -- it does not require knowing the trial's sign, only the
   magnitude v0. Combined with a prior pi = P(Y=upper), Bayes' rule gives

       q(x) = P(Y=upper | X_t=x) = sigmoid(logit(pi) + 2*v0*(x-z0)/s^2)

   still closed form, still no fundamental matrix needed. At pi=0.5 this
   reduces to the known-drift module's formula with v0 in place of v --
   the difference is entirely in the occupancy below, not here.
2. **Occupancy, now a prior-weighted mixture.** The encoder's boundaries
   must be fixed before the trial's drift sign is known, so the occupancy
   it's designed against has to be the *marginal* over both possible
   generative processes, weighted by the prior:

       p(x) = pi * p(x | drift=+v0) + (1-pi) * p(x | drift=-v0)

   each term computed exactly as in the known-drift module (fundamental
   matrix of a fine reference chain), then mixed. At pi=0.5 this mixture is
   provably symmetric about z0 (flipping drift sign and reflecting x about
   z0 leaves the random walk's law unchanged), which is a direct,
   checkable validation: the known-drift module's placement was visibly
   biased toward the correct side (drift-baised midpoint); this one should
   not be, and isn't (checked in the notebook against boundary symmetry,
   not just visually).

Evaluating a placement's performance now also requires mixing: a fixed
set of boundaries is evaluated once under +v0 physics and once under -v0
physics, and accuracy/mean-time are the prior-weighted combination
(mixture_chain_stats, mixture_simulate).

A range of unknown SNR magnitudes (not just an unknown sign)
--------------------------------------------------------------
snr_branches / mixture_fine_reference_multi / mixture_chain_stats_multi
below generalize everything above from "one known magnitude v0, unknown
sign" to "several possible magnitudes, each with its own prior probability,
unknown sign" -- e.g. a task where trial difficulty (coherence) varies from
trial to trial and the encoder has to be built without knowing in advance
which difficulty, or which side, a given trial will turn out to be.

The one thing worth flagging is a worry that turned out not to matter: for
a *single* known magnitude, the current position X_t is an exact sufficient
statistic for Y (the relevance formula above depends only on x). Once the
magnitude itself is uncertain, that stops being exactly true moment-to-
moment -- telling the difference between "far along on an easy trial" and
"further along than usual on a hard trial" needs elapsed time too, not
just position (Girsanov again: the likelihood ratio between two different
*magnitudes* depends on both X_t and t). So it looks, at first, like this
extension needs the representation to track (position, time) jointly,
which would break the whole "1-D position axis, exact interval-partition
DP" structure the rest of this module relies on.

It doesn't, because of what "occupancy" already means here. An occupancy
value at position x is a *sum over every time step* the fine reference
chain could be at x before being absorbed -- it was always time-marginalized,
even in the single-magnitude case, because the representation in this
whole notebook series only ever tracks position, never elapsed time. Mixing
occupancy across magnitude branches (not just across the two signs) inherits
that same time-marginalization for free, so
q(x) = P(Y=upper | representation is at x, at whatever time that happens)
is still exactly computable from occupancy alone -- no time-tracking
needed, no approximation beyond the fine-grid discretization already used
everywhere else. This is checked directly in the notebook: with a single
magnitude, this occupancy-ratio formula and the closed-form sigmoid above
must agree, and they do (to fine-grid precision).
"""
import numpy as np
from absorbing_ib import build_absorbing_chain, walk_params_from_ddm
from ib_optimal_placement import (
    optimal_partition, uniform_partition_boundaries, objective_of_boundaries,
    group_centroids, find_start_group, nonuniform_chain_stats, simulate_nonuniform_chain,
    optimal_partition_fixed_start, uniform_partition_boundaries_fixed_start,
)


def prior_relevance(x, v0, s, z0_pos, pi):
    """q(x) = P(Y=upper | X_t=x) under a prior pi = P(Y=upper), known SNR
    magnitude v0, unknown sign. See module docstring for the derivation."""
    logit_pi = np.log(pi / (1 - pi))
    return 1.0 / (1.0 + np.exp(-(logit_pi + 2 * v0 * (x - z0_pos) / s ** 2)))


def _visits_for_signed_v(k0, v_signed, s, a, z_frac):
    """Expected visits to each fine interior state under a reference chain
    that actually walks with the given signed drift -- one branch of the
    mixture."""
    wp = walk_params_from_ddm(v_signed, s, a, k0)
    Q, R, transient = build_absorbing_chain(k0, wp["p"])
    n = Q.shape[0]
    N = np.linalg.inv(np.eye(n) - Q)
    z0_index = int(np.clip(round(z_frac * k0), 1, k0 - 1)) - 1
    visits = N[z0_index, :]
    x = transient * (a / k0)
    return x, visits, z0_index


def mixture_fine_reference(k0, v0, s, a, z_frac, pi=0.5):
    """Prior-weighted occupancy mixture and prior-aware relevance -- the
    unknown-sign, known-magnitude analogue of
    ib_optimal_placement.fine_reference(). At pi=0.5, w is symmetric about
    z0 and q is antisymmetric about 0.5 there (checked in the notebook)."""
    x_plus, visits_plus, z0_index = _visits_for_signed_v(k0, +v0, s, a, z_frac)
    x_minus, visits_minus, _ = _visits_for_signed_v(k0, -v0, s, a, z_frac)
    x = x_plus  # same fine grid for both branches
    w = pi * visits_plus + (1 - pi) * visits_minus
    w = w / w.sum()
    z0_pos = z_frac * a
    q = prior_relevance(x, v0, s, z0_pos, pi)
    return dict(x=x, w=w, q=q, z0_index=z0_index, z0_pos=z0_pos, k0=k0, a=a,
                pi=pi, v0=v0, s=s)


def mixture_chain_stats(x_nodes, v0, s, z0_group, pi):
    """Accuracy and mean decision time for a fixed placement (x_nodes),
    evaluated honestly: run the SAME placement under both possible true
    drifts and combine with the prior, since the encoder can't tell which
    is active. 'Correct' = upper when Y=upper (drift=+v0), lower when
    Y=lower (drift=-v0)."""
    stats_plus = nonuniform_chain_stats(x_nodes, +v0, s, z0_group)
    stats_minus = nonuniform_chain_stats(x_nodes, -v0, s, z0_group)
    accuracy = pi * stats_plus["p_upper"] + (1 - pi) * stats_minus["p_lower"]
    expected_time = pi * stats_plus["expected_time"] + (1 - pi) * stats_minus["expected_time"]
    return dict(accuracy=accuracy, expected_time=expected_time,
                stats_plus=stats_plus, stats_minus=stats_minus)


def mixture_simulate(x_nodes, v0, s, z0_group, pi, n_trials, n_max, seed=0):
    """Monte Carlo cross-check for mixture_chain_stats: split n_trials by
    the prior, simulate each half under its true physics, and pool. Returns
    (rt, correct) with correct in {0,1} (not "upper"/"lower", since the
    correct side differs by which half a trial came from)."""
    rng = np.random.default_rng(seed)
    n_plus = rng.binomial(n_trials, pi)
    n_minus = n_trials - n_plus
    rt_plus, ch_plus = simulate_nonuniform_chain(x_nodes, +v0, s, z0_group, n_plus, n_max, seed=seed)
    rt_minus, ch_minus = simulate_nonuniform_chain(x_nodes, -v0, s, z0_group, n_minus, n_max, seed=seed + 1)
    correct_plus = (ch_plus == 1).astype(float)
    correct_plus[np.isnan(ch_plus)] = np.nan
    correct_minus = (ch_minus == 0).astype(float)
    correct_minus[np.isnan(ch_minus)] = np.nan
    rt = np.concatenate([rt_plus, rt_minus])
    correct = np.concatenate([correct_plus, correct_minus])
    return rt, correct


def ib_optimal_chain_unknown_drift(k0, k_groups, v0, s, a, z_frac, pi=0.5):
    """End-to-end: mixture fine reference -> DP-optimal partition -> node
    positions -> mixture-evaluated stats, plus the same-method uniform
    baseline. Mirrors ib_optimal_placement.ib_optimal_chain but honest:
    neither the DP nor the evaluation ever sees the trial's true sign."""
    fr = mixture_fine_reference(k0, v0, s, a, z_frac, pi)
    n = len(fr["w"])
    boundaries_opt, obj_opt, qbar = optimal_partition(fr["w"], fr["q"], k_groups)
    boundaries_unif = uniform_partition_boundaries(n, k_groups)
    obj_unif = objective_of_boundaries(fr["w"], fr["q"], boundaries_unif)

    cents_opt = group_centroids(fr["x"], fr["w"], boundaries_opt)
    cents_unif = group_centroids(fr["x"], fr["w"], boundaries_unif)
    xn_opt = np.array([0.0] + list(cents_opt) + [a])
    xn_unif = np.array([0.0] + list(cents_unif) + [a])

    z0g_opt = find_start_group(boundaries_opt, fr["z0_index"])
    z0g_unif = find_start_group(boundaries_unif, fr["z0_index"])

    stats_opt = mixture_chain_stats(xn_opt, v0, s, z0g_opt, pi)
    stats_unif = mixture_chain_stats(xn_unif, v0, s, z0g_unif, pi)

    return dict(fr=fr, boundaries_opt=boundaries_opt, boundaries_unif=boundaries_unif,
                obj_opt=obj_opt, obj_unif=obj_unif, qbar=qbar,
                x_nodes_opt=xn_opt, x_nodes_unif=xn_unif,
                z0_group_opt=z0g_opt, z0_group_unif=z0g_unif,
                stats_opt=stats_opt, stats_unif=stats_unif)


def ib_optimal_chain_unknown_drift_fixed_start(k0, k_groups, v0, s, a, z_frac, pi=0.5):
    """Same honest (prior-only) pipeline as ib_optimal_chain_unknown_drift,
    but using optimal_partition_fixed_start /
    uniform_partition_boundaries_fixed_start so both the optimal and
    uniform chains start from the trial's *exact* true starting position
    z0, rather than from whichever group happens to contain it (see
    ib_optimal_placement.find_start_group's docstring for why that
    approximation is a real source of noise -- it gets worse the more the
    optimal and uniform partitions differ in shape near z0, which is
    exactly when a prior is asymmetric). Requires k_groups >= 3 in the
    typical case where z0 has probability mass on both sides of it."""
    fr = mixture_fine_reference(k0, v0, s, a, z_frac, pi)
    n = len(fr["w"])
    z0_index = fr["z0_index"]

    boundaries_opt, obj_opt, qbar, z0g_opt = optimal_partition_fixed_start(
        fr["w"], fr["q"], k_groups, z0_index)
    boundaries_unif, z0g_unif = uniform_partition_boundaries_fixed_start(
        n, k_groups, z0_index)
    obj_unif = objective_of_boundaries(fr["w"], fr["q"], boundaries_unif, qbar=qbar)

    cents_opt = group_centroids(fr["x"], fr["w"], boundaries_opt)
    cents_unif = group_centroids(fr["x"], fr["w"], boundaries_unif)
    xn_opt = np.array([0.0] + list(cents_opt) + [a])
    xn_unif = np.array([0.0] + list(cents_unif) + [a])

    stats_opt = mixture_chain_stats(xn_opt, v0, s, z0g_opt, pi)
    stats_unif = mixture_chain_stats(xn_unif, v0, s, z0g_unif, pi)

    return dict(fr=fr, boundaries_opt=boundaries_opt, boundaries_unif=boundaries_unif,
                obj_opt=obj_opt, obj_unif=obj_unif, qbar=qbar,
                x_nodes_opt=xn_opt, x_nodes_unif=xn_unif,
                z0_group_opt=z0g_opt, z0_group_unif=z0g_unif,
                stats_opt=stats_opt, stats_unif=stats_unif)


# ------------------------------------------------- a range of SNR magnitudes


def snr_branches(v_levels, v_priors, pi):
    """Build the full set of (signed drift, prior weight) "branches" for a
    task where the SNR magnitude (drift size, i.e. how strong the evidence
    is) is drawn from v_levels with probabilities v_priors (must sum to 1),
    and independently, the sign (which side is correct, i.e. Y) is drawn
    with P(Y=upper)=pi. Returns a list of (v_signed, weight) pairs, one per
    (magnitude, sign) combination -- 2*len(v_levels) of them -- each weight
    being that combination's joint probability. A single-magnitude task is
    just the special case v_levels=[v0], v_priors=[1.0]."""
    v_levels = np.asarray(v_levels, float)
    v_priors = np.asarray(v_priors, float)
    if not np.isclose(v_priors.sum(), 1.0):
        raise ValueError(f"v_priors must sum to 1, got {v_priors.sum()}")
    branches = [(float(v), float(p) * pi) for v, p in zip(v_levels, v_priors)]
    branches += [(-float(v), float(p) * (1 - pi)) for v, p in zip(v_levels, v_priors)]
    return branches


def mixture_fine_reference_multi(k0, branches, s, a, z_frac):
    """Generalizes mixture_fine_reference from a single magnitude v0 to an
    arbitrary list of (signed drift, prior weight) branches (see
    snr_branches). Occupancy mixes every branch's expected visit counts by
    its prior weight, same idea as the two-branch case with more terms.
    Relevance is the ratio of "expected visits to x across branches where
    Y=upper" to "expected visits to x across every branch" -- exact, not an
    approximation (see the module docstring for why marginalizing over
    magnitude this way doesn't need elapsed time to be tracked)."""
    x_ref, z0_index = None, None
    per_branch_visits = []
    for v_signed, _ in branches:
        x, visits, z0i = _visits_for_signed_v(k0, v_signed, s, a, z_frac)
        if x_ref is None:
            x_ref, z0_index = x, z0i
        per_branch_visits.append(visits)
    per_branch_visits = np.array(per_branch_visits)  # (n_branches, n_fine_states)
    weights = np.array([weight for _, weight in branches])

    total = weights @ per_branch_visits  # occupancy, before normalizing to sum to 1
    upper_mask = np.array([v_signed > 0 for v_signed, _ in branches])
    numer = weights[upper_mask] @ per_branch_visits[upper_mask]  # "and Y=upper" version

    w = total / total.sum()
    q = numer / total
    z0_pos = z_frac * a
    return dict(x=x_ref, w=w, q=q, z0_index=z0_index, z0_pos=z0_pos, k0=k0, a=a,
                branches=branches)


def mixture_chain_stats_multi(x_nodes, branches, s, z0_group):
    """Generalizes mixture_chain_stats from +-v0 to an arbitrary branch
    list: evaluate the SAME fixed placement under every branch's physics
    and combine with that branch's prior weight, exactly as before but
    summed over more than two terms."""
    accuracy, expected_time = 0.0, 0.0
    for v_signed, weight in branches:
        stats = nonuniform_chain_stats(x_nodes, v_signed, s, z0_group)
        correct = stats["p_upper"] if v_signed > 0 else stats["p_lower"]
        accuracy += weight * correct
        expected_time += weight * stats["expected_time"]
    return dict(accuracy=accuracy, expected_time=expected_time)


def mixture_simulate_multi(x_nodes, branches, s, z0_group, n_trials, n_max, seed=0):
    """Monte Carlo cross-check for mixture_chain_stats_multi: split
    n_trials across branches in proportion to their prior weight, simulate
    each branch under its own true physics, and pool -- the multi-branch
    analogue of mixture_simulate."""
    rng = np.random.default_rng(seed)
    weights = np.array([weight for _, weight in branches])
    counts = rng.multinomial(n_trials, weights / weights.sum())
    rt_all, correct_all = [], []
    for (v_signed, _), n_i, seed_i in zip(branches, counts, range(len(branches))):
        if n_i == 0:
            continue
        rt, ch = simulate_nonuniform_chain(x_nodes, v_signed, s, z0_group, int(n_i), n_max,
                                            seed=seed + seed_i)
        correct = (ch == 1).astype(float) if v_signed > 0 else (ch == 0).astype(float)
        correct[np.isnan(ch)] = np.nan
        rt_all.append(rt)
        correct_all.append(correct)
    return np.concatenate(rt_all), np.concatenate(correct_all)


def ib_optimal_chain_snr_range_fixed_start(k0, k_groups, branches, s, a, z_frac):
    """Fixed-start pipeline (see ib_optimal_chain_unknown_drift_fixed_start)
    for the multi-magnitude case: builds the honest and uniform placements
    from mixture_fine_reference_multi, evaluates both with
    mixture_chain_stats_multi, and both start from the exact true z0."""
    fr = mixture_fine_reference_multi(k0, branches, s, a, z_frac)
    n = len(fr["w"])
    z0_index = fr["z0_index"]

    boundaries_opt, obj_opt, qbar, z0g_opt = optimal_partition_fixed_start(
        fr["w"], fr["q"], k_groups, z0_index)
    boundaries_unif, z0g_unif = uniform_partition_boundaries_fixed_start(
        n, k_groups, z0_index)
    obj_unif = objective_of_boundaries(fr["w"], fr["q"], boundaries_unif, qbar=qbar)

    cents_opt = group_centroids(fr["x"], fr["w"], boundaries_opt)
    cents_unif = group_centroids(fr["x"], fr["w"], boundaries_unif)
    xn_opt = np.array([0.0] + list(cents_opt) + [a])
    xn_unif = np.array([0.0] + list(cents_unif) + [a])

    stats_opt = mixture_chain_stats_multi(xn_opt, branches, s, z0g_opt)
    stats_unif = mixture_chain_stats_multi(xn_unif, branches, s, z0g_unif)

    return dict(fr=fr, boundaries_opt=boundaries_opt, boundaries_unif=boundaries_unif,
                obj_opt=obj_opt, obj_unif=obj_unif, qbar=qbar,
                x_nodes_opt=xn_opt, x_nodes_unif=xn_unif,
                z0_group_opt=z0g_opt, z0_group_unif=z0g_unif,
                stats_opt=stats_opt, stats_unif=stats_unif)
