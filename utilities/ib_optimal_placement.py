"""Information-optimal placement of discrete states for the *absorbing*-bound
accumulator (companion to absorbing_ib.py, which used a fixed uniform ladder).

KNOWN-DRIFT (ORACLE) VARIANT. `relevance()`, `fine_reference()`, and
`ib_optimal_chain()` below build the encoder using the trial's actual,
signed drift v -- i.e. they assume the direction and strength of the
evidence are known when the placement is chosen, which is not a realistic
capacity-limited observer (if you knew the answer you wouldn't need to
accumulate evidence for it). That makes this the oracle upper bound on what
a capacity-k absorbing encoder could achieve, not a candidate behavioral
model. See ib_optimal_placement_unknown_drift.py for the realistic version,
where the encoder is built from a prior over which hypothesis is true
(optionally asymmetric, optionally over a range of evidence strengths)
instead of from the trial's true drift. That module reuses everything below
except relevance()/fine_reference() -- the DP (optimal_partition) and the
non-uniform chain construction (nonuniform_ctmc_rates, build_birth_death,
nonuniform_chain_stats, simulate_nonuniform_chain) do not depend on how the
occupancy/relevance inputs were computed and carry over unchanged.

The question: notebook 01 built a k-state absorbing chain with uniformly
spaced rungs and validated it against the continuous DDM. But per IB_memo_v6
(sec 2.1-2.2), a capacity-limited representation is not uniformly spaced --
resolution concentrates where (occupancy) x (usefulness) is highest. This
module derives what that placement is for a terminating (absorbing) SPRT/DDM
accumulator, where "usefulness" cannot be borrowed unchanged from the memo's
non-absorbing bead-task construction (there Y is a moving target, the state
*now*; here Y is a single, fixed-per-trial hypothesis).

The derivation in three exact pieces
-------------------------------------
1. **Relevance q(x) = P(Y=upper | X_t=x).** Because the DDM's position is an
   affine transform of the accumulated log-likelihood ratio (Girsanov: for a
   Brownian path with drift +-v against noise s, LLR_t = (2v/s^2) X_t, no
   time-dependent term, since the drift-magnitude terms cancel between the
   two symmetric hypotheses), this has an exact closed form:
   q(x) = sigmoid(2v(x - z0)/s^2), with z0 the trial start. No fundamental
   matrix needed for this piece.
2. **Occupancy p(x).** How much time the representation spends at each point
   of a fine reference grid before absorption -- exactly the expected number
   of visits to each state, which is a column read off the fundamental
   matrix N = (I-Q)^-1 of a fine (large-k0) *uniform* reference chain (reuses
   absorbing_ib.build_absorbing_chain). Normalized, this is P(X=x | still
   deciding).
3. **Optimal grouping.** Both q(x) and the occupancy-weighted marginal are
   monotone in x (q(x) is monotone by construction; the walk is 1-D), so the
   IB-optimal quantizer -- maximize I(Y;R) for a hard quantizer R into
   exactly k-1 contiguous groups of the fine grid -- is provably an interval
   partition, and can be found *exactly* (global optimum, not a local one
   the way Blahut-Arimoto's soft/stochastic search is) via dynamic
   programming over interval boundaries. Verified against brute-force
   enumeration on a small case below.

Turning the resulting group boundaries into an actual chain
-------------------------------------------------------------
The groups define non-uniformly spaced node positions (occupancy-weighted
centroids). To get a valid Markov chain on this non-uniform grid, transition
rates are built with the standard non-uniform-grid finite-difference
discretization of the diffusion generator L f = v f' + (s^2/2) f'' (the
"Markov chain approximation method," Kushner & Dupuis 2001) -- a classical,
citable numerical scheme, not an ad hoc symmetrization. It reduces exactly
to absorbing_ib.walk_params_from_ddm's p and tau in the uniform-spacing
special case (checked in the notebook), which is the key internal
consistency check on this generalization.
"""
import numpy as np

# ------------------------------------------------- 1-2. relevance & occupancy


def relevance(x, v, s, z0_pos):
    """q(x) = P(Y=upper | X_t=x), exact closed form (see module docstring)."""
    return 1.0 / (1.0 + np.exp(-2 * v * (x - z0_pos) / s ** 2))


def fine_reference(k0, v, s, a, z_frac):
    """Fine uniform reference chain: exact occupancy (expected visits before
    absorption, from the fundamental matrix) and relevance q(x) at each fine
    interior state. This is the (p(x), p(y|x)) pair the optimal_partition
    DP below operates on -- the absorbing-chain analogue of ib_core.py's
    (px, pyx)."""
    from absorbing_ib import build_absorbing_chain, walk_params_from_ddm

    wp = walk_params_from_ddm(v, s, a, k0)
    Q, R, transient = build_absorbing_chain(k0, wp["p"])
    n = Q.shape[0]
    N = np.linalg.inv(np.eye(n) - Q)
    z0_pos = z_frac * a
    z0_index = int(np.clip(round(z_frac * k0), 1, k0 - 1)) - 1
    w = N[z0_index, :]
    w = w / w.sum()  # normalize expected visits -> P(X=x | still deciding)
    x = transient * (a / k0)
    q = relevance(x, v, s, z0_pos)
    return dict(x=x, w=w, q=q, z0_index=z0_index, z0_pos=z0_pos, wp=wp, k0=k0, a=a)


# ------------------------------------------------------ 3. optimal partition


def optimal_partition(w, q, k_groups, qbar=None):
    """Exact global optimum: partition the ordered fine grid (weights w,
    relevance q) into exactly k_groups contiguous groups maximizing
    I(Y;R) = sum_r p(r) KL(p(y|r) || p(y)), in bits ("I(Y;R)" is mutual
    information -- how many bits, on average, knowing which group R a point
    fell in tells you about the true hypothesis Y; KL, the
    Kullback-Leibler divergence, is the standard measure of how different
    two probability distributions are). Solved via dynamic programming (DP)
    over interval boundaries: an algorithm that builds the answer by
    combining exact answers to smaller subproblems (here: "best split of
    the first i points into kk groups"), which is why the result is the
    true global optimum and not just a good guess. The inner loop over
    candidate boundaries is vectorized with numpy rather than a Python
    for-loop, since this DP gets called many times across a capacity sweep
    and the naive version is a real bottleneck (a pure-Python version of
    this same recursion took ~30x longer on a 300-point grid, k=2..30).

    qbar: optionally supply the marginal P(Y=upper) to score groups against,
    instead of computing it from this array's own w, q. Needed when w, q
    are only a *piece* of a larger reference distribution (see
    optimal_partition_fixed_start below), so that I(Y;R) contributions
    stay consistent with the true, whole-distribution marginal rather than
    a local one computed from just this piece.

    Returns (boundaries, objective_bits, qbar) where boundaries is a list of
    k_groups+1 fine-grid indices (0 = start, n = end) delimiting the groups.
    """
    n = len(w)
    if qbar is None:
        prefix_w = np.cumsum(w)
        prefix_wq = np.cumsum(w * q)
        qbar = prefix_wq[-1] / prefix_w[-1]
    dp, bp = _dp_full_table(w, q, k_groups, qbar)
    boundaries = _backtrack(bp, k_groups, n)
    return boundaries, dp[k_groups, n], qbar


def _dp_full_table(w, q, k_max, qbar):
    """Shared DP engine behind optimal_partition and
    optimal_partition_fixed_start. Returns the full table dp[kk, i] = best
    I(Y;R) (bits) from splitting the first i fine-grid points into exactly
    kk contiguous groups, for kk = 0..k_max and i = 0..len(w), plus the
    matching backpointer table bp (bp[kk, i] = the boundary j that achieves
    that best value, i.e. the split is "...j groups up to j, then one more
    group covering [j, i)"). -inf marks a combination that isn't reachable
    (e.g. 0 groups can't cover a nonempty range)."""
    n = len(w)
    prefix_w = np.concatenate([[0.0], np.cumsum(w)])
    prefix_wq = np.concatenate([[0.0], np.cumsum(w * q)])
    qb = np.clip(qbar, 1e-15, 1 - 1e-15)

    NEG = -np.inf
    dp = np.full((k_max + 1, n + 1), NEG)
    bp = np.full((k_max + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for kk in range(1, k_max + 1):
        prev = dp[kk - 1]  # best value with kk-1 groups, ending at each j
        for i in range(kk, n + 1):
            j = np.arange(kk - 1, i)  # every candidate previous boundary
            pG = prefix_w[i] - prefix_w[j]  # mass of the group [j, i)
            with np.errstate(invalid="ignore", divide="ignore"):
                qG = (prefix_wq[i] - prefix_wq[j]) / pG  # that group's P(Y=upper|R)
            qG = np.clip(qG, 1e-15, 1 - 1e-15)
            kl = qG * np.log2(qG / qb) + (1 - qG) * np.log2((1 - qG) / (1 - qb))
            vals = prev[j] + pG * kl
            vals[pG <= 0] = NEG
            vals[prev[j] == NEG] = NEG
            best = np.argmax(vals)
            dp[kk, i] = vals[best]
            bp[kk, i] = j[best]
    return dp, bp


def _backtrack(bp, k_groups, n):
    """Read a specific (k_groups, full-length) solution's boundaries back
    out of a backpointer table built by _dp_full_table."""
    boundaries = [n]
    kk, i = k_groups, n
    while kk > 0:
        j = bp[kk, i]
        boundaries.append(j)
        i, kk = j, kk - 1
    return sorted(boundaries)


def uniform_partition_boundaries(n, k_groups):
    """Boundaries for the naive equal-fine-index-count partition -- the
    fine-grid analogue of notebook 01's uniform ladder, for a fair
    same-method comparison (isolating the effect of boundary placement)."""
    return [round(i * n / k_groups) for i in range(k_groups + 1)]


def objective_of_boundaries(w, q, boundaries, qbar=None):
    """I(Y;R) in bits for an arbitrary (not necessarily optimal) partition,
    for scoring uniform_partition_boundaries against optimal_partition.
    qbar: see optimal_partition -- pass the same externally-supplied
    marginal when scoring a fixed-start partition, so both sides of a
    comparison use the identical reference marginal."""
    n = len(w)
    prefix_w = np.concatenate([[0.0], np.cumsum(w)])
    prefix_wq = np.concatenate([[0.0], np.cumsum(w * q)])
    if qbar is None:
        qbar = prefix_wq[-1] / prefix_w[-1]
    qb = np.clip(qbar, 1e-15, 1 - 1e-15)
    tot = 0.0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        pG = prefix_w[hi] - prefix_w[lo]
        if pG <= 0:
            continue
        qG = np.clip((prefix_wq[hi] - prefix_wq[lo]) / pG, 1e-15, 1 - 1e-15)
        kl = qG * np.log2(qG / qb) + (1 - qG) * np.log2((1 - qG) / (1 - qb))
        tot += pG * kl
    return tot


def group_centroids(x, w, boundaries):
    """Occupancy-weighted centroid position of each group -- the node
    position used to build the coarse chain."""
    cents = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        wg, xg = w[lo:hi], x[lo:hi]
        cents.append(float(np.sum(wg * xg) / np.sum(wg)))
    return np.array(cents)


def find_start_group(boundaries, z0_index):
    """Which group *contains* the fine-grid start index z0_index (the point
    where the accumulator begins, before any evidence has arrived). This is
    an approximation: the chain's node for that group sits at the group's
    occupancy-weighted centroid (group_centroids), not exactly at z0_index,
    so two different partitions (e.g. an optimal one and a uniform one)
    generally place their start rung at two different physical positions
    even though both "contain" the same true start point -- a source of
    noise when comparing them (worse the more the two partitions differ in
    shape near z0_index). optimal_partition_fixed_start /
    uniform_partition_boundaries_fixed_start below remove this
    approximation entirely by forcing z0_index to be its own dedicated,
    single-point group in both constructions; prefer those over this
    function for any comparison that needs the two starting positions to
    match exactly, and use this one only where that isn't the goal."""
    for g, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if lo <= z0_index < hi:
            return g
    return len(boundaries) - 2


def optimal_partition_fixed_start(w, q, k_groups, start_index, qbar=None):
    """Same optimization as optimal_partition (an exact, globally-optimal
    split of the fine grid into k_groups contiguous groups, maximizing
    I(Y;R)), but with one added requirement: the point at start_index is
    forced to be its own dedicated, single-point group, rather than being
    absorbed into whichever group happens to contain it. This is the
    intended use when start_index is the accumulator's true starting
    position z0 -- it guarantees that whatever coarse chain gets built from
    the result actually starts *exactly* at z0 (see group_centroids: a
    single-point group's centroid is that point itself), instead of at an
    approximate, group-dependent stand-in for it. See find_start_group's
    docstring for why that approximation otherwise matters.

    Implementation: split the DP in two independent halves, one for the
    fine-grid points strictly left of start_index and one for the points
    strictly right of it, search over how many of the k_groups-1 remaining
    groups (one is spent on start_index itself) go to each half, and take
    the best combination -- exact, not a heuristic, using the same
    _dp_full_table engine as optimal_partition.

    Requires k_groups large enough to cover both sides plus the start point
    (raises ValueError otherwise -- there is no way to satisfy "start_index
    is its own group" with too few groups when both sides have any mass).
    Returns (boundaries, objective_bits, qbar, start_group_index), where
    start_group_index is which of the k_groups groups (0-based) is the
    dedicated start-point group -- always exactly boundaries[start_group_index]
    == start_index, so the caller never needs to search for it."""
    n = len(w)
    if qbar is None:
        prefix_w = np.cumsum(w)
        prefix_wq = np.cumsum(w * q)
        qbar = prefix_wq[-1] / prefix_w[-1]

    w_left, q_left = w[:start_index], q[:start_index]
    w_right, q_right = w[start_index + 1:], q[start_index + 1:]
    n_left, n_right = len(w_left), len(w_right)

    min_k = 1 + (1 if n_left > 0 else 0) + (1 if n_right > 0 else 0)
    if k_groups < min_k:
        raise ValueError(
            f"k_groups={k_groups} is too small to reserve one group for the exact "
            f"start point plus at least one group on each side that has any "
            f"probability mass; need k_groups >= {min_k} here."
        )

    dp_left, bp_left = _dp_full_table(w_left, q_left, k_groups - 1, qbar)
    dp_right, bp_right = _dp_full_table(w_right, q_right, k_groups - 1, qbar)

    qb = np.clip(qbar, 1e-15, 1 - 1e-15)
    q_start = np.clip(q[start_index], 1e-15, 1 - 1e-15)
    value_start = w[start_index] * (
        q_start * np.log2(q_start / qb) + (1 - q_start) * np.log2((1 - q_start) / (1 - qb))
    )

    # total(m) = value from m groups on the left + the start point's own
    # value + value from the remaining k_groups-1-m groups on the right
    totals = np.array([dp_left[m, n_left] + value_start + dp_right[k_groups - 1 - m, n_right]
                        for m in range(k_groups)])
    m = int(np.argmax(totals))
    best_total = totals[m]
    k_right = k_groups - 1 - m

    boundaries_left = _backtrack(bp_left, m, n_left)
    boundaries_right = [b + start_index + 1 for b in _backtrack(bp_right, k_right, n_right)]
    boundaries = boundaries_left + boundaries_right
    return boundaries, best_total, qbar, m


def uniform_partition_boundaries_fixed_start(n, k_groups, start_index):
    """Uniform-spacing counterpart to optimal_partition_fixed_start: also
    reserves start_index as its own dedicated group (so the "uniform"
    baseline pays the same fixed-start cost as the optimal placement,
    rather than being uniform over the whole range and only incidentally
    landing near start_index), then spaces the remaining groups evenly on
    each side. The k_groups-1 non-start groups are split between the two
    sides in proportion to how many fine-grid points are on each side.
    Returns (boundaries, start_group_index), matching
    optimal_partition_fixed_start's return shape (minus the objective)."""
    n_left = start_index
    n_right = n - start_index - 1
    total_free = n_left + n_right

    m_min = 1 if n_left > 0 else 0
    m_max = (k_groups - 1) - (1 if n_right > 0 else 0)
    if m_min > m_max:
        raise ValueError(
            f"k_groups={k_groups} is too small to reserve one group for the exact "
            f"start point plus at least one group on each side that has any "
            f"probability mass."
        )
    m = int(round((k_groups - 1) * n_left / total_free)) if total_free > 0 else 0
    m = max(m_min, min(m_max, m))
    k_right = k_groups - 1 - m

    boundaries_left = uniform_partition_boundaries(n_left, m) if n_left > 0 else [0]
    boundaries_right = [b + start_index + 1
                         for b in (uniform_partition_boundaries(n_right, k_right) if n_right > 0 else [0])]
    return boundaries_left + boundaries_right, m


# --------------------------------------------- non-uniform chain construction


def nonuniform_ctmc_rates(x_nodes, v, s):
    """Up/down jump rates at each interior node of a non-uniformly spaced
    grid x_nodes = [0, x_1, ..., x_{k-1}, a], from the standard non-uniform
    3-point finite-difference discretization of the generator
    L f = v f' + (s^2/2) f'' (Kushner & Dupuis 2001 "Markov chain
    approximation method"):

        c_-1 = (s^2 - v*h_+) / (h_-*(h_- + h_+))     [down-rate]
        c_+1 = (v*h_- + s^2) / (h_+*(h_- + h_+))     [up-rate]

    where h_- , h_+ are the left/right spacings at that node. Reduces
    exactly to walk_params_from_ddm's p, tau when spacing is uniform (this
    is checked in the notebook, not just asserted). Rates can go negative if
    a grid interval is too coarse relative to v, s (the non-uniform analogue
    of k_min_for_validity) -- returns them anyway so the caller can check,
    rather than silently clipping.
    """
    k = len(x_nodes) - 1
    p_up = np.zeros(k - 1)
    rate_tot = np.zeros(k - 1)
    for idx, i in enumerate(range(1, k)):
        h_minus = x_nodes[i] - x_nodes[i - 1]
        h_plus = x_nodes[i + 1] - x_nodes[i]
        c_minus = (s ** 2 - v * h_plus) / (h_minus * (h_minus + h_plus))
        c_plus = (v * h_minus + s ** 2) / (h_plus * (h_minus + h_plus))
        rate_tot[idx] = c_minus + c_plus
        p_up[idx] = c_plus / rate_tot[idx]
    return p_up, rate_tot


def build_birth_death(p_up):
    """Embedded jump chain (Q, R) for a birth-death chain with per-state
    up-probabilities p_up -- generalizes absorbing_ib.build_absorbing_chain
    to non-constant p. Absorption probabilities of a CTMC depend only on
    this embedded chain, not on the holding times."""
    n = len(p_up)
    Q = np.zeros((n, n))
    R = np.zeros((n, 2))
    for i in range(n):
        p = p_up[i]
        if i - 1 == -1:
            R[i, 0] += 1 - p
        else:
            Q[i, i - 1] += 1 - p
        if i + 1 == n:
            R[i, 1] += p
        else:
            Q[i, i + 1] += p
    return Q, R


def nonuniform_chain_stats(x_nodes, v, s, z0_group):
    """Exact accuracy and expected real time to absorption for the
    non-uniform CTMC on x_nodes, starting in interior group z0_group
    (0-based). Real time uses E[real time] = N @ holding_times, holding
    time at a state = 1/(total jump rate there); exact, no simulation."""
    p_up, rate_tot = nonuniform_ctmc_rates(x_nodes, v, s)
    Q, R = build_birth_death(p_up)
    n = Q.shape[0]
    N = np.linalg.inv(np.eye(n) - Q)
    holding = 1.0 / rate_tot
    absorb = (N @ R)[z0_group]
    expected_time = (N @ holding)[z0_group]
    return dict(p_lower=absorb[0], p_upper=absorb[1], expected_time=expected_time,
                p_up=p_up, rate_tot=rate_tot, holding=holding, N=N)


def simulate_nonuniform_chain(x_nodes, v, s, z0_group, n_trials, n_max, seed=0):
    """Gillespie-style Monte Carlo cross-check: sample the embedded jump
    chain, accumulate real elapsed time as a sum of Exponential(rate) holding
    times (a true CTMC simulation, not the fixed-tau-per-step shortcut valid
    only in the uniform case)."""
    rng = np.random.default_rng(seed)
    p_up, rate_tot = nonuniform_ctmc_rates(x_nodes, v, s)
    n = len(p_up)
    state = np.full(n_trials, z0_group)
    t = np.zeros(n_trials)
    rt = np.full(n_trials, np.nan)
    choice = np.full(n_trials, np.nan)
    active = np.ones(n_trials, dtype=bool)
    for _ in range(n_max):
        if not active.any():
            break
        idx = np.where(active)[0]
        r = rate_tot[state[idx]]
        t[idx] += rng.exponential(1.0 / r)
        p = p_up[state[idx]]
        up = rng.random(idx.size) < p
        state[idx] += np.where(up, 1, -1)
        hit_upper = active.copy(); hit_upper[idx] = state[idx] == n
        hit_lower = active.copy(); hit_lower[idx] = state[idx] == -1
        done = hit_upper | hit_lower
        rt[done] = t[done]
        choice[hit_upper] = 1
        choice[hit_lower] = 0
        active &= ~done
    return rt, choice


# ---------------------------------------------------------------- pipeline


def ib_optimal_chain(k0, k_groups, v, s, a, z_frac):
    """End-to-end: fine reference -> DP-optimal partition -> node positions
    -> exact stats, plus the same-method uniform-boundary baseline for
    comparison. One call, everything needed for a k in a frontier sweep."""
    fr = fine_reference(k0, v, s, a, z_frac)
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

    stats_opt = nonuniform_chain_stats(xn_opt, v, s, z0g_opt)
    stats_unif = nonuniform_chain_stats(xn_unif, v, s, z0g_unif)

    return dict(fr=fr, boundaries_opt=boundaries_opt, boundaries_unif=boundaries_unif,
                obj_opt=obj_opt, obj_unif=obj_unif, qbar=qbar,
                x_nodes_opt=xn_opt, x_nodes_unif=xn_unif,
                z0_group_opt=z0g_opt, z0_group_unif=z0g_unif,
                stats_opt=stats_opt, stats_unif=stats_unif)


def ib_optimal_chain_fixed_start(k0, k_groups, v, s, a, z_frac):
    """Same oracle (known-drift) pipeline as ib_optimal_chain, but using
    optimal_partition_fixed_start / uniform_partition_boundaries_fixed_start
    so both the optimal and uniform chains start from the *exact* true
    starting position z0, not an approximate stand-in for it (see
    find_start_group's docstring). Requires k_groups >= 3 in the typical
    case where z0 is strictly interior (mass on both sides of it) --
    smaller k can't reserve a group for z0 and still cover both sides."""
    fr = fine_reference(k0, v, s, a, z_frac)
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

    stats_opt = nonuniform_chain_stats(xn_opt, v, s, z0g_opt)
    stats_unif = nonuniform_chain_stats(xn_unif, v, s, z0g_unif)

    return dict(fr=fr, boundaries_opt=boundaries_opt, boundaries_unif=boundaries_unif,
                obj_opt=obj_opt, obj_unif=obj_unif, qbar=qbar,
                x_nodes_opt=xn_opt, x_nodes_unif=xn_unif,
                z0_group_opt=z0g_opt, z0_group_unif=z0g_unif,
                stats_opt=stats_opt, stats_unif=stats_unif)
