"""Single-trial dynamics for notebook 00_intuitions.ipynb.

The other utility modules (ib_core.py, absorbing_ib.py, ib_optimal_placement*.py)
compute exact *summary* quantities: stationary distributions, IB frontiers,
absorption probabilities, expected RT. None of them expose the moment-to-moment
trajectory of a single decision, which is the thing this notebook is about:
watching L_t (or X_t) unfold as evidence arrives, at full capacity versus
capacity-limited.

This module adds exactly that layer, reusing the exact machinery already
validated elsewhere wherever possible:

* `glaze_trajectory` implements the memo's normative recursion directly
  (Eq. in the memo's Orientation section), the non-absorbing counterpart of
  the DDM.
* `get_optimal_fsm` is the capacity-limited "mental model" for the bead and
  triangles tasks: a cached, canonicalized wrapper around
  `ib_core.optimal_recursive_fsm`, the genuine (exhaustive-search-or-
  hill-climbed) IB-optimal recursive machine for a given number of states
  `k` and this task's actual `(P, H)`. An earlier version of this notebook
  used a hand-built "step one rung toward the evidence" heuristic instead;
  that heuristic's transition rule never looked at H at all (only its
  readout did), so it silently assumed something close to h=0 and was
  measurably wrong once h grew -- confirmed by comparing it against true
  exhaustive search (see the notebook's Part 1 for the numbers). It has
  been removed; every "ladder" in this notebook is now the actual optimum
  for its own `(k, h, SNR)`, not a fixed guess reused across all of them.
* `simulate_ddm_path` / `simulate_discrete_chain_path` /
  `simulate_nonuniform_chain_path` are trajectory-recording (single-trial)
  variants of absorbing_ib.simulate_ddm / simulate_discrete_chain and
  ib_optimal_placement's nonuniform CTMC -- those return only final RT/choice
  because they are built to vectorize over many trials; here a single path is
  what's needed for a picture.
"""
import numpy as np
from scipy.stats import norm

# ----------------------------------------------------------- generative process


def simulate_switching_sequence(T, h, rng, y0=None):
    """y_t in {0,1}, switching with hazard h (P(y_t != y_{t-1}) = h). The
    shared generative process behind both the bead and triangles tasks."""
    y = np.zeros(T, dtype=int)
    y[0] = rng.integers(0, 2) if y0 is None else y0
    flips = rng.random(T - 1) < h
    for t in range(1, T):
        y[t] = 1 - y[t - 1] if flips[t - 1] else y[t - 1]
    return y


# ----------------------------------------------------------------- beads task


def bead_levels(P=0.8):
    """P0, P1: level-probability rows for jar 0 / jar 1, level 1 = majority
    colour of jar 1 (matches ib_core.hmm_window's default convention)."""
    P1 = np.array([1 - P, P])
    P0 = P1[::-1].copy()
    return np.vstack([P0, P1])  # shape (2, 2): [y, level]


def emit_discrete(y_seq, P, rng):
    """Draw a level index for each y in y_seq, given P[y, level]."""
    nlev = P.shape[1]
    out = np.empty(len(y_seq), dtype=int)
    for t, y in enumerate(y_seq):
        out[t] = rng.choice(nlev, p=P[y])
    return out


def llr_table(P):
    """log[P(level|y=1)/P(level|y=0)] for every level, from a (2, nlev) table."""
    return np.log(np.clip(P[1], 1e-300, None)) - np.log(np.clip(P[0], 1e-300, None))


# ------------------------------------------------------------- triangles task


def discretize_gaussian(nlev, mu, sigma=1.0, span=5.0):
    """Fine-bin discretization of the two triangles' Gaussian emission
    densities N(+mu, sigma^2) (y=1) and N(-mu, sigma^2) (y=0), so that the
    continuous task reuses ib_core's discrete-level machinery exactly
    (fsm_stationary/fsm_scores take a (2, nlev) level table regardless of
    what the levels mean). d' = 2*mu/sigma is the discriminability: LLR(x) =
    2*mu*x/sigma^2 exactly for the continuous Gaussian, so the two ends of
    this discretization converge to the closed-form (unbinned) LLR as nlev
    grows -- checked in the notebook.
    """
    lo, hi = -mu - span * sigma, mu + span * sigma
    edges = np.linspace(lo, hi, nlev + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    P1 = norm.cdf(edges[1:], mu, sigma) - norm.cdf(edges[:-1], mu, sigma)
    P0 = norm.cdf(edges[1:], -mu, sigma) - norm.cdf(edges[:-1], -mu, sigma)
    P1 = P1 / P1.sum()
    P0 = P0 / P0.sum()
    P = np.vstack([P0, P1])
    return dict(centers=centers, edges=edges, P=P, llr=llr_table(P), mu=mu, sigma=sigma)


def emit_continuous(y_seq, mu, sigma, rng):
    """Draw the actual continuous star position x_t for each y in y_seq."""
    signed_mu = np.where(y_seq == 1, mu, -mu)
    return signed_mu + sigma * rng.standard_normal(len(y_seq))


def bin_index(x, edges):
    """Which fine bin a continuous observation falls in (for feeding into a
    ladder built on the discretized alphabet)."""
    return np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)


# ------------------------------------------------------- normative recursion


def glaze_psi(L, h):
    """The Glaze discount, computed via logaddexp for stability at large |L|
    (the memo's ψ(L) = L + log[(1-h)+h e^-L] - log[(1-h)+h e^L])."""
    a = np.logaddexp(np.log1p(-h), np.log(h) - L)
    b = np.logaddexp(np.log1p(-h), np.log(h) + L)
    return L + a - b


def glaze_trajectory(llr_seq, h, L0=0.0):
    """Normative log-posterior-odds trajectory L_t, one value per observation."""
    L = np.empty(len(llr_seq) + 1)
    L[0] = L0
    for t, llr in enumerate(llr_seq):
        L[t + 1] = glaze_psi(L[t], h) + llr
    return L[1:]


# --------------------------------------------------- capacity-limited ladder


def ladder_decoder(f, P, H):
    """Exact stationary decoder q_r = p(y=1|r) for every rung, via
    ib_core.fsm_stationary (leading eigenvector of the induced Markov chain)."""
    from ib_core import fsm_stationary

    J = fsm_stationary(f, P, H)  # (k, 2): joint p(r, y)
    pr = J.sum(1)
    q = np.divide(J[:, 1], pr, out=np.full(f.shape[0], 0.5), where=pr > 1e-15)
    return q, pr


def canonicalize_fsm(f, P, H):
    """Relabel a deterministic FSM's states in ascending order of
    p(y=1|state), so state 0 is always the rung most confident in y=0 and
    state k-1 the rung most confident in y=1 -- needed because a bare
    optimum search has no reason to number its states in belief order, and
    plots/comparisons across different (k, h) machines need a consistent
    axis."""
    q, pr = ladder_decoder(f, P, H)
    order = np.argsort(q)          # order[new_rank] = old_label
    inv = np.argsort(order)        # inv[old_label] = new_rank
    f_canon = inv[f[order]]
    return f_canon, q[order], pr[order]


_optimal_fsm_cache = {}


def get_optimal_fsm(k, P, H, target='I_filter', seed=0, n_restarts=30):
    """The actual IB-optimal k-state recursive machine for this task's
    (P, H) -- exact exhaustive search when tractable, discrete hill-climbing
    otherwise (ib_core.optimal_recursive_fsm), canonicalized into ascending
    belief order and cached (the search is not free, and the same (k, h,
    SNR) combination is often reused across cells/figures).

    Returns a dict: f (canonical transition table), q (belief per rung,
    ascending), pr (stationary occupancy per rung), cap, I_filter, method.
    """
    from ib_core import optimal_recursive_fsm

    key = (k, P.tobytes(), H.tobytes(), target, seed, n_restarts)
    if key not in _optimal_fsm_cache:
        result = optimal_recursive_fsm(k, P, H, target=target, seed=seed, n_restarts=n_restarts)
        f_canon, q, pr = canonicalize_fsm(result['f'], P, H)
        _optimal_fsm_cache[key] = dict(f=f_canon, q=q, pr=pr, cap=result['cap'],
                                        I_filter=result['I_filter'], method=result['method'])
    return _optimal_fsm_cache[key]


def simulate_ladder_path(f, levels_seq, r0):
    """Single-trial ladder-state trajectory r_t given a fixed sequence of
    observed levels and a starting rung r0."""
    T = len(levels_seq)
    r = np.empty(T + 1, dtype=int)
    r[0] = r0
    for t, lev in enumerate(levels_seq):
        r[t + 1] = f[r[t], lev]
    return r[1:]


# --------------------------------------------------------- absorbing (DDM) paths


def simulate_ddm_path(v, s, a, z_frac, dt, t_max, seed=0):
    """Single-trial Euler-Maruyama path of the continuous DDM, stopping at
    absorption or t_max. Trajectory-recording counterpart of
    absorbing_ib.simulate_ddm (which vectorizes over trials and returns only
    the final RT/choice)."""
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(t_max / dt))
    sqrt_dt = np.sqrt(dt)
    x = z_frac * a
    ts, xs = [0.0], [x]
    for step in range(1, n_steps + 1):
        x = x + v * dt + s * sqrt_dt * rng.standard_normal()
        ts.append(step * dt)
        xs.append(x)
        if x >= a or x <= 0:
            break
    return np.array(ts), np.array(xs)


def simulate_discrete_chain_path(k, p, z0_index, n_max, seed=0):
    """Single-trial path (state at every step) of the uniform-ladder discrete
    chain from absorbing_ib.build_absorbing_chain. Trajectory-recording
    counterpart of absorbing_ib.simulate_discrete_chain."""
    rng = np.random.default_rng(seed)
    state = z0_index + 1  # transient labels are 1..k-1
    states = [state]
    for _ in range(n_max):
        state += 1 if rng.random() < p else -1
        states.append(state)
        if state <= 0 or state >= k:
            break
    return np.array(states)


def simulate_nonuniform_chain_path(x_nodes, v, s, z0_group, n_max, seed=0):
    """Single-trial path (real time, position) of the non-uniform IB-optimal
    CTMC from ib_optimal_placement. Trajectory-recording counterpart of
    ib_optimal_placement.simulate_nonuniform_chain (Gillespie-style: holding
    time ~ Exponential(rate) at each visited node)."""
    from ib_optimal_placement import nonuniform_ctmc_rates

    rng = np.random.default_rng(seed)
    p_up, rate_tot = nonuniform_ctmc_rates(x_nodes, v, s)
    n = len(p_up)
    idx = z0_group
    t = 0.0
    ts, xs = [0.0], [x_nodes[idx + 1]]
    choice = None
    for _ in range(n_max):
        t += rng.exponential(1.0 / rate_tot[idx])
        idx += 1 if rng.random() < p_up[idx] else -1
        ts.append(t)
        if idx < 0:
            xs.append(x_nodes[0])
            choice = 0
            break
        elif idx >= n:
            xs.append(x_nodes[-1])
            choice = 1
            break
        else:
            xs.append(x_nodes[idx + 1])
    return np.array(ts), np.array(xs), choice
