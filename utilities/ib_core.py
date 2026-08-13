"""Exact IB machinery for sequential (windowed) evidence-accumulation tasks.

Design notes
------------
* Tasks are hidden Markov processes with binary latent state y_t and a
  W-observation sliding window X = (x_{t-W+1..t}).  Targets:
    'filter'  -> Y = y_t          (state now)
    'predict' -> Y = y_{t+1}      (state next; what the bead task asks)
    'obs'     -> Y = x_{t+1}      (next observation; the literal bead colour)
* Because the IB self-consistent equations depend on x only through
  p(y|x), all x sharing a posterior are exchangeable.  We collapse them
  exactly (I(X;R) = I(Q;R) when Q = p(y=1|X)), which keeps enumeration cheap.
"""
import numpy as np, itertools
from scipy.stats import norm

LOG2 = np.log(2.0)


def hmm_window(W=7, h=0.01, levels=None, target='obs'):
    """Exact joint over (window of W observations, target).

    levels: array of p(level | y=1); mirrored for y=0.  Default = beads 0.8/0.2.
    Returns dict with px (marginal over windows), q (=p(Y=1|x)), and the
    collapsed sufficient-statistic representation.
    """
    if levels is None:
        levels = np.array([0.2, 0.8])           # p(white,black | jar=black-majority)
    P1 = np.asarray(levels, float)               # p(level | y=1)
    P0 = P1[::-1].copy()                         # p(level | y=0)
    P = np.vstack([P0, P1])                      # [y, level]
    nlev = P.shape[1]
    H = np.array([[1 - h, h], [h, 1 - h]])       # H[i,j] = p(y_t=i | y_{t-1}=j)

    seqs = list(itertools.product(range(nlev), repeat=W))
    J = np.zeros((len(seqs), 2))                 # joint p(x_{1:W}, y_W)
    for i, s in enumerate(seqs):
        a = np.array([0.5, 0.5])                 # stationary prior
        for t in range(W):
            if t > 0:
                a = H @ a
            a = a * P[:, s[t]]
        J[i] = a
    px = J.sum(1)
    post_now = J / px[:, None]                   # p(y_W | x)
    post_next = post_now @ H.T                   # p(y_{W+1} | x)
    if target == 'filter':
        pyx = post_now
    elif target == 'predict':
        pyx = post_next
    elif target == 'obs':                        # next observation identity
        pyx = post_next @ P                      # p(x_{W+1}=level | x) ; nlev cols
    else:
        raise ValueError(target)
    llr = np.log(P[1] / P[0])
    Lperf = np.array([sum(llr[v] for v in s) for s in seqs])
    return dict(seqs=seqs, px=px, pyx=pyx, post_now=post_now, post_next=post_next,
                llr=llr, Lperf=Lperf, W=W, h=h, P=P, H=H, target=target)


def collapse(px, pyx, tol=1e-12):
    """Merge x's with identical p(y|x).  Exact: R depends on x only via p(y|x)."""
    key = np.round(pyx / tol).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    n = inv.max() + 1
    pxc = np.zeros(n)
    np.add.at(pxc, inv, px)
    pyc = np.zeros((n, pyx.shape[1]))
    np.add.at(pyc, inv, px[:, None] * pyx)
    pyc /= pxc[:, None]
    return pxc, pyc, inv


def ba(px, pyx, beta, nR=24, iters=6000, tol=1e-13, restarts=12, seed=0):
    """Blahut-Arimoto for the IB.  beta in nats-per-nat (natural units)."""
    rng = np.random.default_rng(seed)
    logpyx = np.log(np.clip(pyx, 1e-300, None))
    Hcond = -(pyx * logpyx).sum(1)
    best = None
    for rs in range(restarts):
        if rs == 0:                              # deterministic quantile init helps
            q = pyx[:, -1]
            edges = np.quantile(q, np.linspace(0, 1, nR + 1))
            idx = np.clip(np.digitize(q, edges[1:-1]), 0, nR - 1)
            prx = np.full((len(px), nR), 1e-6)
            prx[np.arange(len(px)), idx] = 1.0
            prx /= prx.sum(1, keepdims=True)
        else:
            prx = rng.dirichlet(np.ones(nR) * (0.5 + rs), size=len(px))
        for _ in range(iters):
            pxr = px[:, None] * prx
            pr = pxr.sum(0)
            pyr = (pxr.T @ pyx) / np.clip(pr[:, None], 1e-300, None)
            D = -Hcond[:, None] - pyx @ np.log(np.clip(pyr, 1e-300, None)).T
            lg = np.log(np.clip(pr, 1e-300, None))[None, :] - beta * D
            lg -= lg.max(1, keepdims=True)
            new = np.exp(lg)
            new /= new.sum(1, keepdims=True)
            d = np.abs(new - prx).max()
            prx = new
            if d < tol:
                break
        pxr = px[:, None] * prx
        pr = pxr.sum(0)
        pyr = (pxr.T @ pyx) / np.clip(pr[:, None], 1e-300, None)
        Ixr = np.nansum(pxr * np.log2(np.clip(prx, 1e-300, None) /
                                      np.clip(pr, 1e-300, None)[None, :]))
        py = px @ pyx
        Iyr = np.nansum(pr[:, None] * pyr * np.log2(np.clip(pyr, 1e-300, None) /
                                                    py[None, :]))
        # Restart-selection criterion: proportional to the true nats-based IB
        # Lagrangian the fixed-point update above actually optimizes (beta is
        # calibrated against the natural-log D there), since Iyr - Ixr/beta
        # equals that Lagrangian divided by the positive constant LOG2 --
        # a uniform rescaling, so it ranks restarts identically.
        score = Iyr - Ixr / beta
        if best is None or score > best['score'] + 1e-12:
            best = dict(score=score, Ixr=Ixr, Iyr=Iyr, prx=prx, pr=pr, pyr=pyr)
    return best


def beta_critical_binary(px, q):
    """First IB phase transition for binary Y (natural units).

    Degenerate solution destabilises at beta_c = pbar(1-pbar) / Var[q].
    Derivation: expand F = I(X;R) - beta I(Y;R) to second order in a
    perturbation of the collapsed solution; the optimal perturbation is
    proportional to (q - qbar), giving I(Y;R) ~ eps^2 Var(q)^2 / (2 pbar qbar)
    and I(X;R) ~ eps^2 Var(q) / 2.
    """
    qbar = float(px @ q)
    var = float(px @ (q - qbar) ** 2)
    return qbar * (1 - qbar) / var, qbar, var


# ---------------------------------------------------------------- recursive IB
def fsm_stationary(f, P, H):
    """Stationary joint p(r_t, y_t) for a deterministic finite-state accumulator.

    Timing: y_{t-1} -> y_t (via H), x_t ~ p(.|y_t), r_t = f(r_{t-1}, x_t).
    f: (k, nlev) int array giving next state.
    Returns joint (k, nY).
    """
    k, nlev = f.shape
    nY = H.shape[0]
    M = np.zeros((k * nY, k * nY))
    for r in range(k):
        for y in range(nY):
            i = r * nY + y
            for yn in range(nY):
                pyn = H[yn, y]
                if pyn == 0.0:
                    continue
                for xl in range(nlev):
                    M[f[r, xl] * nY + yn, i] += pyn * P[yn, xl]
    w, v = np.linalg.eig(M)
    j = np.argmin(np.abs(w - 1.0))
    p = np.real(v[:, j])
    p = np.abs(p)
    p /= p.sum()
    return p.reshape(k, nY)


def _mi(J):
    J = np.clip(np.asarray(J, float), 0, None)
    tot = J.sum()
    if not np.isfinite(tot) or tot <= 0:
        return 0.0
    J = J / tot
    pa, pb = J.sum(1), J.sum(0)
    den = np.outer(pa, pb)
    m = (J > 1e-15) & (den > 1e-15)
    if not m.any():
        return 0.0
    return float(np.sum(J[m] * np.log2(J[m] / den[m])))


def fsm_scores(f, P, H):
    """Capacity H(R) and I(R;Y) for filtering / prediction / next-observation."""
    J = fsm_stationary(f, P, H)                  # p(r_t, y_t)
    pr = J.sum(1)
    cap = float(-np.nansum(pr * np.log2(np.where(pr > 0, pr, 1))))
    J_pred = J @ H.T                             # p(r_t, y_{t+1})
    J_obs = J_pred @ P                           # p(r_t, x_{t+1})
    return dict(cap=cap, I_filter=_mi(J), I_predict=_mi(J_pred),
                I_obs=_mi(J_obs), pr=pr, J=J)


# ------------------------------------------ exact search over deterministic FSMs
def exhaustive_optimal_fsm(k, P, H, target='I_filter'):
    """Brute-force search over every one of the k**(k*nlev) deterministic
    k-state recursive machines, returning the exact best (a real global
    optimum, not an approximation). Only tractable while that candidate
    count stays small (a few hundred thousand) -- use hill_climb_fsm or the
    optimal_recursive_fsm wrapper otherwise."""
    nlev = P.shape[1]
    best_val, best_f = -np.inf, None
    for combo in itertools.product(range(k), repeat=k * nlev):
        f = np.array(combo, dtype=int).reshape(k, nlev)
        val = fsm_scores(f, P, H)[target]
        if val > best_val:
            best_val, best_f = val, f
    result = dict(fsm_scores(best_f, P, H))
    result['f'] = best_f
    return result


def hill_climb_fsm(k, P, H, target='I_filter', n_restarts=30, seed=0):
    """Discrete coordinate-ascent search over deterministic k-state recursive
    machines: from a random transition table, repeatedly change one
    (state, observation) entry at a time to whichever next-state improves
    `target` most, until no single-entry change helps; repeat from
    `n_restarts` random starts and keep the best. This is the practical
    substitute for exhaustive_optimal_fsm once k**(k*nlev) is too large to
    enumerate -- it reproduces the exact exhaustive-search optimum exactly
    everywhere it's been checked against it (k<=4), so it's trusted as a
    close approximation to the true optimum for larger k, though (like the
    memo's own hill-climbed results) it is not a proof of global optimality."""
    rng = np.random.default_rng(seed)
    nlev = P.shape[1]
    best_val, best_f = -np.inf, None
    for _ in range(n_restarts):
        f = rng.integers(0, k, size=(k, nlev))
        cur = fsm_scores(f, P, H)[target]
        improved = True
        while improved:
            improved = False
            for r in range(k):
                for c in range(nlev):
                    orig = f[r, c]
                    local_best_c, local_best_val = orig, cur
                    for cand in range(k):
                        if cand == orig:
                            continue
                        f[r, c] = cand
                        val = fsm_scores(f, P, H)[target]
                        if val > local_best_val + 1e-12:
                            local_best_val, local_best_c = val, cand
                    f[r, c] = local_best_c
                    if local_best_val > cur + 1e-12:
                        cur = local_best_val
                        improved = True
        if cur > best_val:
            best_val, best_f = cur, f.copy()
    result = dict(fsm_scores(best_f, P, H))
    result['f'] = best_f
    return result


def optimal_recursive_fsm(k, P, H, target='I_filter', seed=0, n_restarts=30,
                           exhaustive_limit=70_000):
    """Best deterministic k-state recursive machine for this (P, H): exact
    exhaustive search when the candidate count is tractable, discrete
    hill-climbing otherwise. This is the one function that should be used
    to build "the" capacity-limited recursive representation -- unlike a
    hand-built heuristic (e.g. a fixed step-by-evidence-sign ladder), the
    search here sees H directly and is free to find whatever transition
    structure that particular hazard rate actually calls for."""
    nlev = P.shape[1]
    if k ** (k * nlev) <= exhaustive_limit:
        result = exhaustive_optimal_fsm(k, P, H, target=target)
        result['method'] = 'exhaustive'
    else:
        result = hill_climb_fsm(k, P, H, target=target, n_restarts=n_restarts, seed=seed)
        result['method'] = 'hill_climb'
    return result


# ------------------------------------------------- stochastic recursive IB
def stoch_stationary(K, P, H):
    """Stationary joint p(r_t, y_t) for stochastic kernel K[r_prev, x, r_next]."""
    k, nlev, _ = K.shape
    nY = H.shape[0]
    M = np.zeros((k * nY, k * nY))
    for r in range(k):
        for y in range(nY):
            i = r * nY + y
            for yn in range(nY):
                pyn = H[yn, y]
                if pyn == 0.0:
                    continue
                for xl in range(nlev):
                    w = pyn * P[yn, xl]
                    M[np.arange(k) * nY + yn, i] += w * K[r, xl]
    w_, v_ = np.linalg.eig(M)
    j = np.argmin(np.abs(w_ - 1.0))
    p = np.abs(np.real(v_[:, j]))
    p /= p.sum()
    return p.reshape(k, nY)


def stoch_scores(K, P, H):
    J = stoch_stationary(K, P, H)
    pr = np.clip(J.sum(1), 0, None)
    m = pr > 1e-15
    cap = float(-np.sum(pr[m] * np.log2(pr[m])))
    Jp = J @ H.T
    return dict(cap=cap, I_filter=_mi(J), I_predict=_mi(Jp), I_obs=_mi(Jp @ P),
                J=J, pr=pr)


def optimise_recursive(k, P, H, lam, n_starts=24, seed=0, target='I_obs', seed_dets=None):
    """Maximise  I(R;target) - lam * H(R)  over stochastic recursive kernels.

    lam is the capacity price (bits per bit); the resulting (H(R), I) points
    trace the recursive analogue of the IB bound.
    """
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    nlev = P.shape[1]

    def unpack(z):
        A = z.reshape(k, nlev, k)
        A = A - A.max(axis=2, keepdims=True)
        E = np.exp(A)
        return E / E.sum(axis=2, keepdims=True)

    def neg(z):
        K = unpack(z)
        try:
            s = stoch_scores(K, P, H)
        except Exception:
            return 1e6
        return -(s[target] - lam * s['cap'])

    best = None
    seeds = list(seed_dets or [])
    for i in range(n_starts + len(seeds)):
        if i < len(seeds):
            f = np.asarray(seeds[i], int)
            z0 = np.zeros((k, nlev, k))
            for r in range(k):
                for c in range(nlev):
                    z0[r, c, f[r, c]] = 6.0
            z0 = (z0 + rng.normal(0, 0.15, z0.shape)).ravel()
        else:
            z0 = rng.normal(0, 2.0 + 0.5 * i, size=k * nlev * k)
        res = minimize(neg, z0, method='Nelder-Mead',
                       options=dict(maxiter=20000, maxfev=20000, fatol=1e-12, xatol=1e-10))
        if best is None or res.fun < best.fun:
            best = res
    K = unpack(best.x)
    s = stoch_scores(K, P, H)
    s['K'] = K
    s['obj'] = -best.fun
    return s
