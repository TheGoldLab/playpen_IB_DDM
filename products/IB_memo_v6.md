
# Adaptive evidence integration as an information bottleneck

**Working memo v5, annotated.** All numbers are computed from the current
computational core (`ib_core.py`); §8 gives the methods.

---

## Abstract

This memo applies the information bottleneck (IB) framework to the problem of adaptive evidence integration in a volatile environment, using the "beads task" in which an observer tracks a latent source that switches unpredictably over time. The central result is that capacity constraints do not blur the normative belief (the Glaze log-posterior-odds variable) but instead quantize it: the IB-optimal representation partitions the belief axis into a small number of contiguous discrete categories rather than producing a noisy, graded continuous belief. This categorical structure arises because the IB objective rewards distinctions, not precision, and is proven to be a geometric necessity given binary KL divergence. The first transition from "ignore evidence" to "hold a binary belief" has a closed-form threshold and is continuous (second-order). In contrast, the analogous transition under recursive inference is abrupt (first-order) because of a non-convexity in the achievable region. For the resulting IB curve, SNR sets the evidence ceiling, whereas capacity alone determines the number of belief categories. Compuiting the bound on-line instead of after experiencing all the observations, which adds a major cost in terms of a loss in predictiveness, is basically a discrete version of the normative (Bayesian) dynamics, involving a leaky accumulator with non-absorbing bounds. 

---

## Orientation

The bead task asks a subject to watch a stream of coloured beads drawn from one
of two jars, where the jar occasionally switches without warning, and to report
which jar is the current source. A good observer accumulates evidence across
beads but discounts old evidence, because the jar may have changed. Glaze et
al. (2015) derived the normative solution: a log-posterior-odds variable
updated by

L_t = ψ(L_{t−1}) + LLR(x_t),
ψ(L) = L + log[(1−h)+h·e{−L}] − log[(1−h)+h·e{L}]

where `h` is the hazard rate. `ψ` discounts old evidence, and its effect is
to bound `L` at a *stabilizing* amplitude `±log[(1−h)/h]` — the point past
which further evidence in the same direction stops moving the belief.

This memo asks what happens when such an observer is **limited in how much it
can retain** about the beads it has seen. The tool is the information
bottleneck (IB): for each budget of retained information, find the strategy
that best predicts the current jar.

### A conceptual guide to the IB setup

Before presenting the results, it is worth being explicit about several
aspects of the IB framework that are easy to misread.

**What is R, and what form does it take?**

R is the observer's internal representation — whatever compressed summary of
the bead sequence the observer retains. The IB optimization places almost no
constraints on R going in: it could be continuous, discrete, high-dimensional,
or any other form. No structure is assumed. What determines R's form is
entirely the optimization itself.

For finite X (e.g., the 128 distinct bead sequences in a W=7 window), a
classical result from convex geometry (Carathéodory's theorem) guarantees that
any point on the IB bound achievable by some continuous R is also achievable
by a discrete R with at most |X| states. Restricting to discrete R is
therefore without loss of generality — nothing is thrown away. The
Blahut-Arimoto algorithm exploits this: it optimizes over a k×|X| matrix of
assignment probabilities p(r|x), treating the k states of R as a free
parameter.

Even though up to 128 states are formally available, the optimization finds
that only a handful earn their keep. The reason is that the evidence ceiling
I(X;Y) — the most any observer could know about the jar — is small (≈0.806
bits at h=0.01). Most states would carry no additional information about the
jar even if used.

**The connection to the "add noise to the posterior" result**

A classical result in IB theory states that the IB-optimal encoder depends on
x only through the posterior over the target: p(r|x) is a function of
q(x) = p(Y=1|x). Put plainly: R should be a noisy function of the posterior
over the latent state.

A note on notation that is easy to trip over: q(x) is written as a function
*of* x (the observation sequence), but it is a probability *over* Y (the
latent jar). It is exactly the posterior over the latent state — the same
quantity as "the posterior" in Parker et al. eq. 5. The two notations
emphasize different things: q(x) stresses that the posterior varies across
observations; "the posterior over Y" stresses what it is a belief about. They
are the same object.

The "noise" in the "add noise to posterior" result lives in the **assignment
to R**, not in q(x) itself. Given a posterior value q(x), the observer does
not deterministically land in one state of R — there is residual randomness
about which state gets assigned. This is the noise. It is categorically
different from adding Gaussian noise to L_t (which would smear the belief
value itself). The output R is still a small set of ordered categories; what
is stochastic is which category a given observation lands in, especially near
the boundaries between categories. As β increases, those boundaries sharpen
into hard thresholds.

**Why the optimal R is discrete rather than a blurred continuous variable**

The intuitive default is that a capacity limit should make the belief *noisy*
— the same graded, continuous confidence, only corrupted. This is not what the
IB produces, for a fundamental reason: **the IB objective pays for
distinctions, not for precision.**

Adding Gaussian noise to a continuous belief pays the full capacity cost of a
high-resolution variable (high-entropy R) while realizing only part of the
benefit. Sorting the belief axis into a few well-separated categories pays for
exactly the distinctions it uses. The optimum is therefore atomic — a small
set of discrete states — and the "noise" from the classical result manifests
as soft, probabilistic boundaries between those states, not as a blur of the
belief value itself.

The reconciliation with Parker et al. eq. 5 is then as follows. Both results
derive from the same IB fixed-point equation. The classical result correctly
says R is a noisy function of q(x). This memo adds the structural question:
*what form does that noisy function take when R is low-capacity?* The answer
is an interval partition with soft boundaries. The two framings are the same
object viewed differently:

| | Variable that is noisy | What R looks like |
|---|---|---|
| Intuitive "blur" model | L_t itself | Continuous, smeared belief |
| IB "add noise" result | Assignment r given q(x) | Discrete state; stochastic at boundaries |

These have different behavioral predictions. The blur model predicts graded
confidence varying smoothly with L_t. The IB model predicts categorical
confidence with a sharp (if noisy) criterion — and what varies with capacity
is how sharp that criterion is, not how graded the belief is.

---

| symbol | name | units | meaning |
|---|---|---|---|
| `h` | hazard rate | — | probability the source switches between trials |
| `d'` | discriminability | — | signal-to-noise of one observation |
| `L_t` | belief | nats | log posterior odds; the Glaze decision variable |
| `log[(1−h)/h]` | stabilizing bound | nats | amplitude at which `L` stops growing |
| `X` | evidence | — | the bead sequence, or equivalently `L` |
| `Y` | target | — | the **current source jar** |
| `R` | representation | — | the observer's internal state |
| `I(X;R)` | **capacity** | bits | what the observer retains about the evidence |
| `I(R;Y)` | **predictiveness** | bits | what that buys in jar-knowledge — the payoff |
| `I(X;Y)` | evidence ceiling | bits | the most any observer could achieve |
| `β` | trade-off | — | price paid per bit of capacity |
| `β_c` | critical `β` | — | below it, the optimum ignores the evidence |
| `k` | states | — | number of distinguishable values `R` takes |

Two cautions on quantities that are easy to conflate. First, `I(X;Y)` and
`log[(1−h)/h]` are different objects with different units — a mutual
information in bits versus an amplitude on the belief axis in nats — and they
are not related by any simple transform. At `h = 0.01` they are 0.806 bits
and 4.595 nats; across `h ∈ [0.002, 0.3]` the bound falls 7.3× while
`I(X;Y)` falls 3.0×, and as `h → 0.5` the bound → 0 while `I(X;Y)` → 0.278
and stays there. Second, **every number in this memo is hazard-dependent**;
each table states its `h`, and §5 gives the hazard dependence explicitly,
since it changes one conclusion materially.

**Terms coined here** (not standard, and not findable in the literature):
*bounded ladder* — an observer holding one of a few ordered belief levels,
moved up or down one level per bead, with the extreme levels non-absorbing
(the same object Glaze et al. call a leaky accumulator with non-absorbing
bounds); *cost of recursion* — the drop in predictiveness from requiring the
observer to update its state from its own previous state rather than
recomputing from the window afresh, at matched capacity.

---

# Part I — The central result: capacity quantizes the belief

## 1. The bottleneck does not find a new variable; it coarsens the normative one

Solve the IB on the bead task with no assumption that the answer resembles the
normative model, then ask afterward what the solution turned out to be a
function of. Two facts answer that, and both are exact rather than
statistical.

![The bottleneck coarsens the normative belief into contiguous
intervals](Figure1_bottleneck_coarsens_normative_belief.png)

**Figure 1. The bottleneck coarsens the normative belief, and coarsens it
into intervals.**

**(a)** The HMM forward posterior `p(y_t = 1|x)` — computed by exact
recursion over the hidden Markov chain, with no reference to Glaze — against
`σ(L_t)`, the logistic transform of the belief produced by the Glaze
recursion on the same sequence. All 128 length-7 windows, `W` = 7, `h` =
0.01; maximum deviation `2.2×10⁻¹⁶`. The Glaze recursion is an exact online
implementation of the forward algorithm, so the bottleneck's only channel to
the stimulus is the normative belief itself. This panel establishes that
`q(x) = p(Y=1|x)` — the posterior over the latent jar — is the sufficient
statistic: the IB has nothing to work with beyond what Glaze already computes.

**(b)** Why the assignment must be an interval partition. Under the IB encoder
the r-dependent part of the exponent is affine in `q(x) = p(y_t = 1|x)`, so
each state's assignment score is a straight line in q whose slope is that
state's decoder log-odds. To see why: at any stationary point of the IB
objective, state r's score for window x is proportional to
`exp(−β · D_KL[p(y|x) ‖ p(y|r)])`. Expanding the KL divergence for binary Y
and collecting terms that depend on both x and r gives a quantity that is
linear (affine) in q(x), with slope equal to state r's decoder log-odds
`log(q_r/(1−q_r))`. The winning state at each q is the upper envelope of
those lines (grey). An upper envelope of straight lines necessarily partitions
the axis into contiguous intervals ordered by slope — a state cannot win at
two separated regions of q without its line crossing above all others twice,
which is impossible for a straight line. Four lines are shown, corresponding
to the four states of (c). This is a proof by geometry: the interval partition
structure is not a choice or assumption — it is forced by the algebra of KL
divergence applied to a binary target.

**(c)** The `β` = 9 solution: stacked assignment probabilities `p(r|x)` for
all 128 windows against `L_t`, colours matching (b), with the interval
boundaries at `L` = −1.49, −0.27, +1.43 nats (dashed). Each colour dominates
a contiguous stretch of the L axis. The soft transitions between colours at
finite β are the "noise" from the classical IB result — near a boundary,
there is genuine uncertainty about which state a window is assigned to. As β
increases, these soft transitions sharpen into hard thresholds. The observer
holds whichever state's decoder value corresponds to their assigned state —
not a smeared version of L_t.

**(d)** The closed-form threshold `β_c = q̄(1 − q̄)/Var[q]` against the
numerical onset recovered from the optimizer, over 7 hazards from `h` = 0.002
to 0.4; maximum relative error 0.1%. `β_c` is the minimum exchange rate
between predictiveness and capacity at which drawing even the first boundary
pays off. The numerator `q̄(1−q̄)` measures prior uncertainty about the jar
on average; the denominator `Var[q]` measures how much the evidence moves the
posterior around across different observation sequences. Their ratio is the
break-even price: when evidence barely moves q (small Var[q]), β_c is large
and the observer needs to value predictiveness highly before encoding anything
pays off. All 7 points fall on the diagonal, validating the formula across a
wide range of hazard rates. See §2.3.

**(e)** `I(X;R)` and `I(R;Y)` just above threshold, both growing linearly in
`ε = (β − β_c)/β_c` (grey line, slope 1): the transition is continuous. `ε`
is the reduced distance from the critical point — standard notation from
statistical physics, analogous to reduced temperature `t = (T−T_c)/T_c` near
a phase transition. Both quantities grow as `ε^1` (slope 1 on a log-log
plot), the defining signature of a continuous (second-order) phase transition.
The observer smoothly transitions from "ignores evidence entirely" to "holds a
barely-distinguishable binary belief" as β crosses β_c — there is no
discontinuous jump. The exponent of exactly 1 follows from the perturbation
argument in §2.3: both the capacity cost and predictiveness gain of a small
split scale as δ², so they cancel and growth near threshold is exactly linear.
Contrast with the recursive transition (§7), which is abrupt (first-order).

**(f)** Their ratio `I(R;Y)/I(X;R)` extrapolates to `1/β_c` = 0.811 as
`ε → 0` — at onset each bit of capacity buys exactly `1/β_c` bits of
predictiveness. This is the break-even exchange rate at the moment of onset:
`β_c` is the price at which retaining information first becomes worthwhile, so
`1/β_c` is the return on the first bit invested. As ε increases (moving
further above threshold), the ratio drops — later bits buy less predictiveness
because the most informative boundary (near q=0.5) is already in use, and
additional capacity goes toward sharpening it or drawing new boundaries in
less informative regions of the belief axis.

The bottleneck therefore does not discover a new variable. It depends on the
stimulus only through `p(y|x)`, which is `σ(L_Glaze)` to machine precision
(a), and what it does to that quantity is cut it into contiguous intervals (b,
c). So the capacity limit does not change **what** is represented. It changes
only the **resolution** at which the normative belief is held. Everything that
follows is a statement about that resolution.

## 2. What "coarsening" turns out to mean: a staircase, not a blur

This is the main result of the memo, and it is worth stating slowly because
the intuitive default is wrong.

The intuitive default is that a capacity limit should make the belief
**noisy** — that a subject with half the information capacity holds the same
graded, continuous confidence, only corrupted. That is not what the
optimization does. Instead, at any capacity below the evidence ceiling, the
optimal representation **partitions the belief axis into a small number of
contiguous intervals** and retains only which interval the belief is in. The
observer holds a *category*, not a blurred number.

![Quantization of the normative belief by the information
bottleneck](Figure2_capacity_quantizes_normative_belief.png)

**Figure 2. Capacity quantizes the normative belief.**

**(a)** IB bound for the bead task (`h` = 0.01, `W` = 7, `Y` = current jar).
The x-axis is information capacity I(X;R) — how many bits the observer
retains about the evidence. The y-axis is predictiveness I(R;Y) — how many
bits that buys about which jar is active. Every point on the curve is the best
possible predictiveness achievable at that capacity — no observer can be above
the curve. Colour gives the number of distinct belief states k in the optimal
solution; the curve is a sequence of discrete branches, each holding a fixed
number of states over a whole range of capacities. This staircase shape is the
central result: the frontier is not smooth because the optimizer finds it
better to sharpen existing boundaries before adding new ones. Within each
branch, extra capacity makes the assignment more deterministic (sharper
thresholds) without creating new categories. New states appear only when
sharpening is exhausted, producing a discrete jump to the next branch. Dotted
line is the evidence ceiling `I(X;Y)` = 0.76 bits. Note that the curve
flattens quickly — most available predictiveness is captured by just 2–4
states.

**(b–d)** The optimal decoder `p(jar = 1 | r)` plotted against the normative
belief `L_t`, at three capacities. Each panel is a **step function of `L`**:
the optimum assigns an interval of the belief axis to each state, with each
horizontal line segment showing one state's decoder value over the range of L
values assigned to it. At 0.74 bits there are two states split near `L` = 0
(b): a left state ("probably jar 2", decoder ≈ 0.06) and a right state
("probably jar 1", decoder ≈ 0.94) — a binary belief. At 1.19 bits, four
states (c): extreme states remain near 0 and 1; two new middle states appear
near 0.2 and 0.8. At 1.79 bits, eight states (d): further carving of the
middle, while tails remain covered by single states. Increasing capacity
changes only how finely the L axis is cut — the observer always holds a
category. Boundaries cluster within roughly `|L| < 2` because the optimizer
places them where (time spent) × (usefulness of resolving there) is maximised
(§2.2).

**(e)** Location of the first transition, `β_c`, versus hazard rate `h`, for
two task instructions (predict current jar vs. predict next jar). `β_c` rises
with `h` because faster switching makes the evidence less useful — the
denominator Var[q] falls as the belief is less reliably moved by evidence when
the source switches frequently. The gap between the two instructions widens
dramatically at high h: `β_c^next = β_c^now / (1−2h)²`, which diverges as
`h→0.5`. At `h=0.40`, `β_c^next ≈ 68.3` while `β_c^now ≈ 1.9` — a
capacity-limited observer should rationally disengage from prediction much
sooner than from filtering as switching gets frequent. This is a
parameter-free prediction about instruction effects, testable by changing only
the question asked at the end of each trial.

**(f)** Growth of capacity and predictiveness just above `β_c`: both are
linear in `(β−β_c)/β_c`, the signature of a continuous (second-order)
transition. Shown here to confirm it holds across the range of hazard rates in
panel (e), not just at `h=0.01`.

### 2.1 Why quantization rather than noise

Three facts, in order.

**First: a bit is a distinction.** The capacity `I(X;R)` measures how many
distinguishable states the representation carries. This is not a metaphor — a
representation carrying one bit can support exactly two reliably
distinguishable states. Since `Y` is binary, everything the observer needs
from its state is a single number, `q_r = p(jar = 1 | r)`. So the "codebook"
the observer must build lives on a one-dimensional interval, and a capacity
budget is directly a budget on how many points it may place in that interval.

**Second: the optimization pays for distinctions, not for precision.** At any
stationary point of the IB objective the encoder must satisfy

p(r|x) ∝ p(r) · exp( −β · D_KL[ p(y|x) ‖ p(y|r) ] )

This is not an assumption; it is what setting the derivative of
`I(X;R) − β I(R;Y)` to zero gives, and it is the update Blahut–Arimoto
iterates. Read it as a scoring rule: each candidate state `r` scores window
`x` by how well `r`'s decoder `p(y|r)` matches `x`'s true posterior
`p(y|x)`, and `β` sets how decisively the best-matching state wins. The cost
term `I(X;R)` charges for *how reliably* `x` determines `r`; the benefit term
charges for *how different* the decoders are. Neither term rewards spreading
mass smoothly over the belief axis. Adding Gaussian-like noise to a continuous
belief pays the full capacity cost of a high-resolution variable while
realizing only part of the benefit; sorting the axis into a few
well-separated categories pays for exactly the distinctions it uses. The
optimum is therefore atomic.

The reconciliation with the classical "add noise to the posterior" result
(Parker et al. eq. 5; Tishby et al. 1999) follows directly. The classical
result correctly says R is a noisy function of q(x). This memo adds the
structural question: *what form does that noisy function take when R is
low-capacity?* The answer is an interval partition with soft boundaries. The
"noise" is not a blur of q(x) — it is uncertainty about which discrete
category a given q(x) maps to, concentrated at the boundaries between
categories.

**Third: capacity buys two separable things, and it buys them in order.**
Within a branch, extra capacity is spent making the assignment **sharper** at
a fixed number of states, and only when sharpening is exhausted does a new
state appear.

![Sharpening within a branch, and the geometry that makes intervals
contiguous](Figure3_sharpen_split.png)

**Figure 3. Sharpen, then split.**

**(a)** The two decoders pull apart. Within the two-state branch, the decoder
values `q_r = p(jar = 1|r)` of the upper and lower state against capacity
`I(X;R)`. At the bottom of the branch they sit at 0.61 and 0.39 — the two
states say almost the same thing — and by the top of the branch at 0.94 and
0.06. The shaded region is the gap between them.

**(b)** Assignment goes from coin-flip to deterministic. Mean assignment
entropy `H[p(r|x)]` against capacity. It falls from 0.957 bits (each window
split almost 50/50 between the two states) to 0.012 bits (each window assigned
to one state). Nothing about the *number* of states changes across this range.
"Sharpness" here is a specific quantity — the mean assignment entropy — not an
impression.

**(c)** The criterion stiffens. `p(upper state|x)` against the Glaze belief
`L_t` at three capacities (0.30, 0.86, 0.99 bits). The crossing stays at
`L` = 0; what changes is the steepness — a soft criterion becomes a hard
threshold. The boundary location is fixed from the start; what capacity buys
is a sharper transition.

**(d)** Higher branches leave alphabet capacity unused. Capacity actually
reached at the top of each `k`-state branch, against the alphabet ceiling
`log₂k` (grey dashed), for `k` = 2 to 14. Only `k` = 2 runs to saturation
(0.992 of `log₂2`); every branch above it tops out well short.

**(e)** Why: the middle states are rarely occupied. Occupancy `p(r)` of the
four states at `β` = 9, against each state's decoder `q_r`. The two extreme
states hold 0.41 each; the two middle states hold 0.09 each, far below the
uniform 0.25 (dashed). New states carve up the rarely visited middle of the L
axis, so they are rarely occupied and the alphabet goes underused.

**(f)** Boundaries go where time × usefulness peaks. Occupancy of the `L`
axis (shaded), the usefulness `|dq/dL|` of resolving there (red), and their
product (black), against `L_t`. The `k` = 4 interval boundaries (grey dashed)
sit at the peaks of the product, not of either factor alone. The belief spends
most of its time near the stabilizing bound (high occupancy) but the posterior
is already saturated there (low usefulness). The informative middle is visited
rarely but changes fast. The optimizer balances both.

That is the whole mechanism: **sharpen, then split, then sharpen, then
split.** It is why the frontier is a staircase and not a smooth curve.

### 2.2 Where the boundaries go, and why the branches are uneven

The intervals are not equal. The optimum places boundaries where they buy the
most, and the rule is a product of two things:

> **Put a boundary where (how often the belief is there) × (how fast
> `p(jar)` changes there) is largest.**

Both factors are strongly non-uniform, and they oppose each other. For the
bead task at `h = 0.01`:

| region of `L` | fraction of time spent there | mean `\|dq/dL\|` |
|---|---|---|
| `\|L\| < 1` | 0.040 | 0.231 |
| 1–2 | 0.066 | 0.150 |
| 2–3 | 0.187 | 0.072 |
| 3–4 | 0.074 | 0.030 |
| `\|L\| > 4` (at the stabilizing bound) | **0.633** | **0.009** |

The belief spends **63% of its time pinned near the stabilizing bound**,
where the posterior is already saturated and additional resolution is nearly
worthless. So the optimal code lumps each saturated tail into a single state
and spends its remaining states carving up the middle — visible directly in
Figure 2b–d, where boundaries cluster within roughly `|L| < 2` and the tails
are flat.

| states `k` | capacity range (bits) | max `I(R;Y)` | branch top ÷ `log₂k` |
|---|---|---|---|
| 2 | threshold – 0.980 | 0.725 | **0.98** |
| 3 | 0.996 – 1.237 | 0.768 | 0.78 |
| 4 | 1.264 – 1.559 | 0.788 | 0.78 |
| 5 | 1.647 – 1.731 | 0.793 | 0.75 |
| 6 | 1.809 – 2.127 | 0.801 | 0.82 |

*(belief space, `h` = 0.01, `|LLR| = log 4`; ceiling `I(L;Y)` = 0.806 bits)*

Only the two-state branch saturates. Beyond it, the new states carve up the
rarely visited middle, so they are rarely occupied and contribute less
capacity than a uniform alphabet would — the code stays unbalanced as it
refines.

### 2.3 The first transition has a closed form

Start at very low `β`, where predictiveness is cheap relative to capacity.
There the optimum is degenerate: one state, used for every window, decoder
equal to the prior, zero capacity and zero predictiveness. The observer
ignores the evidence. Now raise `β` and ask when that solution stops being
optimal.

The `δ` in this argument is a **perturbation probe** — a bookkeeping device
for a stability analysis, not a real quantity in the model. The logic is:
start from the known trivial solution, hypothetically split it into two states
whose decoders differ by a small amount `δ`, and ask whether the IB objective
goes up or down. If it goes up for any `δ > 0`, the trivial solution is
unstable and the observer should split.

Expanding the objective in `δ`: the linear term vanishes (the trivial
solution is a stationary point), and the quadratic term is what decides. Both
the capacity cost and predictiveness gain of the split scale as `δ²`:

- Capacity cost of the split ~ `δ²/[q̄(1−q̄)]`
- Predictiveness gain ~ `δ² · Var[q]/[q̄(1−q̄)]²`

Because both scale identically in `δ`, the ratio gain/cost is independent of
how large the split is. This means "does splitting pay off?" has the same
answer for any `δ > 0` — either it always pays or it never does. `δ` cancels
and the comparison reduces to a single ratio:

β_c = q̄(1 − q̄) / Var_p(x)[q(x)],      q(x) = p(y = 1 | x)

The interpretation is direct: the numerator is how uncertain the target is on
average, the denominator is how much the evidence actually moves the posterior
around. When the evidence barely moves `q`, `β_c` is large and the observer
needs a high price on predictiveness before encoding anything at all.

The `δ`-cancellation is also the mathematical signature of a **continuous
(second-order) transition**. If cost and benefit had scaled differently in `δ`
— say cost ~ `δ` and benefit ~ `δ²` — you would need to commit to a
finite-size split before it paid off. That would produce a discontinuous
(first-order) jump. The identical `δ²` scaling means the transition is
marginal for any split size simultaneously, so capacity grows continuously
from zero at `β_c`. The recursive transition in §7 has a different geometric
structure and is abrupt (first-order) for exactly this reason.

For the bead task at `h = 0.01`, `W = 7`: **β_c = 1.2326**. Measured just
above threshold, `I(X;R) ~ ε^0.95` and `I(R;Y) ~ ε^0.94` with
`ε = (β−β_c)/β_c`, both unity within solver tolerance, and the slope ratio
`I(R;Y)/I(X;R) → 0.8113` against the predicted `1/β_c = 0.8113` — at
threshold each retained bit buys exactly `1/β_c` bits of predictiveness.
Validated independently against the optimizer's own onset across seven hazards
from `h` = 0.002 to 0.4, maximum relative error 0.1% (Figure 1d).

### 2.4 The empirical prediction

Subjects in the manuscript sit almost entirely **below 1 bit** of capacity —
which is the two-state branch, end to end. The prediction is therefore sharp
and unusual:

> A low-capacity observer does not hold a weak graded confidence. It holds a
> **binary belief** — "probably jar A" / "probably jar B" — separated by a
> criterion near `L = 0`, and what varies with capacity is how *sharp* that
> criterion is, not how graded the belief is.

The two-state branch tops out at `I(R;Y)` = 0.725 bits, against the
manuscript's reported 0.66-bit ideal-observer predictiveness — i.e. the
reported human performance sits inside the range a strictly binary observer
can reach. This is directly testable with confidence reports, and it separates
the IB account from graded-belief accounts, which have no reason to predict a
categorical structure at all.

---

# Part II — Robustness of the quantization result

## 3. Continuous-valued observations do not dissolve it

The bead task delivers a binary observation per trial; the Glaze triangles
task delivers a continuous one (a star's horizontal position, drawn from a
Gaussian centred on one of two triangles). A natural worry is that the
quantization result is an artifact of the discrete observation alphabet, and
that with continuous evidence the belief would be genuinely graded. It is not,
and it would not.

**Putting both tasks on one signal-to-noise axis.** The SNR of the bead task
is set by the proportion of the majority color in each jar, which in the
manuscript is p = 0.8 (80% of one color in each jar). This single parameter
determines everything downstream:

- One bead: `|LLR| = log(p/(1−p)) = log(0.8/0.2)` = **log 4 ≈ 1.386 nats**
- Log-odds separation between jars: `D = 2 log 4`
- Equivalent Gaussian discriminability: **`d' = √D = √(2 log 4) ≈ 1.66`**

A different bead proportion — say p = 0.7 — would give
`|LLR| = log(7/3) ≈ 0.85` nats and `d' ≈ 1.30`, a meaningfully weaker task.
All specific numbers in this memo (`|LLR| = log 4`, `d' = 1.66`) trace back
to `p = 0.8`. The bead task is therefore not a weak-evidence task — it is the
triangles task at `d'` = 1.66, below Glaze's two difficulty levels but not
qualitatively apart from them:

| condition | `d'` | ceiling `I(L;Y)` | `β_c` |
|---|---|---|---|
| beads, `p` = 0.8 | 1.66 (equivalent) | 0.806 | 1.172 |
| triangles, matched | 1.66 | 0.862 | 1.112 |
| triangles, Glaze mid (`σ`/distance = 0.33) | 3.03 | 0.963 | 1.027 |
| triangles, Glaze easy (`σ`/distance = 0.24) | 4.17 | 0.989 | 1.008 |

![Beads and triangles under the information
bottleneck](Figure4_window_w_not_capacity.png)

**Figure 4. The window `W` is not a capacity ceiling, and the bottleneck acts
the same way on binary and continuous evidence.** All panels `h` = 0.01
unless stated.

**(a)** Number of distinct values the belief `L_t` can take after `t` beads,
computed exactly by propagating the reachable set (merging at 1e-12, no
grid). The count is `2^t`: because the discount `ψ` is nonlinear, `±log 4`
increments never re-land on a shared value. Verified to `t = 20` (1,048,576
distinct values, no collapse). The nonlinearity of `ψ` means each new path
through belief space is unique. The `W=7` window is not discretizing a
continuous belief — it is limiting the history conditioned on. Doubling holds
for all `h < 0.5`. Dotted line marks the 128 sequences a `W = 7` window
enumerates.

**(b)** Stationary distribution of the belief for all four conditions, binned
at 0.25 nats and normalized to peak. All four conditions share the same
stabilizing bound `±log[(1−h)/h]` = ±4.60 nats (dotted lines), set by hazard
alone — not by SNR. What `d'` changes is how mass distributes *within* that
range. For strong evidence (`d'`=4.17), mass concentrates more sharply at the
extremes — reinforcing the case for a coarse code. All four are bounded in
range but continuous in value.

**(c)** Entropy of the belief quantized at resolution `ε`. Both tasks grow
without bound as `ε → 0`; neither has a capacity ceiling set by its
observation alphabet. The discretization in Figure 2 was never inherited from
the stimulus — it is entirely imposed by the capacity limit.

**(d)** Fraction of the unbounded-memory evidence ceiling that a window of
size `W` recovers at 1 bit of capacity, for hazards 0.002–0.2 (colour, dark =
low hazard) at three bead probabilities. Plotted against raw `W` the curves do
not line up: the window a task needs depends on both hazard and SNR.

**(e)** The same curves against `W / W_eff`, where
`W_eff = log[(1−h)/h] / E[|LLR|]` is the number of observations needed to
traverse the stabilizing bound. All curves collapse, and `W ≈ 2.2 W_eff`
(dotted) suffices to reach 95% of the ceiling (dashed). For the manuscript's
main condition (`p=0.8`, `h=0.01`): stabilizing bound ≈ 4.60 nats,
`E[|LLR|] = log 4 ≈ 1.386` nats/bead, `W_eff ≈ 3.3`, so `W ≈ 7.3` —
confirming `W=7` is well-matched to this condition.

**(f)** The window required for 95% of the ceiling, as a grid across hazard
rate `h` and bead probability `p`. Largest requirements cluster at low `h`
and low `p`. For the manuscript's condition (`p=0.8`, `h=0.01`), `W=7` is
acceptable (92.6%). Blank cells: `W=12` does not suffice.

**(g)** IB frontiers for all four conditions, with `X` = the belief itself.
The shape is identical across conditions — same staircase, same branch
pattern. Only the evidence ceiling (dotted, per condition) changes. SNR lifts
or lowers the curve but does not change its shape. Shading marks the capacity
range where subjects sit — below ~1 bit for most — which is the same relative
position on all four frontiers.

**(h)** Branch map: for each condition, the capacity range over which the
optimal code uses `k` states. Every condition climbs the same staircase, and
the onsets barely move with SNR. Across a 2.5× range of `d'`, `k=3` first
appears at 1.00–1.02 bits, `k=4` at 1.20–1.26 bits. **The number of
categories a subject holds is a statement about their capacity, not about the
difficulty of the stimulus.** SNR determines how much predictiveness each
state buys; the staircase rungs themselves are fixed by capacity alone.
Shading marks the subject range.

Three findings matter here.

**The bead-task belief is already continuous.** Because `ψ` is nonlinear,
successive `±log 4` increments never coincide: the reachable set of `L_t` is
exactly `2^t`, verified to `t = 20` (1,048,576 distinct values, no collapse
at merge tolerance 1e-12) and confirmed for every `h ∈ [0.002, 0.4]`. A
binary observation alphabet does not produce a discrete belief. So the
discreteness in Figure 2 was never inherited from the stimulus; it is imposed
by the capacity limit.

**The cascade survives genuinely continuous evidence.** Running the
bottleneck with `X` = the belief itself, on Gaussian observations at the
Glaze signal levels, reproduces the same staircase:

| task (`h` = 0.01) | ceiling `I(L;Y)` | 2-state branch tops at | `I(R;Y)` there | fraction of ceiling |
|---|---|---|---|---|
| beads, `\|LLR\| = log 4` | 0.806 | 0.980 bits | 0.725 | 90.0% |
| triangles, `d'` = 1.66 (matched) | 0.862 | 1.002 | 0.800 | 92.8% |
| triangles, `d'` = 3.03 (Glaze mid) | 0.963 | 1.014 | 0.935 | 97.1% |
| triangles, `d'` = 4.17 (Glaze easy) | 0.989 | 1.022 | 0.981 | 99.2% |

**Continuous observations do not buy graded belief for free.** Note also that
the two-state branch captures a *larger* fraction of the ceiling as evidence
strengthens, which makes the categorical prediction of §2.4 easier to
satisfy, not harder.

**SNR sets the ceiling; capacity sets the number of states.** These are
separable. Across a 2.5× range of `d'`, the capacity at which each new state
appears barely moves:

| condition | `k` = 2 from | `k` = 3 from | `k` = 4 from |
|---|---|---|---|
| beads (`d'` = 1.66 equiv.) | threshold | 1.00 | 1.26 |
| triangles `d'` = 1.66 | threshold | 1.01 | 1.26 |
| triangles `d'` = 3.03 | threshold | 1.02 | 1.20 |
| triangles `d'` = 4.17 | threshold | 1.02 | — |

The number of categories a subject holds is a statement about their capacity,
not about the difficulty of the stimulus.

## 4. The window `W` is a numerical device, not a capacity ceiling

The manuscript estimates `p(x, y)` by enumerating sequences over a finite
window `W`. It is worth being explicit that this is a tractability device for
estimating the empirical joint distribution and never a claim about subject
memory — windowed and unrestricted Bayesian observers agree on 98.6% of
predictions as reported, and 98.3% by direct simulation (`h` = 0.01,
`W` = 7).

The windowed variable has `H(X) = W` bits, so the windowed problem does have
a formal ceiling of `W` bits. But the IB frontier saturates at `I(X;Y)`
instead, and `I(X;Y)` is smaller by an order of magnitude:

**All rows at `h = 0.01`, `|LLR| = log 4`:**

| W | `H(X)` = capacity `W` allows | `I(X;Y)` = capacity worth using | `I(R;Y)` at 1 bit |
|---|---|---|---|
| 3 | 3 bits | 0.571 | 0.494 |
| 5 | 5 | 0.702 | 0.615 |
| 7 | 7 | 0.760 | 0.674 |
| 10 | 10 | 0.792 | 0.709 |
| 12 | 12 | 0.800 | 0.718 |
| ∞ (belief space) | ∞ | 0.806 | 0.725 |

At this hazard the first column never binds: the `W = 7` and `W = ∞`
frontiers are indistinguishable below ~1.5 bits, and subjects sit below 1
bit. Every capacity range quoted in this memo is therefore a statement about
**belief resolution**, not window size; re-solving with no window at all
changes them by <2%. The range of `L` is set by the hazard's stabilizing
bound (Figure 4b), not by `W`.

## 5. Hazard dependence, including where `W` does bind

### β as a price per bit of capacity

Before reading the hazard-dependence table, it helps to have a concrete
intuition for what `β` means. Rearrange the IB objective to make the
economics explicit:

> maximize: `β · I(R;Y) − I(X;R)`

`β` is the **exchange rate** between predictiveness and capacity — how many
bits of predictiveness justify spending one bit of capacity. If `β=2`, you
will spend up to 2 bits of capacity to gain 1 bit of predictiveness. If
`β=10`, you will spend 10 bits for that same gain.

`β_c` is the **minimum viable exchange rate** — the lowest `β` at which
drawing even the first boundary pays off. Below `β_c`, no split ever earns
back its capacity cost. Above it, any split pays — even an infinitesimally
small one earns more predictiveness than it costs in capacity.

Critically, `β_c` is a property of the **task**, not of the observer. It
depends on how much the evidence moves the posterior (`Var[q]`) and how
uncertain the target is on average (`q̄(1−q̄)`). An observer with a fixed `β`
will engage with evidence in tasks where `β > β_c` and ignore it in tasks
where `β < β_c`. This makes `β_c` the key parameter for predicting when a
capacity-limited observer should disengage from evidence encoding entirely.

The ceiling falls steeply with hazard — a faster-switching source is less
predictable — and `β_c` rises with it. Belief space, beads,
`|LLR| = log 4`:

| `h` | stabilizing bound (nats) | `I(X;Y)` (bits) | `β_c` |
|---|---|---|---|
| 0.002 | 6.213 | 0.931 | 1.048 |
| 0.005 | 5.293 | 0.873 | 1.100 |
| **0.01** | **4.595** | **0.806** | **1.172** |
| 0.02 | 3.892 | 0.717 | 1.292 |
| 0.05 | 2.944 | 0.571 | 1.567 |
| 0.10 | 2.197 | 0.455 | 1.899 |
| 0.20 | 1.386 | 0.353 | 2.334 |
| 0.30 | 0.847 | 0.307 | 2.591 |

In words: when the world switches often, evidence about the current state is
worth less, so a capacity-limited observer requires a larger `β` — a lower
effective price per bit — before encoding anything at all. At `h = 0.3`
predictiveness must be valued 2.6× the bit price; at `h = 0.002`, barely
1.05×.

**`W` binds at low hazard.** The fraction of the `W = ∞` frontier a window
recovers, at 1 bit of capacity:

| `h` | `W`=3 | `W`=7 | `W`=12 |
|---|---|---|---|
| 0.002 | 0.573 | **0.852** | 0.958 |
| **0.01** | 0.678 | 0.926 | 0.988 |
| 0.05 | 0.881 | 0.991 | 1.000 |
| 0.20 | 0.996 | 1.000 | 1.000 |

`W = 7` is adequate at `h ≥ 0.05` (>99%) and acceptable at `h = 0.01`
(92.6%), but **not at `h = 0.002` (85.2%)**. The requirement is set by
`W_eff`, the number of observations needed to traverse the stabilizing bound
(Figure 4d–f): windows collapse onto a single curve in units of `W_eff`, and
`W ≈ 2.2 W_eff` suffices for 95% of the ceiling. So `W` is never a capacity
ceiling, but the claim that it is *non-binding* holds only for `h ≳ 0.01`;
below that, use the belief-space solve.

**A dissociation the design can exploit.** `β_c` rises with `h` in both
tasks, but at very different rates:

| `h` | `β_c` beads | `β_c` triangles (`d'` = 3.03) |
|---|---|---|
| 0.002 | 1.048 | 1.008 |
| 0.01 | 1.172 | 1.027 |
| 0.05 | 1.567 | 1.081 |
| 0.10 | 1.899 | 1.124 |
| 0.20 | 2.334 | 1.179 |
| 0.30 | 2.591 | 1.213 |

A 2.5× swing against a 1.2× swing — a genuine `h × d'` interaction, not two
main effects. A capacity-limited observer's threshold for engaging with the
evidence at all should be strongly volatility-sensitive when evidence is weak
and nearly volatility-insensitive when it is strong. A second, separable
prediction concerns instruction: with `Y` = *next* jar rather than current
jar, `β_c^next = β_c^now / (1 − 2h)²` exactly (1.283 at `h` = 0.01 rising
to 68.3 at `h` = 0.40). Prediction becomes catastrophically harder to justify
representing as volatility rises, while filtering degrades gently.

*Range caution:* Glaze's blocks run to `h = 0.95`. Everything above is
`h ≤ 0.45`. At `h = 0.5` the discount vanishes identically; above 0.5 the
source alternates more often than it persists, inverting the sign of useful
evidence. That regime needs the theory restated, not merely re-run.

---

# Part III — Recursion

## 6. Requiring the belief to be updated from itself is costly, and costly exactly where subjects are

Everything above solves a **one-shot** problem: `R` is computed from the
evidence in a single step, with no requirement that it be updatable from its
own previous value. Real accumulation is **recursive**: `r_t = f(r_{t−1},
x_t)`, one new observation at a time. Whether that restriction costs anything
is the central open question of the project, and it does.

**How a recursive accumulator appears on these axes.** A recursive machine is
not an IB solution — it is a mechanism, and it gets *scored* on the same two
axes. Specify `k` states and a transition rule `r_t = f(r_{t−1}, x_t)`; run
that chain to its stationary distribution over `(r_t, y_t)`; read off
`I(R;Y)` for the vertical axis and the capacity the machine consumes for the
horizontal one. Each machine is one point, and the frontier is the upper
envelope of the point cloud.

The horizontal axis requires care. For a **deterministic** machine, `r_t` is
a function of the entire past, so `I(x_{1:t}; r_t) = H(R)` and that entropy
is directly comparable to the one-shot `I(X;R)`. For a **stochastic** kernel
the identity fails — `H(R)` then also counts the noise the kernel injects,
which is not information about the stimulus — so only deterministic machines
are plotted and only they enter the envelope.

![Cost of recursion and the optimal recursive
accumulator](Figure5_recursion_costly_at_low.png)

**Figure 5. Recursion is costly at low capacity, and its optimum is a bounded
ladder.** Bead task, `h` = 0.01; one-shot reference is `W` = 12
(`I(X;Y)` = 0.7997), since the recursive machine sees unbounded history.

**(a)** The one-shot IB bound (blue) against every deterministic recursive
machine found (grey points, `k ≤ 8`) and their upper envelope (orange). The
shaded region between the two curves is the **cost of recursion** —
predictiveness lost by requiring the observer to update from its own previous
state rather than recomputing from the evidence afresh. This cost is 13–38%
of the one-shot bound in the subject range (below ~1 bit) and shrinks to zero
by ~2.25 bits. Dotted line is the unbounded-memory ceiling. Grey band is the
capacity range where subjects sit.

**(b)** Ratio of the recursive envelope to the one-shot bound — what a
*perfectly efficient* recursive observer scores when graded against the
one-shot bound. This is a **prediction**, not a measurement: even a subject
who is perfectly optimal given recursive constraints will appear 62–87%
efficient against the one-shot bound in the subject range. The apparent
inefficiency in the low-capacity tail may be the signature of recursive
inference rather than of suboptimality. Testable by refitting subjects against
the recursive bound on the existing dataset.

**(c)** The optimal 4-state machine as a state diagram. Nodes are the four
states (ordered left to right), with decoder values
`p(jar=1|state) = 0.02 / 0.23 / 0.77 / 0.98` shown below. A green bead
moves the state one rung up; a red bead moves it one rung down; the extreme
states are non-absorbing. This is a **bounded ladder**: a leaky
accumulator with non-absorbing bounds (e.g., from Glaze et al.) but here in discrete form. Here it falls out of exhaustive search over all 65,536 possible
wirings of 4 states and 2 bead colors, with no structural assumptions. The
decoder values are ordered but not equally spaced: outer states are more
extreme, covering the tails where the belief spends most of its time; inner
states cover the informative middle. Note that the optimal 4-state recursive accumulator has the same architectural structure as the Glaze leaky accumulator — ordered states, evidence-driven updates, non-absorbing bounds — and can be understood as the optimal discrete approximation to the Glaze filter under a capacity constraint. However, the transition dynamics differ: the Glaze model applies a continuous, L-dependent leak that compresses extreme beliefs back toward the stabilizing amplitude before each update, while the IB discrete model applies a uniform ±1 step regardless of position within a state. The two converge in the limit of large k (many discrete states) but diverge substantially for extreme states at k = 4.

**(d)** The achievable envelope (orange) against the upper convex hull of the
same point cloud. A `β` sweep can only select points on the upper convex
hull. If the achievable region has an inward dent, the `β` sweep jumps across
it — no value of `β` selects anything inside. The recursive achievable region
has such a dent: the hull exceeds the achievable envelope by up to 0.28 bits,
centered around 0.92 bits, across 0.05–2.07 bits. Consequences: (1) the
recursive frontier must be computed by constrained search at fixed capacity,
not by sweeping `β` — a `β` sweep silently overstates what recursion can do;
(2) the transition is abrupt (§7); (3) the dent sits in the subject range.

**(e)** Best `I(R;Y)` against the number of states `k` in the ladder.
`k ≤ 4` exhaustively enumerated; `k ≥ 5` hill-climbed (lower bounds).
`k=2`: 25% of ceiling; `k=4`: 86%; `k=6`: 100%; `k=8`: 99.4%. Returns
diminish quickly — a 6-state ladder is essentially perfect. Bounded, discrete
accumulation is nearly lossless when capacity permits.

**(f)** The cost-of-recursion table (§6 below), plotted, with the ratio of
recursive to one-shot annotated at four capacities. Blue is the one-shot
bound; orange is the best recursive envelope. Given a subject's estimated
capacity `H(R)`, the orange curve gives the predictiveness a perfectly
efficient recursive observer should achieve at that capacity — suitable for
fitting to subject data.

**The optimal recursive accumulator is a bounded ladder.** Exhaustive search
over all 65,536 wirings of 4 states and 2 bead colours returns
`next-state[state,bead] = [[0,1],[0,2],[1,3],[2,3]]` with
`p(jar=1|state) = [0.02, 0.23, 0.77, 0.98]`. This is precisely the
architecture Glaze et al. derive analytically — a leaky accumulator with
non-absorbing bounds — recovered here as the solution to an
information-constrained optimization rather than assumed. The optimal `k = 2`
machine ignores its own previous state entirely, making it exactly the
one-back heuristic the manuscript already fits: the one-back rule is not a
shortcut but the *optimal* recursive strategy at 1 bit. An 8-state ladder
reaches 0.8007 bits, **99.4% of the unbounded-memory ceiling** — bounded,
discrete accumulation is nearly lossless when capacity permits.

**The cost is concentrated below 2 bits.** Envelope over all 34,564
deterministic machines:

| capacity (bits) | one-shot bound | best recursive | ratio |
|---|---|---|---|
| 0.25 | 0.161 | 0.021 | 13% |
| 0.50 | 0.339 | 0.056 | 17% |
| 0.75 | 0.589 | 0.130 | 22% |
| 1.00 | 0.729 | 0.278 | 38% |
| 1.25 | 0.768 | 0.328 | 43% |
| 1.50 | 0.786 | 0.547 | 70% |
| 1.75 | 0.792 | 0.694 | 88% |
| 2.00 | 0.798 | 0.770 | 96% |
| 2.25 | 0.799 | 0.799 | 100% |

At high capacity recursion is free; in the subject range a recursive
accumulator reaches **13–38%** of what an unconstrained one-shot code
achieves at the same capacity.

**This reinterprets the manuscript's efficiency measure.** Subjects are scored
against the one-shot bound, and most sit below 1 bit — exactly where the two
bounds diverge by a factor of three or more. A subject who is *perfectly
efficient given that their inference must be recursive* will look
substantially *inefficient* against the one-shot bound. The apparent
inefficiency in the low-capacity tail may be the signature of recursion rather
than of suboptimality, and this is testable on the existing dataset by
refitting the same subjects against the recursive bound.

## 7. The recursive transition is abrupt

The one-shot transition of §2.3 is continuous: capacity grows smoothly from
zero at `β_c`. The recursive problem does not behave that way, and the reason
is geometric.

A `β` sweep optimizes `I(R;Y) − (1/β)·I(X;R)`, which is a linear objective
on the `(capacity, predictiveness)` plane. A linear objective can only ever
select points on the **upper convex hull** of the achievable region. If the
region is convex, the hull and the region's upper boundary coincide and the
sweep finds everything. If the region has an inward dent, the sweep jumps
across it: as `β` increases, the optimum leaps discontinuously from one side
of the dent to the other, and no value of `β` selects anything inside.

The bead task's recursive region has such a dent. Taking all 34,564
deterministic machines, constructing their upper convex hull, and comparing
it against the achievable envelope (Figure 5d): the hull exceeds the envelope
by up to **0.28 bits**, at 0.92 bits of capacity, with a gap above 0.01 bits
across the range 0.05–2.07 bits. Good machines exist throughout that range —
they are simply unreachable by any `β`.

That is the signature of an **abrupt (first-order)** transition: the optimal
strategy switches discontinuously rather than deforming smoothly. The one-shot
and recursive problems therefore change character in fundamentally different
ways, and the difference is not an artifact of the search — it is a property
of which `(capacity, predictiveness)` pairs finite recursive machines can
occupy at all.

Two practical consequences. First, any recursive frontier must be computed by
constrained search at fixed capacity, never by sweeping `β`; a `β` sweep
silently returns the hull and overstates what recursion can do in the dent.
Second, the dent sits squarely in the subject range, which is where §6's
reinterpretation of the efficiency measure lives.

---

## 8. Methods

**Windowed problem.** For window `W`, enumerate all `2^W` bead sequences;
compute `p(x)` and `p(y|x)` exactly by forward recursion over the HMM (no
sampling). Merge sequences with identical `p(y|x)` — they are
interchangeable for any IB solution. Solve `min I(X;R) − β I(R;Y)` by
Blahut–Arimoto with restarts, swept over `β`. `Y` is the **current source
jar** throughout; the manuscript's reported ideal-observer values (0.66 bits,
93.9% accuracy) identify it as such, against 0.760 bits / 94.1% for current
jar, 0.712 / 93.2% for next jar, and 0.214 / 75.9% for next bead colour.

**Belief-space problem (no window).** Build the stationary joint `p(L_t,
y_t)` for the Glaze recursion on a grid of `L` (n = 1201–2001 nodes spanning
the stabilizing bound plus one observation's evidence). Binary evidence splits
probability mass linearly between neighbouring nodes; Gaussian evidence uses
CDF differences. The kernel is asserted exactly stochastic, then iterated to
its stationary distribution. Validation: the bead chain's `I(L;Y)` reproduces
the independently computed unbounded-memory ceiling 0.8058 bits, and
`p(y=1|L)` matches `σ(L)` to machine precision.

**SNR matching.** One bead carries `|LLR| = log 4`, giving a log-odds
separation `D = 2 log 4` between jars; the Gaussian observation with the same
separation has `d' = √D` = 1.66. Glaze's `σ`/distance ratios 0.24 and 0.33
correspond to `d'` = 4.17 and 3.03.

**Support growth** (Figure 4a) is exact: the reachable set of `L_t` is
propagated atom-by-atom with merging at 1e-12, no grid.

**State counting.** Solutions are collapsed on `q = p(y=1|L)`, since the IB
depends on `x` only through `q`. Counting states by an absolute separation
threshold is unreliable near `β_c`, where every decoder converges on the
prior and numerically split copies of one state look distinct — such a rule
reports three states at 0.001 bits of capacity, which `I(L;R) ≤ log₂k`
forbids. States are therefore counted by an information criterion instead:
repeatedly merge the two closest decoders (mass-weighted) while the merge
costs less than 1% of `I(R;Y)`, and stop when it does not. The reported `k`
is the number of states carrying distinguishable information, and it is
scale-free — no absolute tolerance enters.

**Recursive problem.** Stationary joint `p(r_t, y_t)` as the leading
eigenvector of the induced Markov chain; capacity `H(R)`, predictiveness
`I(R;y_t)`. Validated against 3×10⁶-step simulation (3–4 decimals).
Deterministic machines exhaustively enumerated for `k ≤ 4` (34,136 machines
at `k` = 4), hill-climbed with hundreds of restarts for `k ≤ 8`; stochastic
kernels by Nelder–Mead seeded from the best deterministic machines. Frontiers,
envelopes and the convex-hull test of §7 use **deterministic machines only**,
since `I(x_{1:t}; r_t) = H(R)` holds only for them; mixing machine classes
on this axis is not meaningful. The one-shot reference is `W = 12`
(`I(X;Y)` = 0.7997), not `W = 7` (0.7598), because the recursive machine
sees unbounded history.

---

## 9. What this buys the grant

1. **A mechanism for categorical belief.** Capacity limits do not blur the
   normative belief; they quantize it (§2). The prediction that low-capacity
   subjects hold binary belief with a sharp criterion — not weak graded
   confidence — is testable with confidence reports and has no counterpart in
   graded-belief accounts.
2. **Two knobs that do different jobs.** SNR sets the ceiling and `β_c`;
   capacity sets the number of categories (§3). Across a 2.5× range of `d'`
   the state-count onsets move by less than 0.06 bits, so the number of
   categories a subject holds is diagnostic of their capacity rather than of
   task difficulty.
3. **The normative model explained rather than replaced.** Glaze's leaky
   accumulator with non-absorbing bounds falls out of
   information-constrained recursive inference (§6) instead of being assumed.
4. **A separable two-parameter prediction grid.** `β_c` falls with `d'` and
   rises with `h`, with a strong interaction between them (§5), plus a
   parameter-free instruction effect
   `β_c^next/β_c^now = 1/(1−2h)²`. The existing Glaze dataset (48 subjects,
   1000-trial blocks, three `d'` levels) already spans it.
5. **A reinterpretation of the manuscript's efficiency measure.** §6 predicts
   that low-capacity subjects appear inefficient against the one-shot bound
   even when optimal, and quantifies by how much — testable on existing data.
6. **Two transitions of different order** (§2.3, §7): the one-shot transition
   is continuous with a closed-form threshold, the recursive one is abrupt,
   and the difference is a property of the achievable region rather than of
   the search.

## 10. Open items

- Does the §7 dent survive `k` > 8? The hull test is exact for the machines
  enumerated, and `k ≤ 4` is exhaustive, but `k ≥ 5` rests on hill-climbing.
- Can the recursive threshold be derived analytically, to match §2.3's closed
  form?
- Refit the manuscript's subjects against the recursive bound (§6) — needs
  raw choice data.
- Extend the theory above `h = 0.5` to cover Glaze's high-hazard blocks
  (§5).
- Confidence-report design to test the categorical prediction of §2.4
  directly.
- Continuous-state accumulators (the Glaze diffusion limit) rather than
  finite `k`.
- Reconcile the two-state branch top onto a single `β` grid: 0.9808 bits in
  `belief_space_summary.csv` versus 0.9916 in `branch_alphabet_usage.csv`
  (§2).

## 11. Files

*Regenerated from the artifact store for v5; entries that no longer resolve
have been removed and previously undocumented outputs added.*

**Code**

- `ib_core.py` — computational core: windowed HMM enumeration, exact
  collapse, Blahut–Arimoto, closed-form `β_c`, deterministic and stochastic
  recursive solvers, simulation validators.

**Belief-space sweeps and the branch map (Parts I–II)**

- `belief_space_ib_sweep.csv` — belief-space IB sweeps, all four task
  variants (`h` = 0.01).
- `branch_map_belief.csv` — state count versus capacity for all four
  conditions, under the §8 counting rule (Figure 4h, §3 table).
- `belief_space_summary.csv` — per-task ceilings, `β_c`, two-state branch
  extent.
- `sharpening_within_branch.csv` — decoder separation, mean assignment
  entropy and mean confidence across the two-state branch (Figure 3a–c).
- `branch_alphabet_usage.csv` — capacity reached at the top of each `k`-state
  branch against the `log₂k` alphabet ceiling, `k` = 2–14 (Figure 3d).

**Window adequacy and capacity ceilings**

- `window_vs_belief_frontier.csv` — windowed frontiers, `W` = 3, 5, 7, 10,
  12.
- `W_adequacy_hazard_snr.csv`, `W_eff_requirement.csv` — window adequacy
  across hazard and SNR, and the `W_eff` collapse (Figure 4d–f, §5).
- `capacity_ceiling_comparison.csv` — `H(X)` vs `I(X;Y)` (§4).
- `ib_bound_W12.csv` — `W` = 12 reference bound.

**Hazard dependence (§5)**

- `hazard_dependence.csv`, `beta_c_vs_hazard.csv`,
  `hazard_frontier_at_1bit.csv`.
- `ib_leak_hazard_graded.csv` — optimal leak `λ*` and its within-tolerance
  band against hazard, with the Glaze `1 − 2h` comparison.

**One-shot and accumulation analyses**

- `ib_bead_jar_sweep.csv` — one-shot bead-jar sweep.
- `ib_accumulation_results.csv` — IB maximum against normative and
  perfect-integration bits across `h` and `T`, with efficiency ratios.
- `ib_capacity_leak_tradeoff.csv` — best leak and its tolerance band as a
  function of capacity budget.
- `ib_dynamic_bottleneck_sweep.csv` — dynamic-bottleneck sweep over `β`
  (`I(X;R)`, `I(R;Y)`, cluster count, `I(R;L)`).

**Recursion (Part III)**

- `recursive_points_all.csv` — all recursive machines found.
- `recursive_envelope_true.csv` — achievable envelope over deterministic
  machines (Figure 5a,d).
- `recursive_constrained_frontier.csv` — constrained frontier by `k` and
  capacity.
- `cost_of_recursion_corrected.csv` — the §6 table.

**Notes**

- `memo_QA_log.md` — audit log; §Q1b records the withdrawn sufficiency ratio
  and §Q2 the exact provenance of Figure 1.
- `lab_corpus_index.csv` — index of the lab reference corpus (251 entries);
  used for citation lookup, not an analysis output.

**Removed from this list (no longer in the artifact store)**

- `belief_ib_solutions.pkl` — stored Blahut–Arimoto solutions behind the
  branch map. Regenerate with `ib_core.ba` if the branch map needs
  rebuilding.
- `one_shot_frontier_beads.csv` — superseded by `ib_bead_jar_sweep.csv`.