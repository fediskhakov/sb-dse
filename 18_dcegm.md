---
title: 📖 DC-EGM for discrete-continuous choice
short_title: 📖 DC-EGM
subtitle: Class 18 — Tuesday, October 27
downloads:
  - file: 18_dcegm.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

The [EGM of Class 17](17_egm.md) solves the consumption-savings model without a single
root-finding operation, but implicitly assumes that the problem is concave. Today we add a discrete
choice, retirement, on top of consumption, and code the **DC-EGM** of
{cite:t}`egm`: an EGM step per discrete choice, an upper envelope to throw away the
points the Euler equation gets wrong, and taste shocks to make the whole thing smooth.

````{seealso} Key reading

{cite:t}`egm` "The endogenous grid method for discrete-continuous dynamic choice models
with (or without) taste shocks", *Quantitative Economics* 8(2), 317–365. The retirement
model, the DC-EGM algorithm and the logit smoothing of this class all come from this
paper.
````

````{hint} Running the code for this lecture
:class: dropdown

The code for this class is in the folder `session18-oct27/` of the course **code
repository**. The notebook `dcegm.ipynb` has all the code of this page.

You should have cloned that repository already — if not, the instructions are in the
[algorithms and complexity lecture](3_algo.md#clone-code-repo).

Update your copy before the class. Editing a file in place makes `git pull` refuse to
update it, so discard whatever you changed while experimenting:

```bash
cd sb-dse-code
git reset --hard HEAD
git pull
```

Nothing in this repository is submitted, so experiment freely — but copy anything you
want to keep out of it first, or commit that to a branch of your own.

Setting up the Python environment is covered in
[](2_workflow.md#python-install).
````

## Discrete and continuous choice together

In economics the two kinds of choice co-exist:

- how much to work, *and* when to retire or become an entrepreneur
- how much to save, *and* when to buy a house, a car, a durable
- which car to buy, *and* how much to drive it

Traditional methods treat them separately: 
- a discrete choice is a maximum over a finite set 
- a continuous choice is a first order condition under concavity
  
Put together, the pieces break:
- the problem is **intrinsically non-concave**: the value function is the maximum of
  several concave functions, one per discrete alternative
- **first order conditions are not sufficient**: the Euler equation has several solutions,
  not all are optimum
- **value functions have kinks and policy functions have discontinuities**, which a
  fixed grid smears and an optimizer trips over

So a traditional solver needs a global optimizer in every point of the state space, has to
locate and track every kink, and needs special numerical procedures for non-smooth
objects. 

**DC-EGM** keeps the two things EGM is good at, no root finding and an exact
treatment of the credit constraint, and adds the machinery for the kinks.

The class of problems is that of [EGM](17_egm.md), extended:

- **A1.** strictly concave, monotone and differentiable utility
- a continuous state $M_t$, wealth, with the budget constraint as its law of motion
- additional discrete states $st_t$, such as the labor market status, with
  **A2.** transition probabilities of $st_t$ independent of $M_t$
- one continuous choice $c_t$, consumption, and one discrete choice $d_t$, which drives
  the discrete states

DC-EGM comes in two flavors. 

1. **Without taste shocks** it iterates on the value and
policy functions and produces exact thresholds for the discrete decisions. 

2. **With taste
shocks** it iterates on choice-specific value functions and produces choice
probabilities, as in the [Rust model](9_zurcher.md#zurcher-choice-probabilities).

:::{div}
:class: discussion

- Why is the maximum of two concave functions not concave?
:::

(retirement-model)=
## The retirement model

A worker in state $\mathbb{W}$ chooses consumption $c$ and whether to keep working or
retire, $d \in \{\mathbb{W}, \mathbb{R}\}$. A retiree in state $\mathbb{R}$ only chooses
consumption, because retirement is absorbing:

$$
V_t(M_t, \mathbb{W}) = \max\left\{
\begin{array}{l}
\max_{0 \leqslant c \leqslant M_t} u(c, \mathbb{R}) + \beta\, \mathbb{E} V_{t+1}\big(\tilde R (M_t - c), \mathbb{R}\big) \\[4pt]
\max_{0 \leqslant c \leqslant M_t} u(c, \mathbb{W}) + \beta\, \mathbb{E} V_{t+1}\big(\tilde R (M_t - c) + y, \mathbb{W}\big)
\end{array}
\right\}
$$

$$
V_t(M_t, \mathbb{R}) = \max_{0 \leqslant c \leqslant M_t} \Big[ u(c, \mathbb{R}) + \beta\, \mathbb{E} V_{t+1}\big(\tilde R (M_t - c), \mathbb{R}\big) \Big]
$$

- $\mathbb{R}$, $\mathbb{W}$ are the retirement and working *states* $st_t$, which evolve
  according to the discrete *choices* $d \in \{\mathbb{R}, \mathbb{W}\}$
- $y$ is the wage, deterministic for now; a retiree has no income
- $\tilde R$ is the gross return on savings, possibly random; we keep it fixed at $R$
- the horizon is finite, $t = 0, 1, \dots, T-1$, and in the last period everything is
  consumed

Utility is additively separable in consumption and the discrete choice, with a
disutility of work $\delta$:

$$
u(c, d) = \frac{c^{\rho} - 1}{\rho} - \delta\, \mathbf{1}(d = \mathbb{W})
\;\;\underset{\rho \to 0}{\longrightarrow}\;\;
\log(c) - \delta\, \mathbf{1}(d = \mathbb{W})
$$

The worker's problem has exactly the EGM structure, choice by choice: 
- fix $d$, and then 
- the post-decision state $A = M - c$ determines next period's wealth. 

What is new is the maximum over $d$ in front.

````{tip} Example: the analytic solution with log utility and no interest

With $u(c) = \log c$, $R = 1$ and deterministic income the model solves by hand. Let
$S_k = \sum_{i=0}^{k} \beta^i$. A worker with $t$ periods to go consumes a piecewise
linear function of wealth:

$$
c^\star(M, \mathbb{W}) =
\begin{cases}
M & M \leqslant y/\beta \\[2pt]
(k y + M) / S_k, \; k = 1, \dots, t-1 & \text{on successive intervals of } M \\[2pt]
(k y + M) / S_t, \; k = t, t-1, \dots, 0 & \text{on successive intervals of } M
\end{cases}
$$

- the first row is the credit constrained consumer, who eats everything and waits for the
  next paycheck
- in the second row the consumer works and smooths consumption over the next $k$ periods,
  after which the constraint binds again: $k$ grows with wealth
- in the third row the constraint never binds, and $k$ is the number of periods the
  consumer still works before retiring: it *falls* with wealth, down to $k = 0$,
  retirement now

Each row is linear in $M$, and the rows are stitched together at thresholds
$\overline{M}$ that are not in closed form. The slope of the policy and its intercept
change at every threshold: those are the kinks and the jumps we are after. The
[check below](#dcegm-analytic-check) compares the numerical solution with this formula.
````

The analytic consumption rule of a worker:

```{figure} _static/img/egm/dcegm_analytic.png
:width: 75%
:align: center

Analytic consumption of a worker at $T-2$, $T-5$, $T-10$ and $T-19$ against
cash-in-hand $M$ (the vertical axis is consumption), with $u(c) = \log c$, $R = 1$,
$\beta = 0.98$, $y = 20$ and $T = 20$. Consumption equals $M$ up to $y / \beta$, then follows the linear
segments of the formula, and drops at every retirement threshold.
```

:::{div}
:class: discussion

- Think of an economics implication of the drops in consumption, why it happens?
- Does the graph imply that in the data we should expect to see discontinuous drops in consumption?
:::

The model object follows the [architecture of Class 16](16_cons_practice.md): parameters,
grids, quadrature and the primitives in one place, every solver a function of it. Two
details are specific to the discrete choice:

- the grid is on **savings** $A$, as in EGM, ascending and starting at zero
- there is disutility of work $\delta$
- income is paid only while working

```{code-cell} python3
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

class retirement():
    '''Consumption-savings model with a discrete retirement choice (Iskhakov et al. 2017)'''

    def __init__(self, beta=0.95, R=1.05, y=1.0, delta=0.5, T=10, sigma=0.0, sigma_y=0.0,
                 Mbar=20, ngrid=500, nquad=5):
        self.beta, self.R, self.y, self.delta = beta, R, y, delta   # discount, return, wage, disutility of work
        self.T, self.sigma, self.sigma_y = T, sigma, sigma_y  # periods, taste shock scale, income shock std
        self.Mbar, self.ngrid, self.nquad = Mbar, ngrid, nquad
        self.eps = np.finfo(float).eps                        # smallest positive float
        self.A = np.linspace(0, Mbar, ngrid)                  # grid on savings, ascending, starts at zero
        x, w = np.polynomial.hermite.hermgauss(nquad)         # Gauss-Hermite nodes and weights
        self.quadp = y * np.exp(np.sqrt(2) * sigma_y * x - sigma_y**2 / 2)   # income nodes, mean y
        self.quadw = w / np.sqrt(np.pi)                       # weights, sum to one

    def u(self, c, d):
        return np.log(c) - self.delta * d                     # utility, d = 1 working, d = 0 retired

    def mu(self, c):
        return 1 / c                                          # marginal utility

    def imu(self, x):
        return 1 / x                                          # inverse marginal utility

    def next_wealth(self, A, d, y):
        return self.R * A + d * y                             # budget: income only while working

def interp(x, xp, fp):
    '''Linear interpolation, extrapolated linearly beyond the last grid point'''
    slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return np.where(x > xp[-1], fp[-1] + slope * (x - xp[-1]), np.interp(x, xp, fp))
```

The defaults used in all the figures below, unless stated otherwise:

| | |
| :-- | :-- |
| discount factor, gross return | $\beta = 0.95$, $R = 1.05$ |
| wage, disutility of work | $y = 1$, $\delta = 0.5$ |
| utility | $\log c$, the limit $\rho \to 0$ |
| horizon | $T = 10$ periods, $t = 0, \dots, 9$ |
| savings grid | 500 equally spaced points on $[0, 20]$ |
| income shocks | none, $\sigma_y = 0$; log-normal with 5 Gauss–Hermite nodes otherwise |
| taste shocks | none, $\sigma = 0$ |

Periods are indexed from zero, as Python does, so the last period is $T - 1$ and the
period before it $T - 2$.

(dcegm-choice-specific)=
## Choice-specific value functions

The plan of {cite:t}`egm` in its first version:

1. an EGM step for each discrete choice $d$ and every discrete state $st$
2. this gives the $d$-specific consumption rules *and value functions*
3. compare the $d$-specific value functions to find the optimal switching points, that is
   compute their **upper envelope**
4. reconstruct the overall consumption rule and value function from the switching points

Nothing here needs a root finder, and the credit constraint keeps its EGM treatment. The
price is that we must now compute **value functions**, which time iterations
and EGM never needed, and compute an **upper envelope**.

So each EGM step must return, for its discrete choice $d$, the endogenous points
$(M_j, c_j)$ *and* the value at those points,

$$
v_t(M_j, d) = u(c_j, d) + \beta\, \mathbb{E} V_{t+1}\big(R A_j + y\,\mathbf{1}(d = \mathbb{W})\big),
$$

where $V_{t+1}$ is the value of the state the choice $d$ leads to.

So, we have to keep and interpolate the value function as well as the policy function.

### The credit constrained region

Credit constraints are handled so well by EGM because it never has to evaluate the
utility of nearly zero consumption. Instead it "connects the dots" $(0, 0)$ and
$(M^{cc}_t, M^{cc}_t)$, where $M^{cc}_t$ is the wealth level that corresponds to zero
savings, $A_t = 0$: for every $M < M^{cc}_{t}$ the optimal consumption is $c^\star_t = M$.

Once value functions are needed, the constrained region needs its own formula:

- for each $d_t$ the EGM loop starts from $A = 0$ and returns $M^{cc}_{t, d_t}$
- for $M < M^{cc}_{t, d_t}$ the value function has the **analytic form**
  $V_t^{d_t}(M) = u(M, d_t) + \beta\, EV^{0}_{t+1}(d_t)$, where $EV^{0}_{t+1}(d_t)$ is the
  expected value of ending period $t$ with $A_t = 0$, a number the EGM step computes anyway
- with additively separable utility **(AS)**,
  $V_t^{d_t}(M) = u(M) + v(d_t) + \beta\, EV^0_{t+1}(d_t)$: the choice-specific value
  functions differ by a constant for $M < \min_{d_t} M^{cc}_{t, d_t}$, so they **do not
  intersect** there, no switching point has to be searched for at nearly zero wealth, and
  the choice probabilities do not change

The solution of one period and one choice is therefore a set of endogenous points plus
one number, $EV^0$. The two functions below evaluate the policy and the value anywhere:
connecting the dots below the first endogenous point, interpolating above it.

```{code-cell} python3
def terminal(m, d):
    '''Last period: everything is consumed, no endogenous points, the whole grid is the constrained region'''
    return {'M': np.array([]), 'c': np.array([]), 'v': np.array([]), 'ev0': 0.0}

def policy(m, s, M):
    '''Consumption at wealth M from the solution s of one period and choice: c = M below the first endogenous point'''
    M = np.asarray(M, float)
    if s['M'].size == 0: return M                             # last period: consume everything
    return np.where(M < s['M'][0], M, interp(M, s['M'], s['c']))

def value(m, s, d, M):
    '''Choice-specific value at wealth M: analytic below the first endogenous point, interpolated above'''
    M = np.asarray(M, float)
    vcc = m.u(np.maximum(M, m.eps), d) + m.beta * s['ev0']    # constrained region: c = M, A = 0
    if s['M'].size == 0: return vcc
    return np.where(M < s['M'][0], vcc, interp(M, s['M'], s['v']))
```

### The discrete choice

Next period's worker chooses again, so the EGM step for $d = \mathbb{W}$ needs the
*maximum* of next period's two choice-specific values, and the policy of whichever choice
wins. The two functions below return that maximum and the indicator of working.

They are written with a scale $\sigma$ of taste shocks already, the subject of
[the second half of the class](#dcegm-taste-shocks): for $\sigma = 0$ they are the plain
maximum and the plain indicator.

```{code-cell} python3
def logsum(m, vW, vR):
    '''Expected value over the taste shocks: the maximum without shocks, the smoothed logsum with'''
    mx = np.maximum(vW, vR)
    if m.sigma == 0: return mx
    return mx + m.sigma * np.log(np.exp((vW - mx) / m.sigma) + np.exp((vR - mx) / m.sigma))   # de-maxed

def choice_prob(m, vW, vR):
    '''Probability to keep working: an indicator without taste shocks, logit with'''
    if m.sigma == 0: return (vW > vR).astype(float)
    eW, eR = np.exp((vW - np.maximum(vW, vR)) / m.sigma), np.exp((vR - np.maximum(vW, vR)) / m.sigma)
    return eW / (eW + eR)                                     # de-maxed, as in Class 6
```

### The EGM step for one discrete choice

The step is the [EGM step of Class 17](17_egm.md) with two additions: the next period
consumption and value depend on which choice is optimal there, and the value at the
endogenous points is computed alongside the policy.

````{attention} Algorithm (EGM step for a discrete choice)

**Input:** a grid $0 = A_1 < \dots < A_n$ on savings, quadrature nodes and weights
$(y_k, \omega_k)$, the choice $d$, and next period's solution for both choices,
$\big(c_{t+1}(\cdot, d'), v_{t+1}(\cdot, d')\big)$, $d' \in \{\mathbb{R}, \mathbb{W}\}$.

**Steps**, for every grid point $A_j$ and every node $k$:

1. next period wealth: $M'_{jk} = R A_j + y_k\, \mathbf{1}(d = \mathbb{W})$
2. next period values $v_{t+1}(M'_{jk}, d')$ and consumptions $c_{t+1}(M'_{jk}, d')$ of
   both choices, and the choice that wins, $P_{jk} = \mathbf{1}\big(v_{t+1}(M'_{jk}, \mathbb{W}) > v_{t+1}(M'_{jk}, \mathbb{R})\big)$;
   if $d = \mathbb{R}$ next period is a retiree and $P_{jk} = 0$
3. expected marginal utility and expected value over the choice:
   $\mu_{jk} = P_{jk}\, u'\big(c_{t+1}(M'_{jk}, \mathbb{W})\big) + (1 - P_{jk})\, u'\big(c_{t+1}(M'_{jk}, \mathbb{R})\big)$,
   $V'_{jk} = \max_{d'} v_{t+1}(M'_{jk}, d')$
4. quadrature: $\mathbb{E}\mu_j = \sum_k \omega_k \mu_{jk}$, $\mathbb{E}V_j = \sum_k \omega_k V'_{jk}$
5. the inverted Euler equation: $c_j = (u')^{-1}(\beta R\, \mathbb{E}\mu_j)$
6. the endogenous grid point and the value there: $M_j = A_j + c_j$,
   $v_j = u(c_j, d) + \beta\, \mathbb{E}V_j$

**Output:** the points $(M_j, c_j, v_j)$, $j = 1, \dots, n$, and $EV^0 = \mathbb{E}V_1$,
the expected value of saving nothing.
````

```{code-cell} python3
def egm_step(m, d, sol1):
    '''EGM step for choice d (1 work, 0 retire) given the solution sol1 = [retire, work] of next period'''
    M1 = m.next_wealth(m.A[:, None], d, m.quadp[None, :])     # next period wealth: savings x income nodes
    vR = value(m, sol1[0], 0, M1)                             # next period value of a retiree
    cR = np.maximum(policy(m, sol1[0], M1), m.eps)            # next period consumption of a retiree, kept positive
    if d == 1:                                                # still a worker tomorrow: chooses again
        vW = value(m, sol1[1], 1, M1)
        cW = np.maximum(policy(m, sol1[1], M1), m.eps)
        P = choice_prob(m, vW, vR)                            # probability to keep working tomorrow
        mu1 = P * m.mu(cW) + (1 - P) * m.mu(cR)               # expected marginal utility over the choice
        V1 = logsum(m, vW, vR)                                # value before the taste shocks are drawn
    else:                                                     # retired tomorrow: no choice
        mu1, V1 = m.mu(cR), vR
    rhs = m.beta * m.R * (mu1 @ m.quadw)                      # RHS of the Euler equation, by quadrature
    c = m.imu(rhs)                                            # inverted Euler equation
    M = m.A + c                                               # endogenous grid points
    ev = V1 @ m.quadw                                         # expected value of next period at each A
    v = m.u(c, d) + m.beta * ev                               # choice-specific value at the endogenous points
    return M, c, v, ev[0]                                     # ev[0]: expected value of saving nothing
```

Two EGM steps from the terminal period give the last three periods. The figures go
backward one period at a time and show where the kinks come from.

In the last period $T-1$ a worker never works: there is no future income to earn, so
retiring is better by exactly $\delta$ at every wealth level. One period earlier, at
$T-2$, the two choice-specific value functions **intersect**, and the value of a worker
is their upper envelope:

```{code-cell} python3
:tags: [hide-input]

def plot_policy(ax, M, c, **kw):
    '''Plot a function with jumps, such as a policy c(M), as a line, the jumps (slope above 3) as dotted vertical segments'''
    j = np.flatnonzero(np.abs(np.diff(c)) > 3 * np.diff(M))   # the steps across a discontinuity
    line, = ax.plot(np.insert(M, j + 1, np.nan), np.insert(c, j + 1, np.nan), **kw)   # broken at the jumps
    seg = lambda x: np.c_[x[j], x[j + 1], np.full(j.size, np.nan)].ravel()   # one segment per jump
    ax.plot(seg(M), seg(c), ':', color=line.get_color(), lw=line.get_linewidth())

def fold(M, pad=(0.3, 0.3)):
    '''Wealth limits that show the whole fold of the raw EGM points M, with a margin on each side'''
    b = np.flatnonzero(np.diff(M) <= 0)                       # steps where the endogenous grid bends back
    F = M[b[0]:b[-1] + 2]                                     # the fold, from the upper turning point to the lower one
    return F.min() - pad[0], F.max() + pad[1]

def fit(lim, *curves, margin=0.05):
    '''Limits of the vertical axis that show all the curves (x, y) on the wealth interval lim'''
    y = np.concatenate([y[(x >= lim[0]) & (x <= lim[1])] for x, y in curves])
    return y.min() - margin * np.ptp(y), y.max() + margin * np.ptp(y)

m = retirement()
sol = [[terminal(m, d) for d in (0, 1)]]                      # last period, index T-1
for t in range(2):
    sol.insert(0, [dict(zip(('M', 'c', 'v', 'ev0'), egm_step(m, d, sol[0]))) for d in (0, 1)])   # T-2, then T-3, raw EGM points
Mg = np.linspace(0.05, m.Mbar, 1000)                          # dense grid for the plots
vW, vR = value(m, sol[1][1], 1, Mg), value(m, sol[1][0], 0, Mg)   # the two choices at T-2
Mstar = Mg[np.argmax(vW <= vR)]                               # the retirement threshold at T-2: the primary kink
env = np.maximum(vW, vR)                                      # the upper envelope of the two choices
fig, ax = plt.subplots(3, 1, figsize=(9, 12), layout='constrained')
for d, name, col in ((0, 'retire', 'C0'), (1, 'work', 'C2')):
    ax[0].plot(Mg, value(m, sol[2][d], d, Mg), color=col, label=name)
    ax[1].plot(Mg, value(m, sol[1][d], d, Mg), color=col, label=name)
    ax[2].plot(Mg, value(m, sol[1][d], d, Mg), color=col, lw=2, label=name)
ax[2].plot(Mg, env, 'C3--', label='upper envelope')
w = (Mstar - 3, Mstar + 3)                                    # window around the primary kink
ax[0].set(ylim=(-3, 4), xlabel='wealth M', title='Last period T-1: choice-specific values')
ax[1].set(ylim=(-3, 5), xlabel='wealth M', title='T-2: choice-specific values')
ax[2].set(xlim=w, ylim=(np.interp(w[0], Mg, env) - 0.2, np.interp(w[1], Mg, env) + 0.2),
          xlabel='wealth M', title='T-2, zoom: the upper envelope and the primary kink')
for a in ax: a.grid(color='0.9'); a.legend()
plt.show()
```

The worker's consumption at $T-2$ follows the choice on top. One more period back, at
$T-3$, the worker's EGM points no longer line up, magnified in the inset:

```{code-cell} python3
:tags: [hide-input]

c = np.where(vW > vR, policy(m, sol[1][1], Mg), policy(m, sol[1][0], Mg))   # the worker's consumption: the choice on top
M3, v3 = sol[0][1]['M'], sol[0][1]['v']                       # the worker's raw EGM points at T-3, in savings order
vR3 = value(m, sol[0][0], 0, Mg)                              # the retiree's value at T-3
fig, ax = plt.subplots(2, 1, figsize=(9, 8), layout='constrained')
plot_policy(ax[0], Mg, c, color='C2', label='worker: retire or work')
ax[0].plot(Mg, policy(m, sol[1][0], Mg), color='C0', lw=0.8, label='retiree')
ax[0].plot(Mstar, policy(m, sol[1][1], Mstar), 's', mfc='none', color='C2', ms=8)   # the last point of working
ax[1].plot(Mg, vR3, color='C0', label='retire')
ax[1].plot(M3, v3, color='C2', label='work, EGM points in savings order')
w3 = fold(M3, (1.5, 2.5))                                     # from the fold to past the retirement threshold
ins = ax[1].inset_axes([0.5, 0.08, 0.45, 0.42])               # a magnifying glass on the zig-zag
ins.plot(M3, v3, '.-', color='C2', ms=3, lw=0.7)
z = fold(M3)
ins.set(xlim=z, ylim=fit(z, (M3, v3)), xticks=[], yticks=[])
ax[1].indicate_inset_zoom(ins, edgecolor='0.5')
ax[0].set(xlim=(0, 10), ylim=(0, 5), xlabel='wealth M', title='T-2: consumption drops at the primary kink')
ax[1].set(xlim=w3, ylim=fit(w3, (M3, v3), (Mg, vR3)),
          xlabel='wealth M', title="T-3: the worker's value folds back")
for a in ax: a.grid(color='0.9'); a.legend(loc='upper left')
plt.show()
```

So, what is going on?

1. the $d$-specific value functions intersect, due to the trade-off between the income
   from work and its disutility: a poor worker works, a rich one retires
2. the **upper envelope** of the value functions has a **kink** at the intersection, and
   the combined consumption function has a **discontinuity** there: the retiree has no
   income to look forward to and consumes less out of the same wealth
3. at $T-3$ the worker's value **folds back** on itself: the kink of $T-2$ propagates back
   through the Euler equation, the subject of the [next section](#dcegm-euler-not-sufficient)

The kink is the **primary kink**, the one that comes from the discrete choice itself.
The first version of the algorithm handles it: compute both choice-specific functions,
find the intersection, and switch.

:::{div}
:class: discussion

- Why is the consumption of a worker who keeps working *higher* than that of a retiree
  at the same wealth, when the worker also pays the disutility $\delta$?
- The worker's value function is the maximum of two concave functions. Where is it not
  concave?
:::

(dcegm-euler-not-sufficient)=
## Why the Euler equation is not enough

One more period back the trouble starts. The worker's EGM step now integrates over
$V_{T-2}(\cdot, \mathbb{W})$, which has a kink, so:

1. the **derivative** of the value function has a discontinuity at the kink, and so does
   next period's consumption, which enters the Euler equation
2. for some values of wealth the Euler equation has **two solutions**: one with savings
   that land below the kink next period, one above
3. the EGM step, which inverts the Euler equation at every point of the savings grid,
   returns *all* of them: the endogenous grid points sorted by savings are no longer sorted
   by wealth, they **bend back** and form a **zig-zag region**

The Euler equation remains a *necessary* condition for the optimal consumption, as shown
by {cite:t}`clausen2020ReverseCalculus` for exactly this kind of non-smooth problem,
so the optimum is among the EGM points. It is no longer *sufficient*: some of the points
are not the optimum.

The figure shows the raw output of the worker's EGM step two periods before the last,
zoomed in on the zig-zag, with the points connected in the order of the savings grid. The
bottom panel plots the Euler residual as a function of consumption at a few wealth levels
in the region, with the overall consumption rule of next period on the right-hand side.
The vertical lines of the same colors mark these wealth levels in the two panels above.

```{code-cell} python3
def euler_residual(m, sol1, M, c):
    '''Euler residual of a worker who keeps working, at wealth M and consumption c, given next period solution'''
    M1 = m.next_wealth(np.asarray(M)[..., None] - np.asarray(c)[..., None], 1, m.quadp)   # ... x nodes
    vW, vR = value(m, sol1[1], 1, M1), value(m, sol1[0], 0, M1)
    P = choice_prob(m, vW, vR)                                # who works next period
    cW, cR = np.maximum(policy(m, sol1[1], M1), m.eps), np.maximum(policy(m, sol1[0], M1), m.eps)
    return m.mu(c) - m.beta * m.R * ((P * m.mu(cW) + (1 - P) * m.mu(cR)) @ m.quadw)
```

```{code-cell} python3
:tags: [hide-input]

M, c, v, ev0 = egm_step(m, 1, sol[1])                         # raw EGM step of the worker, two periods before last
lo, hi = fold(M)                                              # window around the zig-zag
fig, ax = plt.subplots(3, 1, figsize=(9, 12), layout='constrained')
ax[0].plot(M, v, '.-', ms=4, lw=0.7, color='k')
ax[1].plot(M, c, '.-', ms=4, lw=0.7, color='k')
ax[0].set(xlim=(lo, hi), ylim=fit((lo, hi), (M, v)),
          xlabel='wealth M', title='Value at the EGM points, in savings order')
ax[1].set(xlim=(lo, hi), ylim=fit((lo, hi), (M, c)),
          xlabel='wealth M', title='Consumption at the EGM points')
cg = np.linspace(1.6, 2.6, 500)
for k, Mz in enumerate(np.linspace(lo + 0.1, hi - 0.1, 5)):
    plot_policy(ax[2], cg, euler_residual(m, sol[1], Mz, cg), lw=0.8, color=f'C{k}', label=f'M = {Mz:.2f}')
    for a in ax[:2]: a.axvline(Mz, color=f'C{k}', lw=0.8)     # the same wealth level in the value and consumption
ax[2].axhline(0, color='k', lw=0.5)
ax[2].set(ylim=(-0.15, 0.15), xlabel='consumption c', title='Euler residuals: several zeros'); ax[2].legend()
for a in ax: a.grid(color='0.9')
plt.show()
```

What the figure shows:

- the curve of endogenous points goes up, turns back, and goes up again: three monotone
  segments, of which the middle one and parts of the other two are **suboptimal**
- in consumption, the same wealth level has two candidate values, roughly $0.15$ apart
- the Euler residual crosses zero twice for every wealth level in the window

The EGM step has no way of telling which crossing is the optimum: that needs the *value*
at the two points.

:::{div}
:class: discussion

- Time iterations solve the Euler equation in every state with a root finder. Which of the
  two zeros would `brentq` return?
- The kink of $V_{T-2}(\cdot, \mathbb{W})$ sits at one wealth level. Why does the zig-zag
  in period $T-3$ spread over an *interval* of wealth?
:::

(dcegm-upper-envelope)=
## The secondary upper envelope

How to detect the zig-zag regions algorithmically? The theorem from
[Class 17](17_egm.md#egm-corner-solutions) still holds:

````{attention} Theorem (monotonicity of savings)

Under weak regularity conditions on the utility function and the intertemporal budget
constraint, the savings function is weakly increasing: $A_t(M'_t) \geqslant A_t(M''_t)$
for every $M'_t \geqslant M''_t$, for all $t$.
````

Note that the savings function may still have *upward jumps*. The theorem turns the
detection into a sorting problem:

1. sort the exogenous grid over $A$ in **ascending order**
2. then the sequence of endogenous grid points over $M$ has to be in **ascending order
   as well**, as long as the Euler equation is sufficient
3. every time the endogenous grid **bends back**, it is separated into subsets of points
4. calculate the **upper envelope** of the value function over the segments
5. **delete** the suboptimal endogenous points
6. **find and add the kink point** to the endogenous grid

The kink point is where two segments of the value function cross. Consumption is
discontinuous there, so it enters the policy twice: with the consumption of the left
segment and with the consumption of the right one. This is the **secondary kink**, which
comes from the primary kink of the next period propagated through the Euler equation.

````{attention} Algorithm (upper envelope of the endogenous points)

**Input:** the endogenous points $(M_j, c_j, v_j)$, $j = 1, \dots, n$, in the order of
the savings grid, the choice $d$ and $EV^0$.

**Steps:**

1. Split the index sequence $1, \dots, n$ into maximal runs on which $M_j$ is monotone,
   sharing the turning points; flip the descending runs. If there is one run, stop.
2. Add the credit constrained segment, the analytic value $u(M, d) + \beta EV^0$ with
   $c = M$, as one more line; it is only needed at wealth points below $M_1$.
3. Collect all wealth values $x_1 < \dots < x_K$ of all runs. For every run $r$ and every
   $x_k$ *inside the run's interval* let $V_{rk}$ be the linear interpolation of the run's
   values at $x_k$; a run is never evaluated outside its own interval. Let
   $\tau_k = \arg\max_r V_{rk}$ be the run on top at $x_k$.
4. Keep the original point $j$ if $v_j = \max_r V_{rk}$ at $x_k = M_j$; delete it
   otherwise.
5. For every $k$ with $\tau_k \neq \tau_{k+1}$: no run has a point strictly inside
   $[x_k, x_{k+1}]$, so every run that covers it is a straight line there, and two such
   lines cross in closed form from their values at $x_k$ and $x_{k+1}$. Starting from
   $\ell = \tau_k$ at $z = x_k$, find the line that overtakes $\ell$ first after $z$, move
   $z$ to that crossing and $\ell$ to that line, and repeat until $\ell = \tau_{k+1}$. At
   every crossing insert two points: one with the consumption of the line on the left, one
   with that of the line on the right, both with the value at the crossing. If one of the
   two lines is the constrained segment, which is concave rather than straight, refine the
   crossing with a root finder on the analytic value.
6. Sort the surviving and the new points by $M$.

**Output:** the cleaned points $(M_j, c_j, v_j)$, ascending in $M$.
````

In code the runs are index arrays. Each run is interpolated only on the wealth points
that fall inside its own interval, and the envelope is built by letting the runs
overwrite it one at a time where they are higher. The cost is the number of points plus
the number of points in the overlapping regions, which is linear in the grid size. A
matrix of all runs on all points would cost (runs) × (points), and interpolate every run
on hundreds of points where it is not even defined.

```{code-cell} python3
def segments(M):
    '''Indices of the maximal monotone runs of M, each run ascending in M, the turning points shared'''
    up = np.diff(M) > 0                                       # direction of every step
    turn = np.flatnonzero(up[1:] != up[:-1]) + 1              # steps where the direction changes
    bounds = np.r_[0, turn, M.size - 1]
    runs = [np.arange(a, b + 1) for a, b in zip(bounds[:-1], bounds[1:])]
    return [r if M[r[-1]] > M[r[0]] else r[::-1] for r in runs]   # flip the descending runs
```

The lines: the credit constrained curve as line 0, then one line per run, each on the
slice of wealth points it covers.

```{code-cell} python3
def envelope_lines(m, d, M, c, v, ev0, runs):
    '''Every run on the wealth points inside its own interval, the envelope, and the line on top at every point'''
    x = np.sort(M, kind='stable')                             # all wealth points; a stable sort merges the sorted runs
    x = x[np.r_[True, np.diff(x) > 0]]                        # drop the duplicates: the shared turning points
    ncc = np.searchsorted(x, M[0])                            # points below the first endogenous point, if any
    lines = [(0, ncc, m.u(x[:ncc], d) + m.beta * ev0, x[:ncc])]   # line 0: credit constrained, c = M, analytic value
    for r in runs:
        lo, hi = np.searchsorted(x, M[r[0]]), np.searchsorted(x, M[r[-1]], 'right')   # x-points inside the run
        lines.append((lo, hi, np.interp(x[lo:hi], M[r], v[r]), np.interp(x[lo:hi], M[r], c[r])))
    vmax, top = np.full(x.size, -np.inf), np.full(x.size, -1) # the envelope and the line on top at every point
    for i, (lo, hi, V, C) in enumerate(lines):
        better = V > vmax[lo:hi]                              # where this line beats the envelope so far
        vmax[lo:hi] = np.where(better, V, vmax[lo:hi])
        top[lo:hi] = np.where(better, i, top[lo:hi])
    return x, lines, vmax, top
```

The kink: two straight lines on one interval cross in closed form; only the constrained
curve, which is concave, needs the root finder.

```{code-cell} python3
def crossing(m, d, ev0, x, lines, a, b, j):
    '''Kink where line b overtakes line a on [x[j], x[j+1]]: wealth, value, consumption left and right'''
    (lo0, hi0, V0, C0), (lo1, hi1, V1, C1) = lines[a], lines[b]
    if not (lo0 <= j < hi0 - 1 and lo1 <= j < hi1 - 1): return None   # a line ends or starts here: no crossing
    f0, f1 = V1[j - lo1] - V0[j - lo0], V1[j + 1 - lo1] - V0[j + 1 - lo0]   # difference of the lines at both ends
    if f0 * f1 >= 0: return None                              # no sign change: the lines share the point
    at = lambda A, lo, w: A[j - lo] + w * (A[j + 1 - lo] - A[j - lo])   # a line evaluated at x[j] + w (x[j+1] - x[j])
    w = f0 / (f0 - f1)                                        # both lines straight on [x[j], x[j+1]]: closed form
    z, vz = x[j] + w * (x[j + 1] - x[j]), at(V0, lo0, w)       # the kink and the value there
    if a == 0 or b == 0:                                      # the constrained curve is concave: refine by a root finder
        lo, V = (lo1, V1) if a == 0 else (lo0, V0)            # the run that crosses it
        g = lambda z: m.u(z, d) + m.beta * ev0 - at(V, lo, (z - x[j]) / (x[j + 1] - x[j]))
        z = brentq(g, x[j], x[j + 1])
        w, vz = (z - x[j]) / (x[j + 1] - x[j]), m.u(z, d) + m.beta * ev0
    return z, vz, at(C0, lo0, w), at(C1, lo1, w)              # consumption jumps at the kink: left line, right line
```

The envelope itself: keep the original points that sit on top, add the kinks, sort.
Usually the line on top at $x_{k+1}$ overtakes the one on top at $x_k$ directly, but a third
line can sit above both in the middle of the interval, so the code walks through the
crossings one at a time.

```{code-cell} python3
def upper_envelope(m, d, M, c, v, ev0):
    '''Delete the endogenous points below the upper envelope of the curve (M, v), insert the kinks'''
    runs = segments(M)
    if len(runs) == 1: return M, c, v                         # no bend-back: nothing to do
    x, lines, vmax, top = envelope_lines(m, d, M, c, v, ev0, runs)
    vM = vmax[np.searchsorted(x, M)]                          # the envelope at the original points
    keep = v >= vM - 1e-10 * (1 + np.abs(vM))                 # keep the points on the envelope only
    Mn, cn, vn = [M[keep]], [c[keep]], [v[keep]]
    for j in np.flatnonzero(np.diff(top)):                    # the top line changes between x[j] and x[j+1]
        a, z = top[j], x[j]                                   # walk from the line on top at x[j] ...
        while a != top[j + 1]:                                # ... to the line on top at x[j+1]
            first = None                                      # the first line to overtake a after z
            for b in range(len(lines)):
                k = None if b == a else crossing(m, d, ev0, x, lines, a, b, j)
                if k is not None and k[0] > z and (first is None or k[0] < first[0][0]): first = (k, b)
            if first is None: break                           # the lines only share a point here
            (z, vz, c0, c1), a = first
            Mn.append([z - 1e3 * m.eps, z]); vn.append([vz, vz]); cn.append([c0, c1])   # the kink, twice
    Mn, cn, vn = map(np.concatenate, (Mn, cn, vn))
    i = np.argsort(Mn, kind='stable')                         # nearly sorted already: linear time
    return Mn[i], cn[i], vn[i]
```

The tolerance in the `keep` line matters: a point of a suboptimal segment can sit
$10^{-5}$ below the envelope, and `np.isclose` with its default tolerance would keep it, a
stray point that puts a spurious jump into the policy. Linear interpolation reproduces a
point's own value to machine precision, so the tolerance can be tight. The crossing
is computed from the values of the two lines at the ends of one interval, so it costs
nothing beyond the interpolation already done.

```{code-cell} python3
:tags: [hide-input]

Mc, cc, vc = upper_envelope(m, 1, M, c, v, ev0)             # the raw points of the figure above, cleaned
fig, ax = plt.subplots(2, 1, figsize=(9, 8), layout='constrained')
ax[0].plot(M, v, '.-', ms=4, lw=0.7, color='0.6', label='EGM points, in savings order')
ax[0].plot(Mc, vc, 'o-', ms=3, lw=1, color='C3', label='after the upper envelope')
ax[1].plot(M, c, '.-', ms=4, lw=0.7, color='0.6', label='EGM points, in savings order')
plot_policy(ax[1], Mc, cc, marker='o', ms=3, lw=1, color='C3', label='after the upper envelope')
ax[0].set(xlim=(lo, hi), ylim=fit((lo, hi), (M, v), (Mc, vc)),
          xlabel='wealth M', title='Value: the upper envelope and the secondary kink')
ax[1].set(xlim=(lo, hi), ylim=fit((lo, hi), (M, c), (Mc, cc)),
          xlabel='wealth M', title='Consumption: the jump at the secondary kink')
for a in ax: a.grid(color='0.9'); a.legend()
print(f'{M.size} endogenous points in, {Mc.size} out: {np.sum(~np.isin(Mc, M))} inserted at the kink')
plt.show()
```

The middle segment and the tails of the other two are gone, the value function has a
kink where the two surviving segments cross, and consumption jumps down at that point:
the consumer who saves a little more lands above next period's retirement threshold, and
behaves differently from there on.

The whole of period $T-3$ after the envelope: the worker's value with both kinks, and
the worker's consumption, which now drops twice:

```{code-cell} python3
:tags: [hide-input]

s3 = {'M': Mc, 'c': cc, 'v': vc, 'ev0': ev0}                  # the worker at T-3 after the upper envelope
vW3 = value(m, s3, 1, Mg)
kinks = np.unique(np.round(Mc[~np.isin(Mc, M)], 6))           # the secondary kinks: the points the envelope inserted
thr = Mg[np.argmax((vW3 <= vR3) & (Mg > kinks.max()))]        # the retirement threshold at T-3: the primary kink
c3 = np.where(vW3 > vR3, policy(m, s3, Mg), policy(m, sol[0][0], Mg))   # the worker's consumption: the choice on top
fig, ax = plt.subplots(2, 1, figsize=(9, 8), layout='constrained')
ax[0].plot(Mg, vR3, color='C0', label='retire')
ax[0].plot(Mg, vW3, color='C2', label='work, after the upper envelope')
plot_policy(ax[1], Mg, c3, color='C2', label='worker: retire or work')
for a in ax:
    for k in kinks: a.axvline(k, color='C3', ls='--', lw=1.5, label='secondary kink' if k == kinks[0] else None)
    a.axvline(thr, color='k', lw=1, label='retirement threshold')
    a.grid(color='0.9'); a.legend(loc='upper left')
ax[0].set(xlim=(0, 10), ylim=(-1, 3.5), xlabel='wealth M', title='T-3: value, the primary and the secondary kink')
ax[1].set(xlim=(0, 10), ylim=(0, 5), xlabel='wealth M', title='T-3: consumption drops at both kinks')
print(f'secondary kink at M = {", ".join(f"{k:.2f}" for k in kinks)}, retirement threshold at M = {thr:.2f}')
plt.show()
```

- The **secondary kink** at $M \approx 4.05$ is the image of next period's retirement
  threshold. Savings from just below it land in the region where a worker still works at
  $T-2$, savings from just above it in the region where a worker retires.
- The **primary kink** at $M \approx 4.98$ is the retirement threshold of $T-3$ itself,
  where the worker's and the retiree's values cross.
- Consumption drops by about $0.3$ at each. One period further back each kink leaves its
  own image, and the kinks accumulate, as the [full solution](#dcegm-full-algorithm) shows.

Where it breaks:

- a zig-zag region **narrower than the grid spacing** is missed: no bend-back, no
  envelope, and the kink is smeared between two grid points as on a fixed grid
- when the discrete choices are *very* different the $d$-specific endogenous grids may
  not overlap, and the comparison of the value functions needs extrapolation

An alternative that does not rely on the monotonicity of savings is the **fast
upper-envelope scan** of {cite:t}`dobrescu2022FastUpperEnvelope`:

- sort the endogenous points by wealth, and choose a jump threshold for consumption
- walking along the sorted points, delete a point as inferior whenever the value
  decreases while consumption jumps at the same time

It generalizes to higher dimensions, at the price of a pre-set threshold that can
mis-classify points. FUES is the default secondary envelope of the `dcegm` package of
Maximilian Blesch and coauthors (see the references), with the fixed-grid envelope of
{cite:t}`druedahl2017GeneralEndogenousGridb` as the alternative; the implementation above
is the original DC-EGM envelope of {cite:t}`egm`, made local.

:::{div}
:class: discussion

- Which runs can be on top in step 3 of the algorithm, and why is the credit constrained
  segment included at all?
- Count the operations of `envelope_lines` as a function of the number of points $n$ and
  the number of runs. Which part is intrinsic to the problem, and which would a matrix of
  all runs on all points add?
- The kink point is inserted twice in the policy, $10^{-13}$ apart. What would a single
  point do to the linear interpolation of consumption?
:::

(dcegm-full-algorithm)=
## The full algorithm without taste shocks

````{attention} Algorithm (DC-EGM, backward induction)

**Input:** the grids, nodes and weights of the model, and the horizon $T$.

**Iterate** backward for $t = T-2, T-3, \dots, 0$, starting from the terminal solution
$c_{T-1}(M, d) = M$, $v_{T-1}(M, d) = u(M, d)$:

1. for each discrete choice $d \in \{\mathbb{R}, \mathbb{W}\}$: take the
   [EGM step](#dcegm-choice-specific) with next period's solution, to obtain the
   $d$-specific consumption rule and value function on the endogenous grid
2. for the worker's choice $d = \mathbb{W}$: compute the **secondary upper envelope**
   over the zig-zag regions of the $d$-specific value function, and update the
   corresponding consumption rule
3. compare the $d$-specific value functions to find the optimal switching points, the
   **primary upper envelope**
4. reconstruct the overall consumption rule and value function of the worker from the
   switching points

**Output:** $c_t(M, d)$ and $v_t(M, d)$ for all $t$ and $d$, from which the worker's policy
is $d_t(M) = \arg\max_d v_t(M, d)$ and $c_t(M) = c_t(M, d_t(M))$.
````

Steps 1 and 2 are the solver. Steps 3 and 4 are done *on demand*, by comparing the two
interpolated value functions at any wealth level: the function `worker` below is the
primary envelope. The retiree's curve never bends back, because the retiree's problem is
concave, so only the worker's step runs the envelope.

```{code-cell} python3
def solve_dcegm(m, callback=None):
    '''Backward induction with DC-EGM: returns sol[t][d], t = 0..T-1, d = 0 retire, 1 work'''
    sol = [[None, None] for t in range(m.T)]
    sol[-1] = [terminal(m, d) for d in (0, 1)]
    for t in range(m.T - 2, -1, -1):
        for d in (0, 1):
            M, c, v, ev0 = egm_step(m, d, sol[t + 1])
            if d == 1:                                        # only the worker's curve can bend back
                M, c, v = upper_envelope(m, d, M, c, v, ev0)
            sol[t][d] = {'M': M, 'c': c, 'v': v, 'ev0': ev0}
            assert np.all(np.diff(M) > 0)                     # the cleaned endogenous grid is sorted
            if callback: callback(t=t, d=d, sol=sol[t][d])
    return sol

def worker(m, sol_t, M):
    '''Policy of a worker in one period: consumption, the probability to work, and the value'''
    vW, vR = value(m, sol_t[1], 1, M), value(m, sol_t[0], 0, M)
    P = choice_prob(m, vW, vR)
    c = np.where(P > 0.5, policy(m, sol_t[1], M), policy(m, sol_t[0], M))   # the more likely choice
    return c, P, logsum(m, vW, vR)
```

```{code-cell} python3
:tags: [hide-input]

t0 = time.perf_counter()
sol = solve_dcegm(m)
print(f'DC-EGM solved {m.T} periods on {m.ngrid} grid points in {time.perf_counter() - t0:.3f} seconds')
def plot_solution(m, sol, note='', panels=(0, 2, 1)):
    '''Worker's consumption (0), value (2) and decision to keep working (1) in four periods, as separate figures'''
    periods = ((8, 'tab:green'), (6, 'tab:red'), (4, 'tab:blue'), (2, 'tab:purple'))   # four periods
    for i, title, ylim in ((0, "Worker's consumption", (0, 8)), (2, "Worker's value function", (-4, 10)),
                           (1, 'Decision to keep working', (-0.05, 1.05))):   # worker() returns c, P, V
        if i not in panels: continue
        fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
        for t, col in periods:
            y = worker(m, sol[t], Mg)[i]
            if i == 2: ax.plot(Mg, y, color=col, lw=1.5, label=f't = {t} (T-{m.T - t})')   # the value has no jumps
            else: plot_policy(ax, Mg, y, color=col, lw=1.5, label=f't = {t} (T-{m.T - t})')
        ax.set(ylim=ylim, xlabel='wealth M', title=title + note)
        ax.legend(); ax.grid(color='0.9')
        plt.show()

plot_solution(m, sol)
```

Properties of the full solution, all of them attributes of the model itself and not of
the method:

1. value functions are non-concave and have **kinks**, one primary kink per period at the
   retirement threshold, which moves out as the horizon grows
2. consumption functions have **discontinuities**: the drop at the retirement threshold
   and, to the left of it, the smaller drops at the secondary kinks
3. the discontinuities and kinks **propagate** through time and **accumulate**: the policy
   in an early period carries the image of every later threshold, a snowball of jumps

Any solution method has to deal with these complexities. A solver on a fixed grid sees
them as noise.

(dcegm-analytic-check)=
### Check against the analytic solution

With $R = 1$ the [analytic policy](#retirement-model) is a list of candidate lines. The
check evaluates the numerical policy on a dense grid and reports the distance to the
nearest candidate.

```{code-cell} python3
ma = retirement(R=1.0)                                        # log utility, no interest: analytic case
sa = solve_dcegm(ma)
S = lambda k: sum(ma.beta**i for i in range(k + 1))           # S_k = 1 + beta + ... + beta^k
for t in (0, 4, 7, 8):
    tg = ma.T - 1 - t                                         # periods to go
    c, P, V = worker(ma, sa[t], Mg)
    lines = [Mg] + [(k * ma.y + Mg) / S(k) for k in range(1, tg)] + [(k * ma.y + Mg) / S(tg) for k in range(tg + 1)]
    dev = np.min(np.abs(np.array(lines) - c), axis=0)         # distance to the nearest analytic line
    print(f'{tg} periods to go: {np.mean(dev < 1e-10) * 100:5.1f}% of the grid points within 1e-10 of the '
          f'analytic policy, max deviation {dev.max():.1e} at M = {Mg[dev.argmax()]:.2f}')
```

DC-EGM matches the analytic solution to machine precision on most of the grid, and on
all of it one period before the end: the retirement thresholds and the jumps are placed
exactly.

The deviations, at most $2 \cdot 10^{-3}$ and starting at $M \approx 1.15$, sit at the
kinks of the second row of the analytic formula and at their images in earlier periods.
A kink that comes from the credit constraint binding *two* periods ahead is not an
endogenous grid point, so linear interpolation smears it like on any fixed grid, and the
error spreads the longer the horizon.

:::{div}
:class: discussion

- The run time is a few milliseconds for ten periods. Which part of the algorithm is not
  vectorized, and when would it dominate?
- Why does the accumulation of kinks make a *finer* grid more necessary in period 0 than
  in period $T-2$?
:::

(dcegm-random-income)=
## Random income

Random shocks do help. With a log-normal wage the expectation in the Euler equation
averages next period's consumption rule over the income nodes, and the jump at the
retirement threshold is spread over a range of savings. However:

- shocks smooth out **secondary** kinks **only**: the primary kinks, the switching between
  the discrete options, **remain**
- they may not smooth out all of them: a continuous but sharp decline in consumption at
  $t$ can still produce a discontinuity or a kink at $t - 1$
- the expectations in the Euler equation are now taken over **discontinuous functions**,
  and quadrature assumes a smooth integrand: more kinks appear from sloppy computation,
  and a careful implementation integrates over the continuous intervals separately

The figures go through period $T-4$ step by step, the period that has two secondary drops
and the primary one without shocks, as in the [full solution](#dcegm-full-algorithm).
All of them use 200 quadrature nodes, for the reason the last figure shows.

```{code-cell} python3
:tags: [hide-input]

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2.colors)   # a darker palette for the rest of the page
t = m.T - 4                                                   # the period of the figures
zoom = dict(xlim=(3, 11), ylim=(1.6, 2.3), xlabel='wealth M', ylabel='consumption')

def solve_income(sy, nq=200):
    '''Solve the model with log-normal income of scale sy on nq quadrature nodes'''
    ms = retirement(sigma_y=sy, nquad=nq)
    return ms, solve_dcegm(ms)

def drops(ms, sol_t):
    '''Worker's consumption in one period, the last point of working, and the steepest decline below it'''
    c, P, V = worker(ms, sol_t, Mg)
    i = np.argmax(P < 0.5) - 1                                # the last grid point where the worker works
    slope = np.diff(c[:i + 1]) / np.diff(Mg[:i + 1])
    return c, i, Mg[np.argmin(slope)], slope.min()            # where consumption falls fastest, and how fast

ms, ss = solve_income(0.12)
c, i, k, s = drops(ms, ss[t])
fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
plot_policy(ax, Mg, c, color='C0', lw=1.5)
ax.plot(Mg[i], c[i], 's', mfc='none', color='C0', ms=8)       # the last point of working
ax.set(**zoom, title=f'T-4: optimal consumption with σ_y = {ms.sigma_y}')
ax.grid(color='0.9')
plt.show()
```

With $\sigma_y = 0.12$ the first secondary drop, at $M \approx 4.8$, has become a smooth
dip. The second one, at $M \approx 5.7$, is still nearly vertical: it is the image of next
period's retirement threshold. The primary drop at the threshold itself,
$M \approx 6.6$, is as sharp as without shocks.

The same shocks in the two periods before:

```{code-cell} python3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
for tt, col in ((t - 2, 'C2'), (t - 1, 'C1'), (t, 'C0')):
    c, i, k, s = drops(ms, ss[tt])
    plot_policy(ax, Mg, c, color=col, lw=1.5, label=f'T-{ms.T - tt}')
    ax.plot(Mg[i], c[i], 's', mfc='none', color=col, ms=8, label='retirement threshold')
    ax.axvline(k, color=col, ls='--', lw=1, label='steepest decline')
    print(f'T-{ms.T - tt}: retirement threshold at M = {Mg[i]:.2f}, steepest decline at M = {k:.2f}, slope {s:.1f}')
ax.set(**zoom, title=f'T-6 to T-4: optimal consumption with σ_y = {ms.sigma_y}')
ax.legend(loc='lower right', ncols=3); ax.grid(color='0.9')
plt.show()
```

Every period has one steep decline below its threshold, the image of next period's
threshold, while the images of the older kinks are smoothed out. The images move out
with the thresholds, and the further from the end, the less steep they are.

A larger shock spreads the image of the next threshold over a wider range of wealth:

```{code-cell} python3
:tags: [hide-input]

def income_stage(sy):
    '''Consumption and the choice-specific values in period t with income shocks of scale sy'''
    ms, ss = solve_income(sy)
    c, i, k, s = drops(ms, ss[t])
    fig, ax = plt.subplots(2, 1, figsize=(12, 9.5), layout='constrained')
    plot_policy(ax[0], Mg, c, color='C0', lw=1.5)
    ax[0].plot(Mg[i], c[i], 's', mfc='none', color='C0', ms=8)
    ax[1].plot(Mg, value(ms, ss[t][0], 0, Mg), color='C2', lw=1.5, label='retirement')
    ax[1].plot(Mg, value(ms, ss[t][1], 1, Mg), color='C0', lw=1.5, label='work')
    for a in ax:
        if s < -1: a.axvline(k, color='C3', ls='--', lw=1.5, label='steepest decline')
        a.axvline(Mg[i], color='C3', lw=1.5, label='retirement threshold')
        a.grid(color='0.9')
    ax[0].set(**zoom, title=f'T-4: optimal consumption with σ_y = {sy}')
    ax[1].set(xlim=(0, 12), ylim=(-3, 3.5), xlabel='wealth M', ylabel='value', title=f'T-4: value functions with σ_y = {sy}')
    ax[1].legend(loc='upper left')
    print(f'steepest decline of consumption below the threshold: slope {s:.2f} at M = {k:.2f}')
    plt.show()

income_stage(0.16)
```

With $\sigma_y = 0.16$ consumption is continuous below the threshold, but still falls
steeply at $M \approx 5.6$. The value functions show nothing of it at this scale: the
primary kink, where the two values cross, is the one that stands out.

A little more:

```{code-cell} python3
:tags: [hide-input]

income_stage(0.2)
```

With $\sigma_y = 0.2$ the secondary drops are gone, and the only discontinuity left is the
primary drop at the retirement threshold.

The same $\sigma_y = 0.2$ on the default 5 quadrature nodes:

```{code-cell} python3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
for nq, col in ((5, 'C1'), (200, 'C0')):
    ms, ss = solve_income(0.2, nq)
    c, i, k, s = drops(ms, ss[t])
    plot_policy(ax, Mg, c, color=col, lw=1.5, label=f'{nq} quadrature nodes')
ax.set(**zoom, title='T-4: optimal consumption with σ_y = 0.2')
ax.legend(loc='lower right'); ax.grid(color='0.9')
plt.show()
```

With 5 nodes the smooth dips turn into staircases of small jumps, one per node crossing
next period's threshold. That is the sloppy computation of the last bullet above: the
quadrature rule treats a discontinuous integrand as if it were smooth.

```{note} Quadrature over a discontinuous integrand
Next period's consumption rule $c_{t+1}(M)$ jumps at the retirement threshold and at every
secondary kink, and so does the marginal utility of consumption that the Euler equation
integrates. With savings $A$ and a random wage $\tilde y$ with density $f$ the expectation is

$$
\mathbb{E}\, u'\big(c_{t+1}(R A + \tilde y)\big) = \int u'\big(c_{t+1}(R A + y)\big) f(y)\, dy ,
$$

and the integrand jumps at every wage $y_k = M_k - R A$ that puts next period's wealth on a
discontinuity $M_k$. Gauss–Hermite quadrature is accurate for smooth integrands, not for
steps: the weighted sum jumps whenever a node crosses one of the $y_k$ as $A$ moves along the
grid, and these are the stairs in the figure above.

- **Remedy**: more quadrature nodes. The stairs get smaller and denser, but they never
  disappear, and every node adds to the cost of the expectation at every grid point.
- **Full solution**: split the support of the wage at the points $y_k$, which are known
  from the solution of the next period, and apply quadrature separately on each segment.
  The integrand is smooth on every segment, so a few nodes per segment are enough.
```

The figures of this section take the first route: with 200 nodes the stairs are too small
to see at this scale.

:::{div}
:class: discussion

- Why can an income shock never smooth the primary kink of the *same* period?
- The expectation of a discontinuous function by Gauss–Hermite quadrature is itself a
  discontinuous function of savings. Where exactly are its jumps?
:::

(dcegm-taste-shocks)=
## Extreme value taste shocks

Taste shocks smooth out the **primary kinks** too. Add to each discrete alternative an
additive shock $\sigma \epsilon_d$, i.i.d. extreme value type 1, as in the
[logit of Class 6](6_logit.md#probabilistic-choice):

- the extreme value distribution gives closed form expectations, and is standard in
  empirical applications
- two interchangeable interpretations: **structural**, the shocks are unobserved state
  variables, or **logit smoothing**, a device to streamline the solution
- they work together with the other shocks of the model: taste shocks smooth the primary
  kinks, random income or returns smooth the secondary ones
- complete smoothing is not guaranteed in general: secondary kinks may persist

Re-formulate the worker's problem in terms of **choice-specific value functions**:

$$
V_t(M_t, \mathbb{W}) = \max\left\{
\begin{array}{l}
v_t(M_t, \mathbb{W}, \mathbb{R}) + \sigma \epsilon_{\mathbb{R}} \\
v_t(M_t, \mathbb{W}, \mathbb{W}) + \sigma \epsilon_{\mathbb{W}}
\end{array}
\right\}
$$

$$
\begin{aligned}
v_t(M_t, \mathbb{W}, \mathbb{W}) &= \max_{0 \leqslant c \leqslant M_t} \Big[ u(c) + \beta\, EV_{t+1}\big(\tilde R (M_t - c) + y, \mathbb{W}\big) \Big] \\
v_t(M_t, \mathbb{W}, \mathbb{R}) &= \max_{0 \leqslant c \leqslant M_t} \Big[ u(c) + \beta\, EV_{t+1}\big(\tilde R (M_t - c), \mathbb{R}\big) \Big]
\end{aligned}
$$

The expected value of a worker, before the shocks are drawn, is the **logsum** of
[Class 6](6_logit.md#max-stability):

$$
EV_{t+1}(x, \mathbb{W}) = \sigma \log\Big[ \exp\frac{v_{t+1}(x, \mathbb{W}, \mathbb{W})}{\sigma} + \exp\frac{v_{t+1}(x, \mathbb{W}, \mathbb{R})}{\sigma} \Big]
$$

and the retiree's problem is unchanged,
$V_t(M_t, \mathbb{R}) = \max_{c} [u(c) + \beta EV_{t+1}(\tilde R(M_t - c), \mathbb{R})]$.

### The smoothed Euler equation

Without taste shocks the Euler equation is *discontinuous* in savings, because next
period's consumption is that of either a worker or a retiree:

$$
u'(c_t) = \beta\, \mathbb{E}\Big[ u'\big(c_{t+1}(\mathbb{W}/\mathbb{R})\big)\, \tilde R \Big]
$$

With extreme value taste shocks the Euler equation is **smoothed**: the marginal utility
of next period is averaged over the two choices with the choice probabilities,

$$
u'(c_t) = \beta\, \mathbb{E}\Big[ P_{t+1}(\mathbb{W})\, u'\big(c_{t+1}(\mathbb{W})\big)\, \tilde R + P_{t+1}(\mathbb{R})\, u'\big(c_{t+1}(\mathbb{R})\big)\, \tilde R \Big],
$$

where the probability to keep working is the logit

$$
P_{t+1}(\mathbb{W}) = \frac{\exp\frac{v_{t+1}(M_{t+1}, \mathbb{W}, \mathbb{W})}{\sigma}}
{\exp\frac{v_{t+1}(M_{t+1}, \mathbb{W}, \mathbb{W})}{\sigma} + \exp\frac{v_{t+1}(M_{t+1}, \mathbb{W}, \mathbb{R})}{\sigma}}
$$

This is exactly what `egm_step` computes with `choice_prob` and `logsum` when
$\sigma > 0$: the code of the first half of the class was written for both flavors. The
[de-maxing of Class 6](6_logit.md#demaxing) is essential, because $v/\sigma$ overflows
`exp` for any small $\sigma$.

````{attention} Algorithm (DC-EGM with taste shocks)

**Input:** as [before](#dcegm-full-algorithm), plus the scale $\sigma > 0$.

**Iterate** backward for $t = T-2, \dots, 0$ from the terminal solution:

1. for each discrete choice $d$: the EGM step with next period's choice-specific
   solution, with the expected marginal utility weighted by the choice probabilities and
   the expected value given by the logsum
2. for the worker's choice: the secondary upper envelope over the zig-zag regions, if any
3. *dropped:* compare the $d$-specific value functions to find the optimal switching
   points
4. *dropped:* reconstruct the overall consumption rule and value function

**Output:** the choice-specific $c_t(M, d)$ and $v_t(M, d)$; the policy is the choice
probability $P_t(\mathbb{W} \mid M)$ and the consumption conditional on the choice.
````

With taste shocks DC-EGM becomes **simpler**:

1. the problem is re-formulated in terms of choice-specific value functions
2. the calculation of the *primary* upper envelope is replaced by the logsum
3. expectations are easier to compute, because there are fewer discontinuities
4. more memory is required, to store the choice-specific value functions

How far is the smoothed solution from the exact one? The paper gives a bound.

````{attention} Theorem (approximation with the logit smoother)
(dcegm-logit-smoother)=

Let $\sigma$ be the scale of the type 1 extreme value taste shocks for the discrete
choices in a DC problem with $D$ choices. Then

$$
\big| EV_{\sigma, t}(s) - V_t(s) \big| \leqslant \sigma \Big[ \sum_{j=0}^{T-t} \beta^j \Big] \log(D)
$$
````

This implies that the extreme value perturbed policy functions $c_{\sigma, t}(s, \epsilon)$
and $d_{\sigma, t}(s, \epsilon)$ converge pointwise to $c_t(s)$ and $d_t(s)$, the optimal
continuous and discrete decision rules of the problem without taste shocks, as
$\sigma \to 0$.

This is the **extreme value homotopy**: solve with a comfortable $\sigma$, and reduce it
towards the sharp problem if needed.

```{code-cell} python3
:tags: [hide-input]

fig, ax = plt.subplots(3, 1, figsize=(12, 14), layout='constrained')
Mz = 0.5 * (lo + hi)                                          # a wealth level inside the zig-zag region
for s in (0, 0.01, 0.05, 0.1):
    ms = retirement(sigma=s)
    ss = solve_dcegm(ms)
    c, P, V = worker(ms, ss[t], Mg)
    plot_policy(ax[0], Mg, policy(ms, ss[t][1], Mg), lw=1, label=f'σ = {s}')   # consumption conditional on working
    plot_policy(ax[1], Mg, P, lw=1)
    plot_policy(ax[2], cg, euler_residual(ms, ss[m.T - 2], Mz, cg), lw=1)
ax[0].set(xlim=(2, 9), ylim=(1.3, 2.6), xlabel='wealth M', title=f'Consumption of a worker who keeps working, t = {t}')
ax[1].set(xlabel='wealth M', title='Probability to keep working')
ax[2].axhline(0, color='k', lw=0.5)
ax[2].set(ylim=(-0.15, 0.15), xlabel='consumption c', title=f'Euler residual at M = {Mz:.2f}, t = T-3')
ax[0].legend()
for a in ax: a.grid(color='0.9')
plt.show()
```

- **Consumption**: the top panel is the choice-specific policy $c_t(M, \mathbb{W})$,
  because with taste shocks the overall consumption is a lottery between the two
  choice-specific ones. Already with $\sigma = 0.01$ the secondary jumps turn into smooth
  dips, and with $\sigma = 0.1$ the policy is smooth everywhere
- **Choice probability**: the step at the retirement threshold becomes a logit curve,
  wider the larger $\sigma$; with $\sigma = 0.1$ a worker with twice the threshold
  wealth still works with probability close to $0.1$
- **Euler residual**: the two zeros of the discontinuous residual merge into one zero of
  a continuous function; the zig-zag never forms, and the upper envelope has nothing to do

Where it breaks: $\sigma$ is a structural parameter if the shocks are interpreted as
unobserved states, and a numerical one otherwise. A large $\sigma$ changes the economics
of the model, a small one brings the kinks back, and the bound of the theorem is the
only guide to how far a given $\sigma$ is from the sharp problem.

:::{div}
:class: discussion

- Taste shocks make the EGM step *simpler* and the solution *better behaved*, and the
  paper still presents the version without them in full. Why?
- Which parts of the course notation does $\sigma$ collide with, and which object in the
  Rust model of [Class 9](9_zurcher.md) plays its role?
:::

(dcegm-taste-full-solution)=
### The full solution with taste shocks

The figures of the [full algorithm](#dcegm-full-algorithm) once more, with the same four
periods and settings, for a small and a normal scale of the taste shocks. Consumption is
still that of the more likely choice, so the drop where working and retiring are equally
likely stays by construction.

```{code-cell} python3
:tags: [hide-input]

ms = retirement(sigma=0.01)                                   # a small scale of the taste shocks
plot_solution(ms, solve_dcegm(ms), ', σ = 0.01')
```

With $\sigma = 0.01$ the solution is almost the one without shocks: the retirement
thresholds stay at $M \approx 3.25$, $6.6$, $9.62$ and $12.35$, and the decision to keep
working is still a step. The secondary drop next to each threshold has become a smooth dip,
while the older ones further to the left are still jumps: two at $t = 2$ instead of six.

A normal scale:

```{code-cell} python3
:tags: [hide-input]

ms = retirement(sigma=0.1)                                   # a normal scale of the taste shocks
plot_solution(ms, solve_dcegm(ms), ', σ = 0.1')
```

With $\sigma = 0.1$ all secondary drops are gone, and the decision to keep working is a
smooth logit curve. The point where the two choices are equally likely moves out, to
$M \approx 7.16$ at $t = 6$ and $13.73$ at $t = 2$. Side by side with the figures without
shocks, the two scales show the convergence of the [theorem](#dcegm-logit-smoother): the
smaller $\sigma$, the closer the solution is to the one without shocks.

(dcegm-simulation)=
## Simulation

The simulator draws the taste shocks through the choice probability and the income shocks
from the log-normal, and applies the choice-specific policies. Retirement is absorbing,
so the discrete state is carried forward.

```{code-cell} python3
def simulate(m, sol, M0, seed=None):
    '''Simulate the life cycle of workers from initial wealth M0 (array): wealth, consumption, work status'''
    rng = np.random.default_rng(seed)
    N = np.atleast_1d(M0).size
    M, C, D = np.empty((N, m.T)), np.empty((N, m.T)), np.ones((N, m.T), int)
    M[:, 0] = M0
    for t in range(m.T):
        if t > 0: D[:, t] = D[:, t - 1]                       # retirement is absorbing
        w = D[:, t] == 1                                      # those still working decide
        vW, vR = value(m, sol[t][1], 1, M[w, t]), value(m, sol[t][0], 0, M[w, t])
        D[w, t] = rng.random(w.sum()) < choice_prob(m, vW, vR)   # taste shocks, through the logit
        for d in (0, 1):
            i = D[:, t] == d
            C[i, t] = policy(m, sol[t][d], M[i, t])           # consumption conditional on the choice
        if t < m.T - 1:
            y = m.y * np.exp(m.sigma_y * rng.standard_normal(N) - m.sigma_y**2 / 2)   # income draws
            M[:, t + 1] = m.next_wealth(M[:, t] - C[:, t], D[:, t], y)
    return M, C, D
```

```{code-cell} python3
:tags: [hide-input]

ms = retirement(sigma=0.1, sigma_y=0.2, nquad=10)
ss = solve_dcegm(ms)
M0 = np.random.default_rng(1).uniform(2, 20, 1000)            # initial wealth of 1000 workers
Ms, Cs, Ds = simulate(ms, ss, M0, seed=2026)
age = np.where((Ds == 0).any(axis=1), np.argmin(Ds, axis=1), ms.T)   # first period of retirement
fig, ax = plt.subplots(3, 1, figsize=(12, 14), layout='constrained')
for i in range(15):
    ax[0].plot(Ms[i], lw=0.8, color='C0'); ax[1].plot(Cs[i], lw=0.8, color='C1')
    if age[i] < ms.T:
        ax[0].plot(age[i], Ms[i, age[i]], 'ks', ms=4); ax[1].plot(age[i], Cs[i, age[i]], 'ks', ms=4)
ax[2].hist(age, bins=np.arange(ms.T + 2) - 0.5, color='C2')
ax[0].set(xlabel='period', title='Wealth of 15 simulated workers, square: retirement')
ax[1].set(xlabel='period', title='Consumption')
ax[2].set(xlabel='first period of retirement', title='Distribution of the retirement age, 1000 workers')
for a in ax: a.grid(color='0.9')
plt.show()
```

The rich retire in the first period and run their wealth down, the poor work until the
last period, where everyone retires: the retirement age distribution is bimodal, driven
by initial wealth, with the taste shocks filling the middle. The consumption paths are
smooth except for the drop at retirement, the one discontinuity that no shock removes.

:::{div}
:class: discussion

- Which moments of these simulated data would identify $\delta$, and which $\sigma$?
:::

(dcegm-keane-test)=
## Keane's test: make it flat

Keane's test of [Class 16](16_cons_practice.md#simulation-and-keanes-test) carries over
to the retirement model. Without shocks and with $\beta R = 1$ the Euler equation reads
$u'(c_t) = u'(c_{t+1})$ wherever the credit constraint does not bind. The utility is
additively separable in consumption and work, so this holds across the retirement
decision too: every unconstrained path of consumption is flat, before and after retirement.

```{code-cell} python3
:tags: [hide-input]

mk = retirement(R=1 / 0.95)                                   # beta R = 1, no taste shocks, no income shocks
sk = solve_dcegm(mk)
plot_solution(mk, sk, ', βR = 1', panels=(0,))   # consumption only
```

The optimal consumption looks like the one with $R = 1.05$: a primary drop at every
retirement threshold, and secondary drops to the left of it that accumulate going back
in time. Three workers who start with similar wealth:

```{code-cell} python3
:tags: [hide-input]

M0 = np.array([9.0, 10.0, 11.0])                              # three workers, not too far apart
Mk, Ck, Dk = simulate(mk, sk, M0)
age = np.where((Dk == 0).any(axis=1), np.argmin(Dk, axis=1), mk.T)   # first period of retirement
fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
for j, col in enumerate(('tab:green', 'tab:red', 'tab:blue')):
    ax.plot(Ck[j], 'o-', color=col, lw=1.5, ms=4, label=f'M0 = {M0[j]:g}, retires at t = {age[j]}')
    ax.plot(age[j], Ck[j, age[j]], 's', mfc='none', color=col, ms=10)   # the first period of retirement
ax.set(ylim=(1.6, 2.1), xlabel='period', ylabel='consumption',
       title="Keane's test: consumption of three simulated workers, βR = 1")
ax.legend(); ax.grid(color='0.9')
print(f'savings positive on all paths: {(Mk[:, :-1] - Ck[:, :-1] > 0).all()}, '
      f'max |c_t / c_0 - 1| = {np.abs(Ck / Ck[:, :1] - 1).max():.1e}')
plt.show()
```

All three paths are flat to machine precision, so the solver passes the test. The order
of the paths is not the order of the initial wealth: the worker with $M_0 = 10$
consumes less than the one with $M_0 = 9$, because the extra unit of wealth buys a
retirement two periods earlier and costs two periods of wage income.

This is a secondary drop of the consumption function at $t = 0$, seen in simulated
data. Each worker trades consumption for leisure once, at the start of life, and keeps
consumption constant from then on.

The policy behind these paths is anything but simple: a primary drop at every retirement
threshold and a growing number of secondary drops below it, period after period. Still, a
rational worker navigates around all of them perfectly, and consumption is completely
smooth over the life cycle: the jumps of the policy are exactly what keeps consumption
flat when the retirement date changes with wealth.

:::{div}
:class: discussion

- Which part of DC-EGM would a path that is *not* flat point to: the EGM step, the
  upper envelope, or the simulator?
- What happens to the test on a path where the credit constraint binds in some period?
- With income shocks the paths are no longer flat. Which version of the test still applies?
:::

## Conclusion

DC-EGM maintains the accuracy of the original EGM in models with discrete choices. The
upper envelope removes the suboptimal endogenous points and places the kinks exactly, so
the solution matches the [analytic one](#dcegm-analytic-check) to machine precision on
most of the grid and passes Keane's test to machine precision.

The cost of the discrete choice is one envelope scan per period, linear in the number of
points, while the Euler equation is still inverted in closed form: the speed of EGM is
kept as well.



````{note} References and additional resources

(18_dcegm_references)=
- [EGM and DC-EGM slides from DSE2025 summer school](https://github.com/dseconf/DSE2025HKU/blob/main/08_Iskhakov_egm/dcegm_DSE2025hku.pdf)
- 📖 {cite:t}`egm` "The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks"
- 📖 {cite:t}`fella2014GeneralizedEndogenousGrid` "A Generalized Endogenous Grid Method for Non-Smooth and Non-Concave Problems" — identifies the regions where the Euler equation is not sufficient, and uses VFI inside and EGM outside
- 📖 {cite:t}`druedahl2017GeneralEndogenousGridb` "A general endogenous grid method for multi-dimensional models with non-convexities and constraints" — the upper envelope re-interpolated on a regular grid, in several dimensions
- 📖 {cite:t}`clausen2020ReverseCalculus` "Reverse Calculus and nested optimization" — the Euler equation remains necessary in discrete-continuous problems
- 📖 {cite:t}`dobrescu2022FastUpperEnvelope` "Fast Upper-Envelope Scan for Discrete-Continuous Dynamic Programming"
- 📖 {cite:t}`jorgensen2013StructuralEstimation` "Structural estimation of continuous choice models: Evaluating the EGM and MPEC"
- 📖 {cite:t}`carroll2006MethodEndogenousGridpoints` "The method of endogenous gridpoints for solving dynamic stochastic optimization problems"
- 📄 The `dcegm` Python package of Maximilian Blesch and coauthors, a JAX implementation of DC-EGM for large models [github.com/OpenSourceEconomics/dcegm](https://github.com/OpenSourceEconomics/dcegm)
````
