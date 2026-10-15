---
title: 📖 The endogenous gridpoint method
short_title: 📖 EGM
subtitle: Class 17 — Thursday, October 22
downloads:
  - file: 17_egm.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

<span class="tag wip">🚧 🚧 🚧 Not the final version 🚧 🚧 🚧</span>

Every solver of [Class 16](16_cons_practice.md) solves a nonlinear problem in every
state on every iteration, a maximization or a root of the Euler equation. The endogenous
gridpoint method of {cite:t}`carroll2006MethodEndogenousGridpoints` turns the problem
around: fix the choice, and find the state in which it is optimal. For the
consumption-savings model this needs no optimization and no root finding at all.

````{seealso} Key reading

{cite:t}`carroll2006MethodEndogenousGridpoints` "The method of endogenous gridpoints for
solving dynamic stochastic optimization problems", *Economics Letters* 91(3), 312–320.
The method of this class, introduced on the consumption-savings model.
````

````{hint} Running the code for this lecture
:class: dropdown

The code for this class is in the folder `session17-oct22/` of the course **code
repository**. It reuses the model and the solvers of Class 16, `deaton.py` in
`session16-oct20/`, so keep both folders side by side.

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

## The idea on cake eating

Take the [cake eating problem](15_cake_eating.md#cake-eating). The first order condition
for $c_t$ and the envelope theorem,

$$
u'(c_t) = \beta V'(W_{t+1}),
\qquad
V'(W_t) = \beta V'(W_{t+1}),
$$

combine into $V'(W_t) = u'(c_t)$ at every $t$, and into the **Euler equation**

$$
\frac{1}{c_t} = \frac{\beta}{c_{t+1}}
\quad\Leftrightarrow\quad
c_t = \frac{c_{t+1}}{\beta}
$$

Read it backwards: given what is left for the future, $A = W - c$, and the policy for
next period, the Euler equation gives today's consumption *in closed form*, and the
budget gives the cake size at which that consumption is optimal, $W = A + c$.

Value function iterations and time iterations fix a grid on the state $W$ and *search*
for the optimal $c$ at every point. The **endogenous gridpoint method (EGM)** fixes a
grid on the post-decision state $A$ instead, computes $c$ directly, and lets the grid on
the state come out of the computation, *endogenously*:

```
Input: grid A_1 < ... < A_n on the post-decision state, initial policy c_0(W) = W
Algorithm:
  for i = 1, 2, ...
    for every A_j:
      1. next period cake: W'_j = A_j
      2. next period consumption from the previous policy: c'_j = c_{i-1}(W'_j)
      3. this period consumption from the inverted Euler equation: c_j = c'_j / beta
      4. the endogenous grid point: W_j = A_j + c_j
    c_i = interpolation of the points (W_j, c_j)
    stop when c_i and c_{i-1} are within tol
Output: the policy function c(W)
```

There is no maximization and no root finding, only function evaluations. And the Euler
equation holds *exactly* at the endogenous grid points, so the method is as accurate as
time iterations at a fraction of the cost.

:::{div}
:class: discussion

- The endogenous grid points $W_j$ change from one iteration to the next. Why is that
  not a problem for the algorithm?
- Which step of the algorithm would need a root finder if the utility function were not
  $\log(c)$?
:::

## EGM for the consumption-savings model

In the model of {cite:t}`deaton1991SavingLiquidityConstraints` the
[Euler equation](15_cake_eating.md#the-euler-equation) holds with an expectation over
income, and the post-decision state is savings $A = M - c$:

$$
u'\big(c(M)\big) = \beta R\, \mathbb{E}_{y}\, u'\big(c(RA + \tilde{y})\big)
$$

Savings $A$ determine everything on the right-hand side: next period wealth
$M' = RA + \tilde y$ for every income draw and, through next period's policy, the
expected marginal utility. So for a given policy $c(\cdot)$, every value of $A$ gives
one point of the updated policy, as a curve parameterized by $A$:

$$
\begin{cases}
c = (u')^{-1}\Big( \beta R\, \mathbb{E}_{y}\, u'\big(c(RA + \tilde{y})\big) \Big) \\[4pt]
M = A + c
\end{cases}
$$

With $u(c) = \log c$ the inverse marginal utility is $(u')^{-1}(x) = 1/x$, and with the
[quadrature](16_cons_practice.md#expectations-by-quadrature) the expectation is a weighted
sum over the income nodes.

````{attention} Definition

The **EGM step** maps a grid $A_1 < \dots < A_n$ of savings and a policy $c(\cdot)$ into
the points $(M_j, c_j)$ of the updated policy:

1. next period wealth in every income node: $M'_{jk} = R A_j + y_k$
2. next period consumption from the current policy: $c'_{jk} = c(M'_{jk})$
3. the expected marginal utility, by quadrature: $\mathbb{E}u'_j = \sum_k \omega_k\, u'(c'_{jk})$
4. current consumption from the inverted Euler equation: $c_j = (u')^{-1}(\beta R\, \mathbb{E}u'_j)$
5. the endogenous grid point: $M_j = A_j + c_j$
````

The EGM iterations repeat the step, interpolating the points $(M_j, c_j)$ into the policy
for the next step, until the policy stops changing. The figure shows the first
iterations on a grid of only five points, starting from the policy $c_0(M) = M$, defined
by two points: the consumer eats everything.

```{code-cell} python3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

def lin(x, xp, fp):
    '''Linear interpolation, extrapolated linearly beyond the last point'''
    slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return np.where(x > xp[-1], fp[-1] + slope * (x - xp[-1]), np.interp(x, xp, fp))

def egm_iterations(beta, R, y, K, A=np.linspace(0, 10, 5), origin=False, title=''):
    '''Plot the first K iterations of EGM with log utility and deterministic income y'''
    M0, c0 = np.array([0., 10.]), np.array([0., 10.])     # initial policy: consume everything
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[2].plot(M0, c0, 'o-', ms=4, lw=0.8, label='iteration 0')
    for i in range(1, K + 1):
        c1 = np.maximum(lin(R * A + y, M0, c0), 1e-10)    # next period consumption, kept positive
        c = c1 / (beta * R)                               # inverted Euler equation: 1/c = beta R / c1
        M = A + c                                         # endogenous grid points
        if origin:                                        # add the origin: the constrained segment
            M, c, Ap = np.r_[0, M], np.r_[0, c], np.r_[np.nan, A]
        else:
            Ap = A
        for k, (x, yy) in enumerate(((Ap, c), (Ap, M), (M, c))):
            ax[k].plot(x, yy, 'o-', ms=4, lw=0.8, label=f'iteration {i}')
        M0, c0 = M, c
    for a, (xl, t) in zip(ax, (('savings A', 'consumption c(A)'), ('savings A', 'wealth M(A)'),
                                ('wealth M', 'consumption c(M)'))):
        a.set(xlabel=xl, title=t); a.grid(color='0.9')
    ax[2].legend(fontsize=8)
    fig.suptitle(title)
    plt.show()

egm_iterations(beta=0.95, R=1.0, y=0.0, K=4, title='Cake eating, beta = 0.95, R = 1, y = 0')
```

Each iteration produces five points of the policy, and the points move: the grid on $M$
is a product of the iteration, while the grid on $A$ stays put.

## Corner solutions

The EGM step uses the Euler equation, which holds only at interior solutions. The
constraint $0 \leqslant c \leqslant M$ needs separate care.

- **The lower bound** $c \geqslant 0$ never binds when $\lim_{c \to 0} u(c) = -\infty$,
  which holds for $\log c$ and for any CRRA utility with $\rho \geqslant 1$.
- **The upper bound** $c \leqslant M$ binds when the consumer would like to borrow, and
  then $A = 0$.

Monotonicity tells where the bound binds. If the utility function is increasing and
concave, savings $A = M - c(M)$ are non-decreasing in $M$. The EGM step at $A_1 = 0$ gives the
smallest wealth at which saving nothing is optimal,

$$
M_1 = (u')^{-1}\Big( \beta R\, \mathbb{E}_{y}\, u'\big(c(\tilde{y})\big) \Big),
$$

and for every $M < M_1$ the consumer must also save nothing: the constraint binds and
$c(M) = M$. In the code this is one extra point at the origin, $(M, c) = (0, 0)$: linear
interpolation between it and $(M_1, c_1)$ is exactly the 45-degree segment $c = M$.

```{code-cell} python3
:tags: [hide-input]

egm_iterations(beta=0.95, R=1.05, y=1.0, K=12, origin=True,
               title='Consumption-savings with income, beta = 0.95, R = 1.05, y = 1, origin added')
```

With income, the consumer with little wealth eats all of it and waits for the next
paycheck, and the policy has a kink where the constraint stops binding. Without the
point at the origin, the interpolated policy below $M_1$ would be an extrapolation of the
interior solution, and the consumer would be allowed to borrow.

:::{div}
:class: discussion

- Why does the grid on savings start at $A = 0$, and what would a grid point $A < 0$ mean?
- The kink of the policy at $M_1$ moves from one iteration to the next. How do VFI and
  time iterations on a fixed grid represent such a kink?
:::

## Implementation and comparison

The EGM solver for the stochastic model is a few lines, fully vectorized: savings along
the first axis, income nodes along the second, and the expectation as a product with
the weights. It uses the model object and the tools of
[Class 16](16_cons_practice.md), imported from `deaton.py`.

```{code-cell} python3
:tags: [hide-input]

import sys, time, pathlib, tempfile, urllib.request
CODE = pathlib.Path('../gh-code/session16-oct20')          # the code of Class 16
if not CODE.exists():                                     # otherwise fetch it from the code repository
    CODE = pathlib.Path(tempfile.mkdtemp())
    urllib.request.urlretrieve('https://raw.githubusercontent.com/fediskhakov/sb-dse-code/main/session16-oct20/deaton.py',
                               CODE / 'deaton.py')
sys.path.insert(0, str(CODE))
from deaton import deaton, interp, solve_vfi, bellman_discretized, bellman_continuous, \
                   solve_timeiter, accuracy, simulate
```

```{code-cell} python3
def solve_egm(m, maxiter=500, tol=1e-4, callback=None):
    '''EGM for the consumption-savings model, returns the policy on the state grid m.grid'''
    A = np.linspace(0, m.Mbar, m.ngrid)                   # grid on savings, the post-decision state
    M0, c0 = np.array([0., m.Mbar]), np.array([0., m.Mbar])   # initial policy: consume everything
    for i in range(maxiter):
        M1 = m.next_wealth(A[:, None], 0, m.quadp[None, :])   # next period wealth: savings x income nodes
        c1 = np.maximum(interp(M1, M0, c0), m.eps)        # next period consumption, kept positive
        rhs = m.beta * m.R * (m.mu(c1) @ m.quadw)         # RHS of the Euler equation
        c = np.r_[0., m.imu(rhs)]                         # inverted Euler equation, origin added
        M = np.r_[0., A + c[1:]]                          # endogenous grid points
        err = np.max(np.abs(interp(m.grid, M, c) - interp(m.grid, M0, c0)))
        if callback: callback(iter=i, M=M, c=c, err=err)
        if err < tol: break
        M0, c0 = M, c
    else:
        raise RuntimeError('No convergence: maximum number of iterations achieved')
    return interp(m.grid, M, c)                           # policy on the fixed grid, to compare
```

The policy is returned on the fixed state grid of the model, so all four solvers can be
compared with the same [Euler residuals](16_cons_practice.md#accuracy-the-euler-equation-residuals)
and the same [Keane's test](15_cake_eating.md#keane-test).

```{code-cell} python3
:tags: [hide-input]

print(f'{"grid":>5} {"solver":<25} {"time, s":>9} {"Euler residual":>15}')
for n in (50, 100, 200):
    m = deaton(ngrid=n)
    for name, solver in (('VFI, discretized choice', lambda: solve_vfi(m, bellman_discretized)),
                         ('VFI, continuous choice', lambda: solve_vfi(m, bellman_continuous)),
                         ('time iterations', lambda: solve_timeiter(m)),
                         ('EGM', lambda: solve_egm(m))):
        t0 = time.perf_counter()
        c = solver()
        print(f'{n:>5} {name:<25} {time.perf_counter() - t0:>9.4f} {accuracy(m, c):>15.2e}')

k = deaton(beta=0.9, R=1/0.9, sigma=1e-10, nquad=2)       # Keane's test: deterministic, beta*R = 1
_, C = simulate(k, solve_egm(k, tol=1e-10, maxiter=5000), 5.0, 100)
print(f"\nKeane's test for EGM: max |c_t / c* - 1| = {np.max(np.abs(C[0] / (((k.R - 1)*5 + 1)/k.R) - 1)):.1e}")
```

EGM is as accurate as time iterations, the most accurate of the earlier solvers, and two
orders of magnitude faster. It passes Keane's test, too. All of the speed comes from
replacing the search in every state by a closed-form inversion of the Euler equation.

## The class of models EGM solves

EGM in this form needs:

- one continuous state, wealth, and one continuous choice, consumption
- a strictly concave, increasing and differentiable utility function whose marginal
  utility can be inverted analytically
- a budget constraint that makes next period's state a function of the post-decision
  state $A$ alone, plus shocks
- finite or infinite horizon in discrete time, with occasionally binding borrowing
  constraints

Additional state variables are fine if they evolve exogenously, or as functions of $A$
rather than of $M$ or $c$. The class is small, but it includes many workhorse models of
micro and macro economics, and within it EGM is by far the best available method.

Where it breaks:

- **Several continuous states and choices.** The endogenous grid is irregular in more
  than one dimension, and interpolating on an irregular grid is hard and slow
  {cite:p}`barillas2007GeneralizationEndogenousGrid,white2015MethodEndogenousGridpoints,ludwig2018EndogenousGridsHigher,megm,lujanEGMnSequentialEndogenous`.
- **Non-convex problems.** With a discrete choice on top of consumption, for example
  retirement, the value function has kinks, the Euler equation is necessary but no
  longer sufficient, and EGM finds every solution of it, including the suboptimal ones.
  An upper envelope step removes them
  {cite:p}`fella2014GeneralizedEndogenousGrid,egm,druedahl2017GeneralEndogenousGridb`:
  this is the DC-EGM of the next class.

:::{div}
:class: discussion

- Which features of the Rust bus engine model of [Class 9](9_zurcher.md) rule out EGM?
- A model has wealth and human capital, both continuous. What happens to the endogenous
  grid?
:::

````{note} References and additional resources

(17_egm_references)=
- 📖 {cite:t}`carroll2006MethodEndogenousGridpoints` "The method of endogenous gridpoints for solving dynamic stochastic optimization problems"
- 📖 {cite:t}`egm` "The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks"
- 📖 {cite:t}`fella2014GeneralizedEndogenousGrid` "A Generalized Endogenous Grid Method for Non-Smooth and Non-Concave Problems"
- 📖 {cite:t}`druedahl2017GeneralEndogenousGridb` "A general endogenous grid method for multi-dimensional models with non-convexities and constraints"
- 📖 {cite:t}`barillas2007GeneralizationEndogenousGrid` "A Generalization of the Endogenous Grid Method"
- 📖 {cite:t}`white2015MethodEndogenousGridpoints` "The Method of Endogenous Gridpoints in Theory and Practice"
- 📖 {cite:t}`ludwig2018EndogenousGridsHigher` "Endogenous Grids in Higher Dimensions: Delaunay Interpolation and Hybrid Methods"
- 📖 {cite:t}`megm` "Multidimensional endogenous gridpoint method"
- 📺 Foundations of Computational Economics, lectures 41–42 on EGM [fedor.iskh.me/compecon](https://fedor.iskh.me/compecon)
````
