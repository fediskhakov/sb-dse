---
title: "🔬 Practice: successive approximations and multiple equilibria"
short_title: "🔬 Equilibrium practice"
subtitle: Class 22 — Tuesday, November 10
downloads:
  - file: 22_eqb_practice.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

<span class="tag wip">🚧 🚧 🚧 Not the final version 🚧 🚧 🚧</span>

Today we solve the [market of platforms of Class 21](21_eqb_cars.md#eqb-platform-model)
by successive approximations, the simplest algorithm for an equilibrium fixed point. A
stronger network effect brings multiple equilibria, and with them the ways this algorithm fails.


````{hint} Running the code for this lecture
:class: dropdown

The code for this class is in the folder `session22-nov10/` of the course **code
repository**. The notebook `eqb.ipynb` has all the code of this page.

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


(eqb-platform-recap)=
## The platform market

The model of [Class 21](21_eqb_cars.md#eqb-platform-model) in brief:

- $m$ platforms $i$ and a unit mass of consumers of $n$ types $j$, with population shares
  $p_j$
- the utility $u_{ij} = c_{ij} + \lambda s_i$ of platform $i$ to type $j$: the valuation
  $c_{ij}$ plus the network effect of strength $\lambda$ times the market share $s_i$
- EV1 taste shocks, so the choice probabilities $P_{ij}$ are logit, computed with the
  [de-maxing trick of Class 6](6_logit.md#demaxing), and the market shares add them up,
  $s = P p$

An equilibrium is a fixed point of the map from the shares to the shares:

$$
s_i = \Phi_i(s) = \sum_{j=1}^n p_j \frac{\exp(c_{ij} + \lambda s_i)}{\sum_{k=1}^m \exp(c_{kj} + \lambda s_k)},
\qquad i = 1, \dots, m .
$$

Of the three equivalent ways to close the loop, the shares have the fewest unknowns, so
the code iterates on them. The first half of the class keeps $\lambda = 1$, as in the
original example.

(eqb-successive-approximations)=
## Successive approximations

Any equation $g(x) = 0$ can be written as a **fixed point equation**

$$
x = F(x), \qquad \text{for example } F(x) = x + g(x) ,
$$

although not every such $F$ is a good one, as we see below. **Successive
approximations** apply the operator again and again, $x_{k+1} = F(x_k)$, and stop when the
iterations stop moving:

```text
Input: operator F, starting value x0, tolerance tol, maximum number of iterations
Algorithm:
  for k = 0, 1, ..., maxiter - 1:
      x1 = F(x0)
      if max |x1 - x0| < tol: return x1
      x0 = x1
  fail: no convergence in maxiter iterations
```

This is the algorithm of value function iterations in
[Class 8](8_dp_infinite.md#dptheory), with the Bellman operator as $F$. There the
**contraction** property of the operator guaranteed convergence from any starting value,
by the Banach fixed point theorem. Most equilibrium operators are not contractions, and
the guarantee is local at best.

When does it converge? Near a fixed point $x^\star$ the iterations behave like the linear
map given by the Jacobian $\nabla F(x^\star)$:

$$
x_{k+1} - x^\star \approx \nabla F(x^\star) \, (x_k - x^\star) .
$$

Therefore:

- the iterations converge to $x^\star$ from nearby starting values when the **spectral
  radius** $\rho = \max |\text{eig}\, \nabla F(x^\star)| < 1$, and such $x^\star$ is called
  **stable**; in one dimension the condition is $|F'(x^\star)| < 1$
- the convergence is **linear**: each iteration shrinks the error by the factor $\rho$, so
  reaching the tolerance takes about $\log(tol) / \log(\rho)$ iterations
- a fixed point with $\rho > 1$ repels the iterations: it is **unstable**, and
  successive approximations never find it

For the platform market the Jacobian follows from the derivative of the logit
probabilities, $\partial P_{ij} / \partial u_{kj} = P_{ij} (\mathbf{1}\{i = k\} - P_{kj})$:

$$
\frac{\partial \Phi_i(s)}{\partial s_k} = \lambda \sum_{j=1}^n p_j P_{ij} \big( \mathbf{1}\{i = k\} - P_{kj} \big),
\qquad \nabla \Phi(s) = \lambda \sum_{j=1}^n p_j \big( \text{diag}(P_j) - P_j P_j' \big),
$$

with $P_j$ the column of choice probabilities of type $j$. The strength of the network
effect $\lambda$ scales the whole Jacobian, and with it the speed of convergence and the
stability of every equilibrium.

Where successive approximations break:

- only **stable** fixed points can be computed
- the **starting value** decides which fixed point is found, when there are several
- convergence is **slow** when $\rho$ is close to one: $\rho = 0.99$ needs about 2300
  iterations for $tol = 10^{-10}$
- an eigenvalue below $-1$ makes the iterations **oscillate** instead of converging

But the method is very easy to implement, and every iteration is cheap: no derivatives, no
linear systems.

(eqb-implementation)=
## Implementation

The code follows the architecture of the course: a model object that holds the parameters
and the parts of the model, and a solver that knows nothing about the model except the
operator.

```{code-cell} python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

class platform:
    '''Market of m platforms and n consumer types, utility u_ij = c_ij + s_i'''

    def __init__(self, m=3, n=2, seed=2025):
        rng = np.random.default_rng(seed)
        self.m, self.n = m, n                                 # platforms i, consumer types j
        self.c = rng.uniform(size=(m, n))                     # valuations of platform i by type j
        self.p = rng.dirichlet(np.ones(n))                    # population shares of the types

    def utility(self, s):
        '''Utilities of all platforms for all types, m x n, given the market shares s'''
        return self.c + s[:, None]                            # the network effect: same s_i for every type

    def ccp(self, u):
        '''Logit choice probabilities, one column per consumer type'''
        e = np.exp(u - u.max(axis=0))                         # de-max by column: no overflow
        return e / e.sum(axis=0)

    def shares(self, P):
        '''Market shares from the choice probabilities: unit mass of consumers'''
        return P @ self.p

    def F(self, s):
        '''The fixed point operator s -> Phi(s)'''
        s1 = self.shares(self.ccp(self.utility(s)))
        assert np.isclose(s1.sum(), 1)                        # every consumer joins one platform
        return s1
```

The solver has the signature shared by all solvers of the course, and a callback that
receives every iteration, for printing or plotting:

```{code-cell} python3
def solve_sa(F, x0, tol=1e-10, maxiter=1000, callback=None):
    '''Successive approximations x_k+1 = F(x_k) from x0, until the largest change is below tol'''
    x0 = np.asarray(x0, dtype=float)
    for k in range(maxiter):
        x1 = F(x0)                                            # one application of the operator
        err = np.max(np.abs(x1 - x0))                         # sup-norm distance between iterations
        if callback: callback(iter=k, err=err, x=x1, x0=x0)
        if err < tol: break
        x0 = x1
    else:
        raise RuntimeError(f'Failed to converge in {maxiter} iterations')
    return x1
```

Three platforms, two consumer types, valuations drawn at random, iterations started from
equal shares:

```{code-cell} python3
md = platform()
print('valuations c:', md.c.round(3), 'type mix p:', md.p.round(3), sep='\n')
errs = []                                                     # the callback keeps the errors for later
s = solve_sa(md.F, np.ones(md.m) / md.m, callback=lambda **kw: errs.append(kw['err']))
P = md.ccp(md.utility(s))
print(f'converged in {len(errs)} iterations, residual {np.max(np.abs(md.F(s) - s)):.1e}')
print('equilibrium choice probabilities:', P.round(4), 'equilibrium market shares:', s.round(4), sep='\n')
```

Platform 2 has the largest market share, 0.388, although type 1, the majority, values it
least. Type 2 puts half of its consumers on it, while type 1 spreads almost evenly over
the three platforms, because its valuations differ by less than $0.2$.

The theory predicts the speed: the error should shrink by the spectral radius of the
Jacobian at every iteration.

```{code-cell} python3
def jacobian(md, s, lam=1.0):
    '''Jacobian of Phi at s: lam * sum_j p_j (diag(P_j) - P_j P_j')'''
    P = md.ccp(md.utility(s))
    return lam * sum(md.p[j] * (np.diag(P[:, j]) - np.outer(P[:, j], P[:, j])) for j in range(md.n))

rho = np.max(np.abs(np.linalg.eigvals(jacobian(md, s))))      # spectral radius at the equilibrium
ratio = np.array(errs[1:]) / np.array(errs[:-1])              # how much each iteration shrinks the error
print(f'spectral radius {rho:.4f}, error ratio in the last iterations {ratio[-3:].round(4)}')
print(f'iterations predicted for tol = 1e-10: {np.log(1e-10) / np.log(rho):.1f}')
```

```{code-cell} python3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
k = np.arange(len(errs))
ax.semilogy(k, errs, 'o-', color='tab:blue', lw=1.5, label='successive approximations')
ax.semilogy(k, errs[0] * rho**k, '--', color='tab:red', lw=1.5, label='$\\rho^k$, the spectral radius')
ax.set(xlabel='iteration k', ylabel='$\\max_i |s^{k+1}_i - s^k_i|$', title='Linear convergence')
ax.legend(); ax.grid(color='0.9')
plt.show()
```

The error falls along a straight line on the log scale, by the factor $0.35$ per
iteration, as the spectral radius $\rho = 0.352$ predicts. The prediction of 22 iterations
is close to the 20 it took, which started from an error already below one.

:::{div}
:class: discussion

- Why is $\nabla \Phi$ singular, whatever the parameters? What does its null vector mean?
- How would the number of iterations change if each consumer type valued the platforms
  very differently?
:::

(eqb-network-strength)=
## The strength of the network effect

The parameter $\lambda$ controls how strongly the equilibrium shares feed back into the
utility. A subclass adds it and changes only the utility:

```{code-cell} python3
class network(platform):
    '''Platform market with the strength of the network effect lam: u_ij = c_ij + lam s_i'''

    def __init__(self, lam=1.0, **kw):
        super().__init__(**kw)
        self.lam = lam

    def utility(self, s):
        return self.c + self.lam * s[:, None]
```

Two platforms make the problem one-dimensional. With $s_2 = 1 - s_1$ the equilibrium is a
fixed point of the scalar map $s_1 \mapsto \Phi_1(s_1, 1 - s_1)$, with the slope

$$
\frac{d \Phi_1}{d s_1} = 2 \lambda \sum_{j=1}^n p_j P_{1j} (1 - P_{1j}) \leqslant \frac{\lambda}{2} ,
$$

because $P(1 - P) \leqslant 1/4$. Therefore:

- for $\lambda < 2$ the map is a contraction on $[0, 1]$, whatever the valuations and the
  type mix: the equilibrium is unique, and successive approximations find it from any start
- for $\lambda > 2$ the slope can exceed one, and then the map can cross the 45-degree line
  three times
- with identical valuations of the two platforms the threshold is exact: three equilibria
  for every $\lambda > 2$, the result of {cite:t}`brock2001DiscreteChoiceSocial`

The example: each type prefers its own platform by one unit of utility, and type 1 is the
majority, 60% of the consumers.

```{code-cell} python3
mn = network(m=2, n=2)
mn.c, mn.p = np.eye(2), np.array([0.6, 0.4])                  # each type prefers its own platform, type 1 is larger
Phi = lambda s1: mn.F(np.array([s1, 1 - s1]))[0]              # two platforms: s_2 = 1 - s_1, a scalar map

def equilibria(mn, grid=np.linspace(1e-9, 1 - 1e-9, 2001)):
    '''All equilibria s_1 of the two-platform market, with the slope of Phi at each'''
    g = np.array([Phi(x) for x in grid]) - grid
    eq = [brentq(lambda x: Phi(x) - x, a, b, xtol=1e-14)      # bracket every sign change, then refine
          for a, b, ga, gb in zip(grid[:-1], grid[1:], g[:-1], g[1:]) if ga * gb < 0]
    slope = [jacobian(mn, np.array([x, 1 - x]), mn.lam)[0, 0] * 2 for x in eq]   # dPhi/ds_1, s_2 = 1 - s_1
    return np.array(eq), np.array(slope)
```

The function `equilibria` does not use successive approximations: it brackets every
sign change of $\Phi_1(s_1) - s_1$ on a grid and refines it with `brentq`, which finds
unstable equilibria as well. That is possible because the problem is one-dimensional.

```{code-cell} python3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
sg = np.linspace(0, 1, 401)
ax.plot(sg, sg, color='0.5', lw=1, label='45-degree line')
for lam, col in ((1, 'tab:green'), (3, 'tab:red'), (4, 'tab:blue')):
    mn.lam = lam
    ax.plot(sg, [Phi(x) for x in sg], color=col, lw=1.8, label=f'$\\Phi(s_1)$, λ = {lam}')
    eq, slope = equilibria(mn)
    for x, d in zip(eq, slope):                               # filled: stable, |slope| < 1; open: unstable
        ax.plot(x, x, 'o', ms=9, color=col, mfc=col if abs(d) < 1 else 'white', mew=2)
    print(f'λ = {lam}: equilibria s_1 = {eq.round(4)}, slope of Phi {slope.round(3)}')
ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xlabel='$s_1$', ylabel='$\\Phi(s_1)$',
       title='Equilibria of the two-platform market')
ax.legend(loc='upper left'); ax.grid(color='0.9')
plt.show()
```

What the figure shows:

- with $\lambda = 1$ the shares follow the preferences: platform 1 gets 58% of the market,
  a little less than its 60% of loyal consumers would suggest, because of the taste shocks
- with $\lambda = 3$ the market has **tipped**: one equilibrium, with 90% on platform 1
- with $\lambda = 4$ there are three equilibria: platform 1 dominates with 97%, platform 2
  dominates with 96%, or the market splits with $s_1 = 0.42$

In the second stable equilibrium the platform preferred by the minority wins almost the
whole market, a **coordination failure**: every consumer of type 1 would be better off if
all of them switched together, and none of them is better off switching alone. The split
equilibrium has slope $1.6$ and is unstable.

(eqb-sa-multiple)=
## Successive approximations with multiple equilibria

The **cobweb diagram** draws the iterations: up to the curve $\Phi$, across to the
45-degree line, up to the curve again.

```{code-cell} python3
:tags: [hide-input]

def cobweb(ax, path, col, label):
    '''Draw the steps of successive approximations on the 45-degree diagram'''
    x = [p[0] for p in path]
    for a, b in zip(x[:-1], x[1:]):
        ax.plot([a, a, b], [a, b, b], color=col, lw=1)         # up to the curve, across to the 45-degree line
    ax.plot(x[0], x[0], 's', color=col, ms=8, label=label)

mn.lam = 4
eq, slope = equilibria(mn)
fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
ax.plot(sg, sg, color='0.5', lw=1)
ax.plot(sg, [Phi(x) for x in sg], color='black', lw=1.8, label='$\\Phi(s_1)$, λ = 4')
for s0, col in ((0.40, 'tab:red'), (0.45, 'tab:blue'), (0.05, 'tab:purple'), (0.99, 'tab:green')):
    path = [np.array([s0, 1 - s0])]
    s = solve_sa(mn.F, path[0], callback=lambda **kw: path.append(kw['x']))
    cobweb(ax, path, col, f'$s_1^0$ = {s0}: {len(path) - 1} iterations to $s_1$ = {s[0]:.4f}')
ax.plot(eq[1], eq[1], 'o', ms=10, color='black', mfc='white', mew=2, label=f'unstable equilibrium {eq[1]:.4f}')
ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xlabel='$s_1$', ylabel='$\\Phi(s_1)$',
       title='Successive approximations with three equilibria')
ax.legend(loc='upper left'); ax.grid(color='0.9')
plt.show()
```

The unstable equilibrium splits the interval into two **basins of attraction**. Starting at
$0.40$ the iterations move away from $0.4227$ down to the low equilibrium, starting at
$0.45$ up to the high one. No starting value other than $0.4227$ itself leads to the split
equilibrium.

The same experiment for a range of $\lambda$, with all the equilibria computed by
`equilibria` and the limits of successive approximations started from $s_1^0 = 0$ and
$s_1^0 = 1$:

```{code-cell} python3
:tags: [hide-input]

lams = np.linspace(0, 6, 121)
iters = {s0: [] for s0 in (0, 0.5, 1)}                        # iterations to converge from three starting points
n3 = []                                                       # where the three equilibria appear
fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
for lam in lams:
    mn.lam = lam
    eq, slope = equilibria(mn)
    stable = abs(slope) < 1
    n3.append(eq.size == 3)
    ax.plot(np.full(stable.sum(), lam), eq[stable], '.', ms=5, color='black')
    ax.plot(np.full((~stable).sum(), lam), eq[~stable], '.', ms=5, color='tab:red')
    for s0, col in ((0, 'tab:purple'), (0.5, None), (1, 'tab:green')):
        path = []
        s = solve_sa(mn.F, [s0, 1 - s0], callback=lambda **kw: path.append(kw['x']))
        iters[s0].append(len(path))
        if col and lam in lams[::4]: ax.plot(lam, s[0], 'o', mfc='none', color=col, ms=9)
print(f'three equilibria from λ = {lams[np.argmax(n3)]:.2f} on, on this grid of λ')
ax.plot([], [], '.', color='black', label='stable equilibria')
ax.plot([], [], '.', color='tab:red', label='unstable equilibria')
ax.plot([], [], 'o', mfc='none', color='tab:purple', label='SA from $s_1^0 = 0$')
ax.plot([], [], 'o', mfc='none', color='tab:green', label='SA from $s_1^0 = 1$')
ax.set(xlabel='strength of the network effect λ', ylabel='$s_1$', title='Equilibrium market share of platform 1')
ax.legend(loc='center right'); ax.grid(color='0.9')
plt.show()
```

- Below $\lambda \approx 3.1$ the equilibrium is unique, and the share of platform 1 rises
  with $\lambda$ as the market tips towards it.
- At $\lambda \approx 3.1$ a pair of new equilibria appears, a stable one with platform 2
  dominant and the unstable one between them.
- Successive approximations trace the stable branches only. Which branch they find depends
  on the starting value: from $s_1^0 = 0$ the low one, from $s_1^0 = 1$ or from equal
  shares the high one. The red branch is invisible to them.

The cost of finding an equilibrium changes with $\lambda$ too:

```{code-cell} python3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
for s0, col in ((0, 'tab:purple'), (0.5, 'tab:blue'), (1, 'tab:green')):
    ax.plot(lams, iters[s0], color=col, lw=1.5, label=f'SA from $s_1^0$ = {s0}')
ax.set(xlabel='strength of the network effect λ', ylabel='iterations to tol = 1e-10',
       title='The cost of successive approximations')
ax.legend(); ax.grid(color='0.9')
plt.show()
```

The number of iterations peaks twice:

- around $\lambda \approx 2.2$ from every starting value, where the slope of $\Phi$ at the
  unique equilibrium is largest, $0.71$: the rule $\log(tol) / \log(\rho)$ gives 68
  iterations, close to the 63 observed
- at $\lambda \approx 3.1$ from $s_1^0 = 0$, at 180 iterations: the new equilibrium is born
  with slope exactly one, and just before it is born the iterations crawl through the
  narrow gap between the curve and the 45-degree line

:::{div}
:class: discussion

- Data on the market shares show one equilibrium. With $\lambda = 4$, how would you know
  which one, and what does that do to a likelihood that needs the equilibrium shares for
  every trial value of the parameters?
- A small change in $c_{ij}$ moves the market from the high branch to the low one at some
  $\lambda$. Is the market share then a continuous function of the parameters?
- How would you find all the equilibria with $m = 3$ platforms?
:::

(eqb-congestion)=
### Congestion: negative feedback

With $\lambda < 0$ a larger platform is less attractive, for example because of
congestion. $\Phi_1$ is then decreasing in $s_1$, it crosses the 45-degree line exactly
once, and the equilibrium is unique. Successive approximations can still fail:

```{code-cell} python3
def damped(F, w):
    '''Damped operator: a step of size w from x towards F(x); w = 1 is plain successive approximations'''
    return lambda x: (1 - w) * x + w * F(x)

mn.lam = -4                                                   # congestion: a larger platform is less attractive
eq, slope = equilibria(mn)
print(f'λ = {mn.lam}: equilibrium s_1 = {eq.round(4)}, slope of Phi {slope.round(3)}')
try:
    solve_sa(mn.F, [0.5, 0.5])
except RuntimeError as e:
    print('plain SA:', e)
path_d = [np.array([0.5, 0.5])]
s = solve_sa(damped(mn.F, 0.3), path_d[0], callback=lambda **kw: path_d.append(kw['x']))
print(f'damped SA, w = 0.3: s_1 = {s[0]:.4f} in {len(path_d) - 1} iterations')
```

```{code-cell} python3
:tags: [hide-input]

path = [np.array([0.5, 0.5])]
for k in range(30): path.append(mn.F(path[-1]))               # plain SA, stopped after 30 steps
Phi_w = lambda s1: 0.7 * s1 + 0.3 * Phi(s1)                   # the damped map, w = 0.3
fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
ax.plot(sg, sg, color='0.5', lw=1)
ax.plot(sg, [Phi(x) for x in sg], color='black', lw=1.8, label='$\\Phi(s_1)$, λ = -4')
ax.plot(sg, [Phi_w(x) for x in sg], '--', color='tab:blue', lw=1.8, label='damped map, w = 0.3')
cobweb(ax, path, 'tab:red', f'plain SA: a cycle between {path[-2][0]:.3f} and {path[-1][0]:.3f}')
cobweb(ax, path_d, 'tab:blue', f'damped SA: {len(path_d) - 1} iterations to {path_d[-1][0]:.4f}')
ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xlabel='$s_1$', ylabel='$\\Phi(s_1)$',
       title='Successive approximations with congestion')
ax.legend(loc='upper right'); ax.grid(color='0.9')
plt.show()
```

The slope of $\Phi$ at the equilibrium is $-1.59$, so every iteration overshoots by more
than it corrects. The iterations spiral out into a **cycle** between $0.04$ and $0.97$:
the crowd leaves the congested platform for the empty one, which is then the congested one.

**Damping** takes only a step of size $w$ towards $\Phi(x)$. The damped map has the same
fixed point and the slope $1 - w + w \Phi'(s^\star) = 0.22$ at $w = 0.3$, which converges in
14 iterations. The slope is zero with $w = 1 / (1 - \Phi'(s^\star))$, a Newton step in one
dimension.

Damping does not help with the unstable equilibria of the previous section: their slope is
above one, and $1 - w + w \Phi'$ is then above one for every $w$ between zero and one. A
solver that uses the derivative, such as [Newton's method of Class 5](5_solvers.md#multivariate-newton),
finds them, and finding *all* equilibria comes back when we study games.

:::{div}
:class: discussion

- Why can the damped map not have a fixed point that the original map does not have?
- The NPL iterations of [Class 14](14_npl.md#npl-zero-jacobian) are successive
  approximations on a policy operator. Which of the failures of this class could they run into?
:::

With more than two platforms the bracketing of `equilibria` no longer works, and the
picture of the basins of attraction has two dimensions.

````{warning} Practical task: all equilibria of a three-platform market

(task-eqb-three-platforms)=
Pull the class code and open the notebook `eqb.ipynb` in `session22-nov10/`:

```bash
cd sb-dse-code && git pull
# first time: git clone https://github.com/fediskhakov/sb-dse-code.git
```

Take three platforms and three consumer types, each preferring its own platform,
`c = np.eye(3)`, with the type shares `p = (0.45, 0.35, 0.2)` and $\lambda = 5$.

1. Run successive approximations from a grid of starting shares on the simplex and
   collect the distinct limits. How many stable equilibria are there?
2. Color every starting point by the equilibrium it converges to, and plot the basins of
   attraction in the $(s_1, s_2)$ triangle.
3. Find the unstable equilibria with `scipy.optimize.root` on $\Phi(s) - s$, in the two
   free shares, started from many points. Check the stability of every solution with
   the spectral radius of `jacobian`.
4. Repeat for $\lambda$ from 2 to 7. Where do the new equilibria appear, and which
   platforms do they favor?
````

````{note} References and additional resources

(eqb-practice-references)=
- 📖 {cite:t}`brock2001DiscreteChoiceSocial` "Discrete Choice with Social Interactions"
- 📖 {cite:t}`katz1985NetworkExternalitiesCompetition` "Network Externalities, Competition, and Compatibility"
- 📖 {cite:t}`rochet2003PlatformCompetitionTwosided` "Platform Competition in Two-Sided Markets"
- 📖 {cite:t}`judd1998NumericalMethodsEconomics` "Numerical Methods in Economics", on fixed-point iteration and its acceleration
- 📺 The lecture on successive approximations in *Foundations of Computational Economics*, the source of the platform example [youtu.be/AQt9Q9qc3io](https://youtu.be/AQt9Q9qc3io)
- 📺 Oscar Veliz on fixed point iteration [youtube.com/watch?v=OLqdJMjzib8](https://www.youtube.com/watch?v=OLqdJMjzib8) and [youtube.com/watch?v=FyCviw2ZA2o](https://www.youtube.com/watch?v=FyCviw2ZA2o)
- 📄 Paul Belleflamme's introduction to the economics of platform competition, [part 1](http://www.ipdigit.eu/2020/04/an-introduction-to-the-economics-of-platform-competition-part-1/), [part 2](http://www.ipdigit.eu/2020/04/an-introduction-to-the-economics-of-platform-competition-part-2/), [part 3](http://www.ipdigit.eu/2020/04/an-introduction-to-the-economics-of-platform-competition-part-3/)
````
