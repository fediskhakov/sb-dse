---
title: "🔬 Practice: successive approximations and multiplicity in the equilibrium models"
short_title: "🔬 Equilibrium practice"
subtitle: Class 22 — Tuesday, November 10
downloads:
  - file: 22_eqb_practice.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

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

Recall the simple *static* model from [Class 21](21_eqb_cars.md#eqb-platform-model)

A **platform** is a product or a service with **network effects**
{cite:p}`katz1985NetworkExternalitiesCompetition`: its value to a user grows with the
number of other users.
The setup:

- $m$ platforms, $i = 1, \dots, m$
- a **unit mass** of consumers of $n$ types, $j = 1, \dots, n$, 
- $p_j$ the share of type $j$ in the population
- $c_{ij}$ the **valuation** of platform $i$ by a consumer of type $j$
- $s_i$ the **market share** of platform $i$, the fraction of all consumers on it

The utility of a consumer of type $j$ from platform $i$ is

$$
u_{ij} = c_{ij} + \lambda s_i ,
$$

where $\lambda$ is the strength of the network effect.

Each consumer also draws an extreme value type 1 taste shock $\varepsilon_i$ for every
platform, i.i.d. across consumers and platforms, and joins the platform with the highest
$u_{ij} + \varepsilon_i$. The choice probabilities are logit, as in
[Class 6](6_logit.md#probabilistic-choice):

$$
P_{ij} = \frac{\exp(u_{ij})}{\sum_{k=1}^m \exp(u_{kj})} .
$$

$P_{ij}$ is the probability that a consumer of type $j$ chooses platform $i$, the CCP

Each consumer chooses once, and a unit mass of them turns the probabilities into shares:

$$
s_i = \sum_{j=1}^n p_j P_{ij}, \qquad s = P p ,
$$

a matrix product with $p$ the column vector of type shares.

Combining the three expressions gives $m$ equations in the $m$ market shares:

$$
s_i = \Phi_i(s) = \sum_{j=1}^n p_j \frac{\exp(c_{ij} + \lambda s_i)}{\sum_{k=1}^m \exp(c_{kj} + \lambda s_k)},
\qquad i = 1, \dots, m .
$$

In **equilibrium** the market shares $s^\star$ solve $s^\star = \Phi(s^\star)$: every
consumer chooses the best platform given the shares $s^\star$, and these choices add up
to the shares $s^\star$. The loop that the equilibrium closes:

```{image} _static/eqb/platform_loop.png
:width: 75%
:align: center
:alt: The fixed point loop of the platform market: shares, utilities, choices
```

The expectations of the consumers are fulfilled in equilibrium: nobody would choose
differently knowing the shares that result. 

The fixed point can be computed in the **shares**, $s = \Phi(s)$, in the **utilities**, $u_{ij} = c_{ij} + \lambda \sum_t p_t P_{it}(u)$, or in the **choice probabilities**, $P = \Lambda(c + \lambda P p)$ with $\Lambda$ the logit map.
All three have the same solutions, but from the computational standpoint, it is more efficient to work in *shares* where the fixed point problem has the lowest dimension, $m-1$ instead of $m (n-1)$ in choice probabilities or $mn$ in utilities.


(eqb-successive-approximations)=
## Successive approximations

The simplest method for solving for the fixed point of an operator.

**Successive
approximations** apply the operator again and again, $x_{k+1} = \Phi(x_k)$, and stop when the
iterations stop moving.

````{attention} Algorithm (successive approximations)

**Input:** an operator $\Phi$, a starting value $x_0$, a tolerance $\epsilon > 0$ and the
maximum number of iterations $K$.

**Steps:** for $k = 0, 1, \dots, K - 1$:

1. **Iterate.** Compute $x_{k+1} = \Phi(x_k)$.
2. **Stop.** If $\max_i |x_{k+1, i} - x_{k, i}| < \epsilon$, return $x_{k+1}$.

If the loop ends without stopping, report a failure to converge in $K$ iterations.
````

:::{div}
:class: discussion

- Have you seen such an algorithm before?
:::

Actually any equation $g(x) = 0$ can be written as a fixed point equation, making this algorithm fairly general

$$
x = \Phi(x), \qquad \text{for example} \quad \Phi(x) = x + g(x)
$$

We need to understand when this algorithm is appropriate, namely what properties of $\Phi$ make it appropriate for the problem.

Of course, this is VFI of
[Class 8](8_dp_infinite.md#dptheory) with the Bellman operator as $\Phi$. There the
**contraction** property of the operator guaranteed convergence from any starting value,
by the Banach fixed point theorem. 

Most equilibrium operators are **not contractions**, and
the guarantee is local at best.

When does it converge? Near a fixed point $x^\star$ the iterations behave like the linear
map given by the Jacobian $\nabla \Phi(x^\star)$:

$$
x_{k+1} - x^\star \approx \nabla \Phi(x^\star) \, (x_k - x^\star) .
$$

Therefore:

- the iterations converge to $x^\star$ from nearby starting values when the **spectral radius** 
$$\rho = \max |\text{eig}\, \nabla \Phi(x^\star)|$$
is less than one, $\rho<1$, in which case such $x^\star$ is called **stable**
- in one dimension the condition is $|\Phi'(x^\star)| < 1$
- the convergence is **linear**: each iteration shrinks the error by the factor $\rho$, so
  reaching the tolerance takes about $\log(tol) / \log(\rho)$ iterations — **slow convergence**
- a fixed point with $\rho > 1$ repels the iterations: it is **unstable**, and
  successive approximations **never** find it

For the platform market the Jacobian follows from the derivative of the logit
probabilities, 
$$\partial P_{ij} / \partial u_{kj} = P_{ij} (\mathbf{1}\{i = k\} - P_{kj})$$

We have
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

:::{div}
:class: discussion

- Value function iterations converged from any starting value. What property of the
  Bellman operator does $\Phi$ lack in general?
- Consider $\Phi(x) = x + \alpha g(x)$ to solve $g(x) = 0$ using successive approximations. Is this more advantageous than simple approach with $\alpha=1$ above? What is the best choice of $\alpha$ in comparison with a step of Newton's method?
:::

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
    '''Market of m platforms and n consumer types, utility u_ij = c_ij + lam s_i'''

    def __init__(self, m=3, n=2, lam=1.0, seed=2025):
        rng = np.random.default_rng(seed)
        self.m, self.n = m, n                                 # platforms i, consumer types j
        self.lam = lam                                        # strength of the network effect
        self.c = rng.uniform(size=(m, n))                     # valuations of platform i by type j
        self.p = rng.dirichlet(np.ones(n))                    # population shares of the types

    def utility(self, s):
        '''Utilities of all platforms for all types, m x n, given the market shares s'''
        return self.c + self.lam * s[:, None]                 # the network effect: same s_i for every type

    def ccp(self, u):
        '''Logit choice probabilities, one column per consumer type'''
        e = np.exp(u - u.max(axis=0))                         # de-max by column: no overflow
        return e / e.sum(axis=0)

    def shares(self, P):
        '''Market shares from the choice probabilities: unit mass of consumers'''
        return P @ self.p

    def Phi(self, s):
        '''The fixed point operator s -> Phi(s)'''
        s1 = self.shares(self.ccp(self.utility(s)))
        assert np.isclose(s1.sum(), 1)                        # every consumer joins one platform
        return s1
```

The solver has the signature shared by all solvers of the course, and a callback that
receives every iteration, for printing or plotting:

```{code-cell} python3
def solve_sa(Phi, x0, tol=1e-10, maxiter=1000, callback=None):
    '''Successive approximations x_k+1 = Phi(x_k) from x0, until the largest change is below tol'''
    x0 = np.asarray(x0, dtype=float)
    for k in range(maxiter):
        x1 = Phi(x0)                                          # one application of the operator
        err = np.max(np.abs(x1 - x0))                         # sup-norm distance between iterations
        if callback: callback(iter=k, err=err, x=x1, x0=x0)
        if err < tol: break
        x0 = x1
    else:
        raise RuntimeError(f'Failed to converge in {maxiter} iterations')
    return x1
```

Three platforms, two consumer types, valuations drawn at random, $\lambda = 1$, iterations
started from equal shares:

```{code-cell} python3
md = platform()
print('valuations c:', md.c.round(3), 'type mix p:', md.p.round(3), sep='\n')
errs = []                                                     # the callback keeps the errors, to count the iterations
s = solve_sa(md.Phi, np.ones(md.m) / md.m, callback=lambda **kw: errs.append(kw['err']))
P = md.ccp(md.utility(s))
print(f'converged in {len(errs)} iterations, residual {np.max(np.abs(md.Phi(s) - s)):.1e}')
print('equilibrium choice probabilities:', P.round(4), 'equilibrium market shares:', s.round(4), sep='\n')
```

Platform 2 has the largest market share, 0.388, although type 1, the majority, values it
least. Type 2 puts half of its consumers on it, while type 1 spreads almost evenly over
the three platforms, because its valuations differ by less than $0.2$.

The theory predicts the speed: near the equilibrium each iteration should shrink the error
by the spectral radius of the Jacobian. A callback prints both at every iteration: the
relative error, which is the current error divided by the previous one, and the spectral
radius computed at the current approximation.

```{code-cell} python3
def jacobian(md, s):
    '''Jacobian of Phi at s: lam * sum_j p_j (diag(P_j) - P_j P_j')'''
    P = md.ccp(md.utility(s))
    return md.lam * sum(md.p[j] * (np.diag(P[:, j]) - np.outer(P[:, j], P[:, j])) for j in range(md.n))

def spectral_radius(md, s):
    '''Largest absolute eigenvalue of the Jacobian of Phi at s'''
    return np.max(np.abs(np.linalg.eigvals(jacobian(md, s))))

errs = [np.nan]                                               # no previous error at the first iteration
def report(iter, err, x, x0):
    '''Callback: print the error, the relative error and the spectral radius at the current x'''
    print(f'iter {iter:2d}: error {err:.3e}, relative error {err / errs[-1]:.4f}, spectral radius {spectral_radius(md, x):.4f}')
    errs.append(err)

s = solve_sa(md.Phi, np.ones(md.m) / md.m, callback=report)
print(f'eigenvalues at the equilibrium: {np.linalg.eigvalsh(jacobian(md, s)).round(4)}')
print(f'iterations predicted for tol = 1e-10: {np.log(1e-10) / np.log(spectral_radius(md, s)):.1f}')
```

The spectral radius settles at $0.3522$ after four iterations, long before the solver stops.
The relative error approaches it from below, from $0.33$ to $0.35$: the second eigenvalue,
$0.297$, also shapes the error early on, and its part dies out faster. The prediction of 22
iterations is close to the 20 it took, which started from an error already below one.

:::{div}
:class: discussion

- What is the spectral radius of the Bellman operator in Zurcher model?
:::


````{tip} Example: spectral radius of the Bellman operator in the Zurcher model
:class: dropdown

The [finite approximation of the Bellman operator of Class 9](9_zurcher.md#zurcher-bellman-matrix) is

$$
\Gamma(EV) = \Pi \cdot L \big( U(\text{keep}) + \beta EV, U(\text{replace}) + \beta EV[0] \big) ,
$$

with $\Pi$ the matrix of mileage transition probabilities and $L$ the logsum. The derivative
of the logsum is the probability-weighted derivative of its arguments. With $P[l]$ the
probability of keeping at state $l$ and $\bar{P}[l] = 1 - P[l]$ the probability of replacing,

$$
\frac{\partial L[l]}{\partial EV[k]} = \beta \big( P[l] \, \mathbf{1}\{k = l\} + \bar{P}[l] \, \mathbf{1}\{k = 0\} \big) ,
$$

because the keep value in row $l$ depends on $EV[l]$ and the replacement value in every row
depends on $EV[0]$. In matrix form the Jacobian is the
[Fréchet derivative of Class 9](9_zurcher.md#zurcher-frechet):

$$
\nabla \Gamma(EV) = \beta \, \Pi Q ,
$$

where

$$
Q = \text{diag}(P) + \bar{P} \, \mathbf{e}_0' =
\begin{pmatrix}
P[0] + \bar{P}[0] & 0 & 0 & \cdot & 0 \\
\bar{P}[1] & P[1] & 0 & \cdot & 0 \\
\bar{P}[2] & 0 & P[2] & \cdot & 0 \\
\cdot & \cdot & \cdot & \cdot & \cdot \\
\bar{P}[n-1] & 0 & 0 & \cdot & P[n-1]
\end{pmatrix}
$$

with $\mathbf{e}_0$ the unit vector of state 0. The matrix $Q$ moves the bus according to the
decision: it stays at $l$ with probability $P[l]$ and returns to state 0 with probability
$\bar{P}[l]$. It has non-negative elements and its rows sum to one, so $Q$ is a **stochastic
matrix**, and so is the product $A = \Pi Q$. Therefore:

- $\beta$ is an eigenvalue of $\nabla \Gamma$: rows that sum to one mean $A \mathbf{1} = \mathbf{1}$,
  so $\nabla \Gamma \, \mathbf{1} = \beta \mathbf{1}$
- no eigenvalue is larger in absolute value: from $A v = \mu v$ it follows that
  $|\mu| \max_i |v_i| = \max_i \big| \sum_k A_{ik} v_k \big| \leqslant \max_k |v_k|$, so $|\mu| \leqslant 1$

The spectral radius of the Bellman operator is $\rho = \beta$, at any $EV$ and for any
values of the parameters. The eigenvector $\mathbf{1}$ is a parallel shift: adding a constant
$c$ to $EV$ adds exactly $\beta c$ to $\Gamma(EV)$.
````



(eqb-network-strength)=
## The strength of the network effect

The parameter $\lambda$ controls how strongly the equilibrium shares feed back into the
utility.

Consider the case with two platforms, $m = 2$. Denote the share of platform 1 by $s = s_1$,
so that $s_2 = 1 - s$, and the equilibrium is a fixed point of a scalar map:

$$
s = \Phi_1(s) = \sum_{j=1}^n p_j \frac{\exp(c_{1j} + \lambda s)}{\exp(c_{1j} + \lambda s) + \exp\big(c_{2j} + \lambda (1 - s)\big)} .
$$

The first exponent grows with $s$ at the rate $\lambda$, and the second falls at the same
rate. The quotient rule, with $a_j = \exp(c_{1j} + \lambda s)$ and
$b_j = \exp\big(c_{2j} + \lambda (1 - s)\big)$, gives

$$
\frac{d}{d s} \frac{a_j}{a_j + b_j}
= \frac{\lambda a_j (a_j + b_j) - a_j (\lambda a_j - \lambda b_j)}{(a_j + b_j)^2}
= 2 \lambda \frac{a_j}{a_j + b_j} \frac{b_j}{a_j + b_j}
= 2 \lambda P_{1j} (1 - P_{1j}) .
$$

Therefore the slope of the map is

$$
\frac{d \Phi_1}{d s} = 2 \lambda \sum_{j=1}^n p_j P_{1j} (1 - P_{1j}) \leqslant \frac{\lambda}{2} ,
$$

because $P(1 - P) \leqslant 1/4$ and the type shares $p_j$ sum to one. Thus:

- for $\lambda < 2$ the map $\Phi_1$ is a contraction on $[0, 1]$, whatever the valuations and the
  type mix: the equilibrium is unique, and successive approximations find it from any start
- for $\lambda > 2$ the slope can exceed one, and then the map can cross the 45-degree line
  three times
- with identical valuations of the two platforms the threshold is exact: three equilibria
  for every $\lambda > 2$, the result of {cite:t}`brock2001DiscreteChoiceSocial`

The example: each type prefers its own platform by one unit of utility, and type 1 is the
majority, 60% of the consumers.

```{code-cell} python3
mn = platform(m=2, n=2)
mn.c, mn.p = np.eye(2), np.array([0.6, 0.4])                  # each type prefers its own platform, type 1 is larger
Phi1 = lambda s1: mn.Phi(np.array([s1, 1 - s1]))[0]          # two platforms: s_2 = 1 - s, a scalar map

def equilibria(mn, grid=np.linspace(1e-9, 1 - 1e-9, 2001)):
    '''All equilibria s of the two-platform market, with the slope of Phi at each'''
    g = np.array([Phi1(x) for x in grid]) - grid
    eq = [brentq(lambda x: Phi1(x) - x, a, b, xtol=1e-14)      # bracket every sign change, then refine
          for a, b, ga, gb in zip(grid[:-1], grid[1:], g[:-1], g[1:]) if ga * gb < 0]
    slope = [jacobian(mn, np.array([x, 1 - x]))[0, 0] * 2 for x in eq]   # dPhi/ds, s_2 = 1 - s
    return np.array(eq), np.array(slope)
```

The function `equilibria` does not use successive approximations: it brackets every
sign change of $\Phi_1(s) - s$ on a grid and refines it with `brentq`, which finds
unstable equilibria as well. That is possible because the problem is one-dimensional.

```{code-cell} python3
:tags: [hide-input]

sg = np.linspace(0, 1, 401)
fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True, layout='constrained')
for ax, lam in zip(axes.flat, (1, 2, 3, 4)):
    mn.lam = lam
    eq, slope = equilibria(mn)
    ax.plot(sg, sg, color='0.5', lw=1, label='45-degree line')
    ax.plot(sg, [Phi1(x) for x in sg], color='tab:blue', lw=1.8, label='$\\Phi_1(s)$')
    for x, d in zip(eq, slope):                               # filled: stable, |slope| < 1; open: unstable
        ax.plot(x, x, 'o', ms=9, color='tab:blue', mfc='tab:blue' if abs(d) < 1 else 'white', mew=2)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', title=f'λ = {lam}')
    ax.legend(loc='upper left'); ax.grid(color='0.9')
    print(f'λ = {lam}: equilibria s = {eq.round(4)}, slope of Phi {slope.round(3)}')
fig.supxlabel('$s$'); fig.supylabel('$\\Phi_1(s)$')
fig.suptitle('Equilibria of the two-platform market')
plt.show()
```

What the figures show, with filled markers for the stable equilibria and open ones for the
unstable:

- with $\lambda = 1$ the shares follow the preferences: platform 1 gets 58% of the market,
  a little less than its 60% of loyal consumers would suggest, because of the taste shocks
- with $\lambda = 2$, at the bound on the slope, the equilibrium is still unique: platform 1
  has 68%, and the slope there is $0.70$, well below one
- with $\lambda = 3$ the market has **tipped**: one equilibrium, with 90% on platform 1
- with $\lambda = 4$ there are three equilibria: platform 1 dominates with 97%, platform 2
  dominates with 96%, or the market splits with $s = 0.42$

In the second stable equilibrium the platform preferred by the minority wins almost the
whole market, a **coordination failure**: every consumer of type 1 would be better off if
all of them switched together, and none of them is better off switching alone. The split
equilibrium has slope $1.6$ and is unstable.

(eqb-sa-multiple)=
## Successive approximations with multiple equilibria

The **cobweb diagram** draws the SA iterations: 
- up to the curve $\Phi$ $\to$ computing the operatore,
- across to the 45-degree line $\to$ updating the argument to the new value of $s$
- repeat until convergence.
  
```{code-cell} python3
:tags: [hide-input]

def cobweb(ax, path, col, label):
    '''Draw the steps of successive approximations on the 45-degree diagram'''
    x = [p[0] for p in path]
    xs, ys = [x[0]], [0.0]                                    # the first step starts from the horizontal axis
    for a, b in zip(x[:-1], x[1:]):
        xs += [a, b]; ys += [b, b]                            # up to the curve, across to the 45-degree line
    ax.plot(xs, ys, color=col, lw=1, label=label)
    for xa, ya, xb, yb in zip(xs[:-1], ys[:-1], xs[1:], ys[1:]):
        if np.hypot(xb - xa, yb - ya) < 0.03: continue        # no arrows on the tiny steps near the fixed point
        xm, ym = (xa + xb) / 2, (ya + yb) / 2                 # open arrowhead in the middle, showing the direction
        ax.annotate('', xy=(xm + 0.01 * (xb - xa), ym + 0.01 * (yb - ya)), xytext=(xm, ym),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1, shrinkA=0, shrinkB=0))

mn.lam = 4
eq, slope = equilibria(mn)
fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
ax.plot(sg, sg, color='0.5', lw=1)
ax.plot(sg, [Phi1(x) for x in sg], color='black', lw=1.8, label='$\\Phi_1(s)$, λ = 4')
for s0, col in ((0.40, 'tab:red'), (0.45, 'tab:blue'), (0.63, 'tab:green')):
    path = [np.array([s0, 1 - s0])]
    s = solve_sa(mn.Phi, path[0], callback=lambda **kw: path.append(kw['x']))
    cobweb(ax, path, col, f'$s^0$ = {s0}: {len(path) - 1} iterations to $s$ = {s[0]:.4f}')
for x, d in zip(eq, slope):                                   # filled: stable, |slope| < 1; open: unstable
    ax.plot(x, x, 'o', ms=10, color='black', mfc='black' if abs(d) < 1 else 'white', mew=2,
            label=f'{"stable" if abs(d) < 1 else "unstable"} equilibrium {x:.4f}')
ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xlabel='$s$', ylabel='$\\Phi_1(s)$',
       title='Successive approximations with three equilibria')
ax.legend(loc='upper left'); ax.grid(color='0.9')
plt.show()
```

The unstable equilibrium splits the interval into two **basins of attraction**. Starting at
$0.40$ the iterations move away from $0.4227$ down to the low equilibrium, starting at
$0.45$ up to the high one. No starting value other than $0.4227$ itself leads to the split
equilibrium.

The same experiment for a range of $\lambda$, with all the equilibria computed by
`equilibria` and the limits of successive approximations started from $s^0 = 0$ and
$s^0 = 1$:

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
        s = solve_sa(mn.Phi, [s0, 1 - s0], callback=lambda **kw: path.append(kw['x']))
        iters[s0].append(len(path))
        if col and lam in lams[::4]: ax.plot(lam, s[0], 'o', mfc='none', color=col, ms=9)
print(f'three equilibria from λ = {lams[np.argmax(n3)]:.2f} on, on this grid of λ')
ax.plot([], [], '.', color='black', label='stable equilibria')
ax.plot([], [], '.', color='tab:red', label='unstable equilibria')
ax.plot([], [], 'o', mfc='none', color='tab:purple', label='SA from $s^0 = 0$')
ax.plot([], [], 'o', mfc='none', color='tab:green', label='SA from $s^0 = 1$')
ax.set(xlabel='strength of the network effect λ', ylabel='$s$', title='Equilibrium market share of platform 1')
ax.legend(loc='center right'); ax.grid(color='0.9')
plt.show()
```

- Below $\lambda \approx 3.1$ the equilibrium is unique, and the share of platform 1 rises
  with $\lambda$ as the market tips towards it.
- At $\lambda \approx 3.1$ a pair of new equilibria appears, a stable one with platform 2
  dominant and the unstable one between them.
- Successive approximations trace the stable branches only. Which branch they find depends
  on the starting value: from $s^0 = 0$ the low one, from $s^0 = 1$ or from equal
  shares the high one. The red branch is invisible to them.


(eqb-congestion)=
### Congestion: negative feedback

With $\lambda < 0$ a larger platform is less attractive, for example because of
congestion. $\Phi_1$ is then decreasing in $s$, it crosses the 45-degree line exactly
once, and the equilibrium is unique. Successive approximations can still fail:

```{code-cell} python3
mn.lam = -4                                                   # congestion: a larger platform is less attractive
eq, slope = equilibria(mn)
print(f'λ = {mn.lam}: equilibrium s = {eq.round(4)}, slope of Phi {slope.round(3)}')
try:
    solve_sa(mn.Phi, [0.5, 0.5])
except RuntimeError as e:
    print('plain SA:', e)
```

```{code-cell} python3
:tags: [hide-input]

path = [np.array([0.5, 0.5])]
for k in range(30): path.append(mn.Phi(path[-1]))             # plain SA, stopped after 30 steps
fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
ax.plot(sg, sg, color='0.5', lw=1)
ax.plot(sg, [Phi1(x) for x in sg], color='black', lw=1.8, label='$\\Phi_1(s)$, λ = -4')
cobweb(ax, path, 'tab:red', f'plain SA: a cycle between {path[-2][0]:.3f} and {path[-1][0]:.3f}')
ax.plot(eq[0], eq[0], 'o', ms=10, color='black', mfc='white', mew=2, label=f'unstable equilibrium {eq[0]:.4f}')
ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xlabel='$s$', ylabel='$\\Phi_1(s)$',
       title='Successive approximations with congestion')
ax.legend(loc='upper right'); ax.grid(color='0.9')
plt.show()
```

The slope of $\Phi$ at the equilibrium is $-1.59$, so every iteration overshoots by more
than it corrects. The iterations spiral out into a **cycle** between $0.04$ and $0.97$:
the crowd leaves the congested platform for the empty one, which is then the congested one.

(eqb-damping)=
## Damping

**Damping** takes only a step of size $w$ towards $\Phi(x)$ instead of the full step:

$$
x_{k+1} = (1 - w) x_k + w \Phi(x_k), \qquad 0 < w \leqslant 1 .
$$

The damped map has the same fixed points as $\Phi$, and its slope at a fixed point is
$1 - w + w \Phi'(s^\star)$.

```{code-cell} python3
def damped(Phi, w):
    '''Damped operator: a step of size w from x towards Phi(x); w = 1 is plain successive approximations'''
    return lambda x: (1 - w) * x + w * Phi(x)

path_d = [np.array([0.2, 0.8])]
s = solve_sa(damped(mn.Phi, 0.3), path_d[0], callback=lambda **kw: path_d.append(kw['x']))
print(f'damped SA, w = 0.3: s = {s[0]:.4f} in {len(path_d) - 1} iterations')
```

```{code-cell} python3
:tags: [hide-input]

Phi1_w = lambda s1: 0.7 * s1 + 0.3 * Phi1(s1)                 # the damped map, w = 0.3
fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
ax.plot(sg, sg, color='0.5', lw=1)
ax.plot(sg, [Phi1(x) for x in sg], color='black', lw=1.8, label='$\\Phi_1(s)$, λ = -4')
ax.plot(sg, [Phi1_w(x) for x in sg], '--', color='tab:blue', lw=1.8, label='damped map, w = 0.3')
cobweb(ax, path_d, 'tab:blue', f'damped SA: {len(path_d) - 1} iterations to {path_d[-1][0]:.4f}')
ax.plot(eq[0], eq[0], 'o', ms=10, color='black', mfc='white', mew=2, label=f'equilibrium {eq[0]:.4f}')
ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xlabel='$s$', ylabel='$\\Phi_1(s)$',
       title='Damped successive approximations')
ax.legend(loc='upper right'); ax.grid(color='0.9')
plt.show()
```

With $w = 0.3$ the slope at the equilibrium of the congestion example is $0.22$, and the
iterations converge in 16 steps from $s^0 = 0.2$. The slope is zero with $w = 1 / (1 - \Phi'(s^\star))$, a
Newton step in one dimension.

Damping does not help with the unstable equilibria of the
[market with a positive network effect](#eqb-sa-multiple): their slope is above one, and
$1 - w + w \Phi'$ is then above one for every $w$ between zero and one. A solver that uses
the derivative, such as [Newton's method of Class 5](5_solvers.md#multivariate-newton),
finds them.


(eqb-all-equilibria)=
## How to find all equilibria?

No method is guaranteed to find all solutions of a general system of equations, and a
**poly-algorithm**, a combination of methods that cover each other's failures, is often a
good idea. The possible approaches:

- **Grid and Brent's method.** The function `equilibria` above evaluates $\Phi_1(s) - s$ on
  a fine grid and refines every sign change with
  [`brentq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html).
  Brent's method combines bisection with secant and inverse quadratic interpolation steps,
  so it keeps the bracket and converges superlinearly. It finds stable and unstable
  equilibria alike. It misses two equilibria that fall between neighboring grid points, and
  a fine grid is only possible in one dimension.
- **Successive approximations and bracketing.** With $\lambda > 0$ the map $\Phi_1$ is
  increasing, so successive approximations from $s^0 = 0$ and from $s^0 = 1$ reach the lowest
  and the highest equilibria. If they reach the same point, the equilibrium is unique.
  Otherwise $\Phi_1(s) - s$ changes sign between them, and bisection finds an unstable
  equilibrium in between. This needs no grid, but with $m = 3$ platforms there is no
  interval with two ends to start from and to bracket.
- **Newton's method with multistart.** Run
  [Newton's method](5_solvers.md#multivariate-newton) on $\Phi(s) - s = 0$ from many starting
  values. It converges to stable and unstable equilibria alike and works in any dimension,
  but there is **no guarantee**: nothing tells that the starting values covered the basin of
  attraction of every equilibrium.
- **Piecewise linear approximation.** In a game of two players every equilibrium is an
  intersection of the two best response curves, as [Class 23](23_entry_games.md) shows for an
  entry game. Approximate each curve by a polyline, take the intersections of the segments as
  candidate equilibria, and add points near the candidates until they stop moving. This
  handles any number of equilibria where the curves cross rather than touch, in two
  dimensions. It becomes numerically unstable when the best responses are close to 0 or 1,
  where the curves are nearly flat or vertical. The figure shows a stage game of the
  dynamic Bertrand game of [Class 25](25_rls.md) with five equilibria:

  ```{image} _static/games/leap_5stage_eqb.png
  :width: 80%
  :align: center
  :alt: A stage game with five equilibria found by piecewise linear approximation
  ```

- **Homotopy method.** Deform an easy problem into the problem of interest and follow the
  path of solutions along the way, including its turning points
  {cite:p}`borkovsky2008UsersGuideSolving`. For example, raise $\lambda$ from zero, where the
  equilibrium is unique. The method follows one path, and equilibria off the path stay
  unknown: in the [figure above](#eqb-sa-multiple) the path from $\lambda = 0$ follows the high
  branch, and never reaches the pair of equilibria that appears at $\lambda \approx 3.1$.
- **Algebraic methods.** When the equilibrium conditions can be written as polynomial
  equations, Gröbner bases and all-solution homotopy methods find every solution
  {cite:p}`juddFindingAllPureStrategy2012,kubler2014ComputingAllSolutions`. The number of
  solutions to track grows quickly with the number of equations, and the logit shares of the
  platform market are not polynomial, so the approach does not apply to it directly.


````{note} References and additional resources

(eqb-practice-references)=
- 📖 {cite:t}`brock2001DiscreteChoiceSocial` "Discrete Choice with Social Interactions"
- 📖 {cite:t}`katz1985NetworkExternalitiesCompetition` "Network Externalities, Competition, and Compatibility"
- 📖 {cite:t}`rochet2003PlatformCompetitionTwosided` "Platform Competition in Two-Sided Markets"
- 📖 {cite:t}`judd1998NumericalMethodsEconomics` "Numerical Methods in Economics", on fixed-point iteration and its acceleration
- 📖 {cite:t}`borkovsky2008UsersGuideSolving` "A User's Guide to Solving Dynamic Stochastic Games Using the Homotopy Method"
- 📖 {cite:t}`juddFindingAllPureStrategy2012` "Finding All Pure-Strategy Equilibria in Games with Continuous Strategies"
- 📖 {cite:t}`kubler2014ComputingAllSolutions` "Computing All Solutions to Polynomial Equations in Economics"
- 📺 The lecture on successive approximations in *Foundations of Computational Economics*, the source of the platform example [youtu.be/AQt9Q9qc3io](https://youtu.be/AQt9Q9qc3io)
````
