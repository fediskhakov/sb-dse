---
title: 📖 Cake eating on and off a grid and function approximation
short_title: 📖 Cake eating
subtitle: Class 15 — Thursday, October 15
downloads:
  - file: 15_cake_eating.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

So far every dynamic choice in the course was discrete, and the maximum in the Bellman equation
was taken over a short list. With a continuous choice the maximum is an optimization
problem in every state! We will learn a few solution techniques for the models with continuous 
choice in the next few classes before addressing the estimation of these models.
Along the way we need to remind ourselves of function approximation.

````{seealso} Key reading

{cite:t}`deaton1991SavingLiquidityConstraints` "Saving and Liquidity Constraints",
*Econometrica* 59(5), 1221–1248. The consumption-savings model with a borrowing
constraint and stochastic income, which this class and the next three solve.

````

````{hint} Running the code for this lecture
:class: dropdown

The code for this class is in the folder `session15-oct15/` of the course **code
repository**.

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

````{danger} Homework iota_deaton: the Deaton model with discretized choice
:class: dropdown

(homework-iota_deaton)=
Graded homework: task `iota_deaton` in the class repository. Collect it and work on a
copy in your own repository:

```bash
git pull upstream main                         # collect the task
cp -r tasks/iota_deaton solutions/iota_deaton  # work on the copy
```

Implement the consumption-savings model with the formulas of this class and solve it
with the two solvers with discretized choice, [on the grid](#cake-ongrid) and with a
[separate choice grid](#cake-discretized). Check both solutions with the
[two accuracy checks](#accuracy-checks), constant simulated consumption and the Euler
equation residuals, explain what they show, and find settings of each solver for which
it passes both. Then compare the two methods.

The notebook `deaton.ipynb` in the task folder carries the details of the task, step by
step.

Submit it as a pull request, following the [git workflow](2_workflow.md#submission). One
of you presents a solution at the start of the next Tuesday class.
````

## Cake eating

The cake eating problem has the structure of the all so important consumption-savings model and an analytic solution, so every numerical method can be checked against the truth.

- **Choice:** how much cake $c_t$ to eat in period $t = 0, 1, \dots, \infty$
- **State:** the size of the cake $W_t$, with $W_0$ given
- **Transition:** what is not eaten is left for the future, $W_{t+1} = W_t - c_t$
- **Preferences:** flow utility $u(c_t) = \log(c_t)$ and discount factor $\beta \in (0,1)$

The consumer maximizes the discounted utility over the whole horizon,

$$
\max_{\{c_t\}_{t=0}^{\infty}} \sum_{t=0}^{\infty} \beta^t u(c_t),
\qquad 0 \leqslant c_t \leqslant W_t.
$$

Look how easy it is to get the familiar Bellman equation in this case. Let

$$
V_t(W_t) = \max_{\{0 \leqslant c_\tau \leqslant W_\tau\}_{\tau=0}^{\infty}} \sum_{\tau=0}^{\infty} \beta^\tau u(c_\tau) =
$$
$$
= \max_{0 \leqslant c_t \leqslant W_t} \Big(
u(c_t) + \beta \max_{\{0 \leqslant c_\tau \leqslant W_\tau-c_t\}_{\tau=1}^{\infty}}
\sum_{\tau=1}^{\infty} \beta^\tau u(c_\tau) 
\Big) =
$$
$$
= \max_{0 \leqslant c_t \leqslant W_t} \Big(
u(c_t) + \beta V_{t+1}(W_t-c_t)
\Big)
$$

The last equation is due to the fact that infinitely many time periods look the same from the point of view of today and tomorrow.
Realizing further that in this case nothing can depend on index of time, we can drop the time index to arrive at

$$
V(W) = \max_{0 \leqslant c \leqslant W} \big\{ u(c) + \beta V(W - c) \big\}
$$

The right-hand side defines the Bellman operator $T(V)(W)$ from space of value functions to itself

- operator $T$ is a contraction by Blackwell's sufficient conditions, as in [Class 8](8_dp_infinite.md)
- therefore the value function iterations (VFI) $V_{i} = T(V_{i-1})$ converge to the unique solution from any starting point

The question of this class is how to *compute* $T(V)$ when $c$ and $W$ are continuous.


:::{div}
:class: discussion

- Classify this dynamic optimization problem by type (see the
  [classification in Class 7](7_dp.md#dp-classification))
- What is different from the models you saw in macro economics?
- Can we solve this analytically?
:::


### The analytic solution

Let $u(c) = \log(c)$, what is the solution?

$$
c^\star(W) = (1-\beta)\, W,
\qquad
V(W) = \frac{\log W}{1-\beta} + \frac{\log(1-\beta)}{1-\beta} + \frac{\beta \log\beta}{(1-\beta)^2}
$$

The consumer eats a constant fraction $1-\beta$ of what is left, every period.

````{note} Derivation by guess and verify
:class: dropdown

A guess $V(W) = A + B \log W$ and verify!

Because $\log(c) \to -\infty$ when $c \to 0$ (Inada condition), the lower bound never binds. But so the upper bound never binds either as long as $\beta>0$. (why?)

Substitute the guess into the Bellman equation:

$$
A + B\log W = \max_{c} \big\{\log c + \beta \big(A + B\log(W - c)\big)\big\}
$$

The first order condition $1/c = \beta B/(W - c)$ gives
$c = W/(1+\beta B)$ and $W - c = \beta B W/(1+\beta B)$. Substituting back,

$$
A + B\log W = \log W - \log(1+\beta B) + \beta A + \beta B\log W + \beta B\log\frac{\beta B}{1+\beta B}
$$

Matching the coefficients on $\log W$ gives $B = 1 + \beta B$, so $B = 1/(1-\beta)$
and $c^\star(W) = (1-\beta)W$. Matching the constants gives $A$.
````

The analytic solution exists only because the model is so simple
- only few models admit an analytical solution
- more complicated models require backward induction by hand + spotting the regularities before actual guessing

Its value, at least here, as well as any analytic property is the ability to verify numerical methods

:::{div}
:class: discussion

- Would the same approach be possible with $u(c) = \sqrt{c}$?
- The Bellman operator is a contraction. Given this, how should we compute the maximum on its right-hand side?
:::

(cake-ongrid)=
## Solving on a grid

The simplest numerical approach replaces both the state and the choice with a finite
grid, $\vec{W} = \{W_1, \dots, W_n\}$ on $[0, \overline{W}]$. To avoid evaluating $V$
between grid points, make the *next period cake* the decision variable so we have:

$$
V_i(W_j) = \max_{W_k \leqslant W_j} \big\{ u(W_j - W_k) + \beta V_{i-1}(W_k) \big\},
\qquad j = 1, \dots, n
$$

- here $i$ denotes the iteration of the VFI algorithm

Every term is known on the grid, so one iteration is a maximum over an $n \times n$
matrix of choices, exactly as in the discrete models of the earlier classes:

```{code-cell} python3
import numpy as np
import matplotlib.pyplot as plt

class cake_ongrid():
    '''Cake eating on a grid: the next period cake is chosen from the state grid'''

    def __init__(self, beta=.9, Wbar=10, ngrid=50):
        self.beta = beta                               # discount factor
        self.Wbar = Wbar                               # upper bound on cake size
        self.ngrid = ngrid                             # number of grid points
        self.epsilon = np.finfo(float).eps             # smallest positive float
        self.grid = np.linspace(self.epsilon, Wbar, ngrid)   # grid for both state and choice

    def bellman(self, V0):
        '''Bellman operator, V0 is a vector of values on the grid'''
        c = self.grid - self.grid[:, np.newaxis]       # states in columns, next cake in rows
        c[c == 0] = self.epsilon                       # avoid log(0) when nothing is eaten
        mask = c > 0                                   # feasible choices: eat a positive amount
        matV1 = np.full((self.ngrid, self.ngrid), -np.inf)   # -inf marks infeasible choices
        matV0 = np.repeat(V0.reshape(self.ngrid, 1), self.ngrid, 1)   # V0 of the next cake, in rows
        matV1[mask] = np.log(c[mask]) + self.beta * matV0[mask]       # maximand of the Bellman equation
        V1 = np.amax(matV1, axis=0)                    # maximum in every column
        c1 = self.grid - self.grid[np.argmax(matV1, axis=0)]          # optimal consumption
        return V1, c1

    def solve(self, maxiter=1000, tol=1e-4, callback=None):
        '''Value function iterations'''
        V0 = np.log(self.grid)                         # start from eating everything
        for i in range(maxiter):
            V1, c1 = self.bellman(V0)
            if callback: callback(iter=i, grid=self.grid, value=V1, policy=c1)
            if np.all(abs(V1 - V0) < tol):
                break
            V0 = V1
        else:
            raise RuntimeError('No convergence: maximum number of iterations achieved')
        return V1, c1
```

```{code-cell} python3
:tags: [hide-input]

def check_analytic(model, title=''):
    '''Plot the numerical solution of a cake eating model against the analytic one'''
    b = model.beta
    aV = lambda w: np.log(w)/(1-b) + np.log(1-b)/(1-b) + b*np.log(b)/(1-b)**2
    aP = lambda w: (1-b)*w
    xg = np.linspace(model.Wbar/1000, model.Wbar, 1000)   # dense grid for the analytic solution
    V, policy = model.solve()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(model.grid[1:], V[1:], label='numerical')
    ax1.plot(xg, aV(xg), label='analytic')
    ax2.plot(model.grid, policy, label='numerical')
    ax2.plot(xg, aP(xg), label='analytic')
    for ax, t in ((ax1, 'Value function'), (ax2, 'Policy function')):
        ax.set(xlabel='Cake size, W', title=t)
        ax.grid(color='0.9')
        ax.legend()
    fig.suptitle(title)
    plt.show()
    return np.max(np.abs(policy - aP(model.grid)))

err = check_analytic(cake_ongrid(beta=0.9, Wbar=10, ngrid=100), 'Same grid for state and choice, 100 points')
print(f'Largest error in consumption: {err:.3f}')
```

The value function is close, but the policy is a staircase. On this grid the consumer
can only eat the *difference of two grid points*, so with 100 points on $[0,10]$ every
consumption level is a multiple of about $0.1$, whatever the true optimum.

Where it breaks: the choice set is tied to the state grid. Refining the state grid is
the only way to refine the choices, the cost of one iteration grows with $n^2$, and in
a model where next period's state is not a grid point at all, the trick is not
available. That is the case as soon as savings earn interest or income is random.

:::{div}
:class: discussion

- Why is the error largest at small cakes, relative to the size of the cake?
- With interest, $W_{t+1} = R(W_t - c_t)$ for $R > 1$, do we have a solver in which the problem can be converted in the same way?
:::

## Function approximation

Any method that separates the choice from the state grid needs the value function
between grid points. The **interpolation problem**: given nodes
$a = x_0 < x_1 < \dots < x_n = b$ and the values $f(x_i)$ of an unknown function, find a
function $s$ from a chosen class with $s(x_i) = f(x_i)$ for all $i$, and use $s(x)$ in
place of $f(x)$ at any $x \in [a,b]$.

- **Interpolation** matches the data *exactly*: the approximation passes through every
  $(x_i, f(x_i))$. This is what a solver needs, because the data are computed exactly.
- **Curve fitting** finds the central tendency of noisy data, as in regression, with
  more data points than parameters. Better fitted for the situations with (potential) measurement errors.
- **Extrapolation**, evaluating the approximation/fitted curve outside the initial domain $[a, b]$, should be avoided unless the theoretical properties of $f$ are known there. A wider grid is almost always the better fix.

The quality of an approximation is measured in the **uniform norm** aka **infinity norm**,
$\lVert f - s \rVert_\infty = \max_{x \in [a,b]} |f(x) - s(x)|$, the largest error
anywhere on the interval, which is what matters when $s$ stands in for $f$ inside a
solver. 

There are two families of methods: 
- connect the dots piece by piece, or 
- replace $f$ by one simpler function on the whole interval (which could still connect all dots to *interpolate*)

### Splines

````{attention} Definition

A function $s(x)$ on $[a,b]$ is a **spline of order $k$** (degree $k-1$) with knots
$a = x_0 < x_1 < \dots < x_n = b$ if $s$ is a polynomial of degree $k-1$ on each
interval $[x_i, x_{i+1}]$, passes through each knot and has continuous derivatives up to order $k-2$ on $[a,b]$.
````

- *linear splines*: order 2, degree 1, continuous
- *quadratic splines*: order 3, degree 2, continuous first derivative
- *cubic splines*: order 4, degree 3, continuous first and second derivatives

To compute spline interpolation one would need to solve the following system:
- passing through both ends on $n$ intervals = $2n$ equations
- continuous derivatives on $n-1$ interior knots = $(n-1)(k-2)$ equations
- number of coefficients for one interval = $k$
- total number of unknowns = $nk$
- $2n+(n-1)(k-2)=nk-k+2$ equations
- $k-2$ additional *boundary* conditions on the derivatives at $a$ and $b$

**Linear interpolation** is the spline of order 2: continuous, with kinks at the knots.

- on $[x_i, x_{i+1}]$ it is
$s(x) = f(x_i) + \frac{f(x_{i+1}) - f(x_i)}{x_{i+1} - x_i}(x - x_i)$, and 
- for a twice continuously differentiable $f$ and the largest spacing $h = \max_i (x_{i+1} - x_i)$,

$$
\lVert f - s \rVert_\infty \leqslant \frac{h^2}{8} \lVert f'' \rVert_\infty
$$

Halving the spacing divides the error by four.

The **cubic spline**, need two boundary conditions
- $s''(a) = s''(b) = 0$ for the *natural* spline, or 
- the slopes $s'(a) = f'(a)$, $s'(b) = f'(b)$ for the *clamped* spline. 

The coefficients solve a tridiagonal linear system, at a cost
proportional to $n$. 

For a clamped spline and $f$ with four continuous derivatives,

$$
\lVert f - s \rVert_\infty \leqslant \frac{5}{384}\, h^4 \lVert f^{(4)} \rVert_\infty
$$

so halving the spacing divides the error by sixteen.

In Python both are available: 
- `np.interp` for the linear spline
- `scipy.interpolate.CubicSpline` for the cubic spline

(polynomial-approximation)=
## Polynomial approximation

A polynomial of degree $n$ through $n+1$ distinct nodes exists and is unique. 

How good it is (in terms of approximation error) depends on where the nodes are.

````{attention} Theorem (interpolation error)

Let $f$ have $n+1$ continuous derivatives on $[a,b]$, and let $p_n$ be the polynomial of
degree $n$ that interpolates $f$ at the nodes $x_0, \dots, x_n \in [a,b]$. Then for every
$x \in [a,b]$ there is $\xi \in [a,b]$ such that

$$
f(x) - p_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!} \prod_{i=0}^{n} (x - x_i)
$$
````

The error has two factors. 
- first depends only on $f$. 
- second, the **node polynomial** $\prod_i (x - x_i)$, depends only on the nodes, and it is the one we control.

1. With **equally spaced nodes** the node polynomial is small in the middle of the
  interval and grows rapidly towards its ends. 
  
  For functions whose derivatives grow
  fast, such as $f(x) = 1/(1+25x^2)$ on $[-1,1]$, the error near the ends grows without
  bound as $n$ increases: the **Runge phenomenon**. More nodes make the approximation
  worse.

2. The nodes that make $\max_{x \in [-1,1]} \big|\prod_i (x - x_i)\big|$ smallest are the
  zeros of the *Chebyshev polynomial* $T_{n+1}$, where the maximum equals $2^{-n}$, its
  smallest possible value. 
  
  The error bound becomes
  $\lVert f - p_n \rVert_\infty \leqslant \dfrac{\lVert f^{(n+1)} \rVert_\infty}{2^n (n+1)!}$.

The **Chebyshev polynomials** are

$$
T_k(x) = \cos\big(k \arccos x\big),
$$
or equivalently
$$
T_0(x) = 1,\; T_1(x) = x,\quad T_{k+1}(x) = 2x\,T_k(x) - T_{k-1}(x),
$$

The $n+1$ zeros of $T_{n+1}$, **Chebyshev nodes**, are

$$
z_j = \cos\big(\tfrac{2j+1}{2(n+1)}\pi\big), j = 0, \dots, n
$$ 

They cluster towards the ends of $[-1,1]$. 

The polynomials are orthogonal with respect to the weight
$(1-z^2)^{-1/2}$, and also *discretely* orthogonal on the Chebyshev nodes, so the
interpolating polynomial $p_n(z) = \sum_{k=0}^{n} c_k T_k(z)$ has explicit coefficients,
no linear system needed:

$$
c_k = \frac{2 - \mathbb{1}\{k = 0\}}{n+1} \sum_{j=0}^{n} f(z_j)\, T_k(z_j)
$$

A general interval $[a,b]$ is mapped to $[-1,1]$ by $z = 2(x-a)/(b-a) - 1$, as in the
[flexible logit of Class 13](13_ccp_practice.md#ccp-first-stage). 

For $f$ with $k$
continuous derivatives, Chebyshev interpolation of degree $n$ satisfies
{cite:p}`judd1998NumericalMethodsEconomics`

$$
\lVert f - p_n \rVert_\infty \leqslant \Big(\frac{2}{\pi}\log(n+1) + 1\Big)
\frac{(n-k)!}{n!} \Big(\frac{\pi}{2}\Big)^k \frac{(b-a)^k}{2^k} \lVert f^{(k)} \rVert_\infty,
$$

which falls faster than any power of $n$ when $f$ is smooth: Chebyshev interpolation is
within a logarithmic factor of the best polynomial approximation in the uniform norm.

The four methods side by side, first with 12 nodes on a function with some curvature,
then with a growing number of nodes on the Runge function:

```{code-cell} python3
:tags: [hide-input]

from scipy import interpolate
from numpy.polynomial import polynomial, chebyshev
import warnings

def approximations(f, a, b, n):
    '''Linear spline, cubic spline, polynomial on equal nodes and Chebyshev interpolation with n nodes'''
    xe = np.linspace(a, b, n)                          # equally spaced nodes
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')                # polyfit warns about poor conditioning
        p = polynomial.polyfit(xe, f(xe), n-1)
    return {'linear spline':           lambda x: np.interp(x, xe, f(xe)),
            'cubic spline':            interpolate.CubicSpline(xe, f(xe)),
            'polynomial, equal nodes': lambda x: polynomial.polyval(x, p),
            'Chebyshev, own nodes':               chebyshev.Chebyshev.interpolate(f, n-1, domain=[a, b])}

f = lambda x: np.exp(-x/4)*np.sin(x) + 1/(1 + x**2)   # function to approximate
a, b, n = -5, 10, 12                                   # interval and number of nodes
xd = np.linspace(a, b, 1000)                           # dense grid to measure errors
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))
ax1.plot(xd, f(xd), 'k', lw=2, label='f(x)')
ax1.scatter(np.linspace(a, b, n), f(np.linspace(a, b, n)), c='k', zorder=3)
for name, g in approximations(f, a, b, n).items():
    ax1.plot(xd, g(xd), label=name)
    ax2.semilogy(xd, np.abs(g(xd) - f(xd)) + 1e-16, label=name)
ax1.set(ylim=(-1.5, 2), title=f'{n} nodes')
ax2.set(ylim=(1e-6, 15), title='Absolute error')
for ax in (ax1, ax2):
    ax.legend(fontsize=8); ax.grid(color='0.9')
plt.show()
```
In economic models we control the three things that determine the accuracy:

- **the number of nodes**, more is better, at a cost in every iteration of the solver
- **the location of the nodes**, denser where the function curves more: the error
  bounds scale with the derivatives of $f$, which for the log-utility value function
  are largest near zero wealth
- **the type of the approximation**, matched to the function: smooth functions favor
  Chebyshev polynomials, functions with kinks favor splines, because the bounds above
  need the derivatives to exist

Where it breaks: 
- polynomials and cubic splines do not preserve monotonicity or
concavity, so an approximated value function can have a local maximum that the true one does not
- linear interpolation preserves both, which is why it is the default in the
solvers below, despite its lower accuracy. 
- in several dimensions all of these methods generalize as tensor products, but the number of nodes grows exponentially with the dimension. `Curse of dimentionality`
- taking derivatives of the approximations usually make little sense, much better to compute and interpolate the derivative function ⚠️


(cake-discretized)=
## Discretized choice with interpolation

With an interpolated value function the choice no longer has to live on the state grid.

Keep the grid $\vec{W}$ for the state, and give the choice its own grid
$\vec{D}(W_j)$ for every state point. 

Scale to the feasible interval $[0, W_j]$ so that
the constraint $0 \leqslant c \leqslant W$ is respected by construction:

$$
V_i(W_j) = \max_{c \in \vec{D}(W_j)} \big\{ u(c) + \beta \hat V_{i-1}(W_j - c) \big\}
$$

where $\hat V_{i-1}$ is the interpolated value function and $i$ denotes the VFI iteration index

:::{div}
:class: discussion

- Where in the code below are the grids?
- Why not use common grid over choices?
- Where is interpolation in the code and which algorithm is used?
:::


```{code-cell} python3
class cake_discretized():
    '''Cake eating with a separate choice grid and interpolated value function'''

    def __init__(self, beta=.9, Wbar=10, ngrid=50, nchgrid=100):
        self.beta = beta                               # discount factor
        self.Wbar = Wbar                               # upper bound on cake size
        self.ngrid = ngrid                             # number of points on the state grid
        self.nchgrid = nchgrid                         # number of points on the choice grid
        self.epsilon = np.finfo(float).eps             # smallest positive float
        self.grid = np.linspace(self.epsilon, Wbar, ngrid)   # grid for the state
        self.share = np.linspace(self.epsilon, 1, nchgrid)   # choices as shares of the cake

    def bellman(self, V0):
        '''Bellman operator, V0 is a vector of values on the state grid'''
        c = self.share[:, np.newaxis] * self.grid      # choices in rows, scaled to [0,W] in each column
        W1 = self.grid - c                             # next period cake, off the grid in general
        matV1 = np.log(c) + self.beta * np.interp(W1, self.grid, V0)   # linear interpolation of V0
        i = np.argmax(matV1, axis=0)                   # optimal choice in every column
        return matV1[i, np.arange(self.ngrid)], c[i, np.arange(self.ngrid)]

    solve = cake_ongrid.solve                          # the same value function iterations
```

```{code-cell} python3
:tags: [hide-input]

err = check_analytic(cake_discretized(beta=0.9, Wbar=10, ngrid=100, nchgrid=100),
                     'Separate choice grid, 100 + 100 points')
print(f'Largest error in consumption: {err:.3f}')
print(f'{"state grid":>10} {"choice grid":>12} {"max error in c":>15}')
for ng, nch in ((50, 50), (100, 100), (100, 500), (500, 100), (500, 500)):
    m = cake_discretized(beta=0.9, Wbar=10, ngrid=ng, nchgrid=nch)
    V, c = m.solve()
    print(f'{ng:>10} {nch:>12} {np.max(np.abs(c - (1-m.beta)*m.grid)):>15.4f}')

```

The policy is now five times closer to the truth with the same number of points. The
table shows that both grids matter. A choice grid of $n_c$ shares pins consumption down
to within $W/n_c$, but here the grid of shares happens to contain a point within $0.001$
of the optimal share $1-\beta$, so most of the remaining error comes from the
interpolation of $\hat V$ on the state grid. The error is not even monotone in the grid
sizes, a warning against reading accuracy off a single run.

How to increase the accuracy further:

- more points on both grids, at a cost of $n \times n_c$ evaluations per iteration
- choice grids that respect the constraints of the model, as above
- state grid points relocated to where the value function curves most
- a better approximation of $\hat V$, or no discretization of the choice at all: the
  next class maximizes over a continuous $c$ in every state, and solves the first order
  condition instead

:::{div}
:class: discussion

- Linear interpolation of $V$ under-states it between the grid points, because $V$ is
  concave. Which consumption choices does that bias favor?
- Why does the error not disappear when the state grid is refined with the choice grid
  held fixed?
:::

## The consumption-savings model

The cake eating model becomes the workhorse consumption-savings model of
{cite:t}`deaton1991SavingLiquidityConstraints` once savings earn interest and the
consumer has an income:

$$
V(M) = \max_{0 \leqslant c \leqslant M} \Big\{ u(c) + \beta\, \mathbb{E}_{y}\, V\big(\underbrace{R(M - c) + \tilde{y}}_{=M'}\big) \Big\}
$$

- $M$ is wealth at the beginning of the period, cash on hand
- $0 \leqslant c \leqslant M$: consumption cannot exceed wealth, the consumer **cannot
  borrow**
- $R$ is the gross return on savings and $\tilde{y}$ next period's income, both of which
  can be random
- $u(c) = \log(c)$ and the discount factor $\beta$ as before

Cake eating is the special case $R = 1$ and $\tilde{y} = 0$.

The timing of the model separates the choice from the uncertainty:

$$
M \;\rightarrow\; c(M) \;\rightarrow\; A = M - c(M) \;\rightarrow\; M' = RA + \tilde{y}
$$

Savings $A$, the **post-decision state**, carry all the information about the current
period that next period needs, and the constraint $0 \leqslant c \leqslant M$ is the
same as $0 \leqslant A \leqslant M$.

### Computing expectations

Let income be log-normal, $\log\tilde{y} \sim N(0, \sigma^2)$, so $\tilde{y} > 0$ and
$\mathbb{E}\tilde{y} = \exp(\sigma^2/2)$. The expectation in the Bellman equation is an
integral, and it has to be computed at every state and every choice on every iteration.
We approximate it by **quadrature**, a weighted sum over fixed nodes $y_k$ with weights
$\omega_k$:

$$
\mathbb{E}_y V(RA + \tilde y) = \int V(RA + y)\, dF(y)
\approx \sum_{k=1}^{K} \omega_k\, V(RA + y_k)
$$

How to choose the nodes and the weights, and how quadrature compares with Monte Carlo
integration, is covered in [Class 16](16_cons_practice.md#expectations-by-quadrature).

Both discretized solvers carry over to this model, with one difference. On the grid,
next period wealth is chosen from the state grid and consumption is backed out as
$c = M - (M' - y)/R$, which works only when income is deterministic: with random income
$M'$ is not a choice, so the on-grid trick fails. With a choice grid, consumption is the
choice and $\hat V$ is interpolated at $M' = R(M-c) + y_k$ for every quadrature node.

### The Euler equation

The first order condition for an interior choice and the envelope theorem,

$$
u'(c^\star) = \beta R\, \mathbb{E}_y V'\big(R(M - c^\star) + \tilde y\big),
\qquad
V'(M) = \beta R\, \mathbb{E}_y V'\big(R(M - c^\star) + \tilde y\big),
$$

give $V'(M) = u'(c^\star(M))$, and together the **Euler equation**

$$
u'\big(c^\star(M)\big) = \beta R\, \mathbb{E}_y\, u'\Big(c^\star\big(\underbrace{R[M - c^\star(M)] + \tilde{y}}_{=M'}\big)\Big)
$$

The marginal utility of consuming today equals the discounted expected marginal utility
of saving it and consuming tomorrow. When the borrowing constraint binds, $c^\star = M$
and the left side is larger. The Euler equation holds for the exact solution, whichever
method computed it, so it is the basis of the solvers of the next two classes and of the
accuracy checks for all of them.

(accuracy-checks)=
## Checking the accuracy of a solution

The cake eating model has an analytic solution; the consumption-savings model does not.
Two checks still measure the accuracy of any numerical solution: a special case of the
model with a sharp prediction, and the Euler equation, which holds at every point.

(keane-test)=
### Constant consumption: Keane's test

Take the deterministic model, $\tilde{y} = y$ with certainty, and let $\beta R = 1$.
Then the Euler equation becomes

$$
u'\big(c^\star(M)\big) = u'\big(c^\star(M')\big)
\quad\Rightarrow\quad
c^\star(M) = c^\star(M')
$$

so as long as the borrowing constraint does not bind, **consumption is the same in
every period**. Along such a path wealth must also stay constant, otherwise it
diverges. From $M_0 \geqslant y$ the consumer eats the annuity value of the wealth,

$$
c^\star = \frac{(R-1)\, M_0 + y}{R},
\qquad
M_t = M_0 \text{ for all } t,
$$

and saves $A = (M_0 - y)/R \geqslant 0$, so the constraint never binds.

**Keane's test** uses this prediction: solve the deterministic model with $\beta R = 1$
numerically, simulate the consumer from some initial wealth, and check that the
simulated consumption profile is flat, at the level $c^\star$. An accurate solver passes.
An inaccurate one produces consumption that drifts up or down over time. The error of
the policy function accumulates through wealth, so the test magnifies small errors that
a plot of the policy function hides.

The test is cheap and has a known answer, but it checks the solution only along the
simulated paths, and only for one parameter combination. It is a necessary condition
for an accurate solver, not a sufficient one.

(euler-residuals)=
### Euler equation residuals

The Euler equation holds for the exact solution at every point where the borrowing
constraint does not bind, at any parameter values. For a numerical policy $c(M)$,
interpolated between the grid points, the relative **Euler equation residual**

$$
\varepsilon(M) = \frac{u'\big(c(M)\big) - \beta R\, \mathbb{E}_y\, u'\Big(c\big(R(M - c(M)) + \tilde y\big)\Big)}{u'\big(c(M)\big)}
$$

measures how far the policy is from satisfying it, as a fraction of marginal utility.
It is computed on a grid much denser than the solver's, to catch the errors between the
grid points, and only where $c(M) < M$: at a corner solution the Euler equation holds
as an inequality, and the residual is not an error.

The two checks complement each other. The Euler residuals apply at any parameter
values and at every point of the state space, but only to one step of the dynamics. The
constant consumption check applies to one special case, but follows the errors as they
accumulate over a whole simulated path. The [homework](#homework-iota_deaton) runs both
on the two solvers with discretized choice of this class.

:::{div}
:class: discussion

- Why must wealth stay constant when consumption is constant, $\beta R = 1$ and income
  is deterministic? What happens to wealth if consumption is a little too high?
- Can a solver pass Keane's test and still be inaccurate? Can it have small Euler
  residuals and still fail Keane's test?
:::

````{note} References and additional resources

(15_cake_eating_references)=
- 📖 {cite:t}`deaton1991SavingLiquidityConstraints` "Saving and Liquidity Constraints"
- 📖 {cite:t}`judd1998NumericalMethodsEconomics` "Numerical Methods in Economics", Chapter 6
- 📄 SciPy interpolation tutorial [docs.scipy.org](https://docs.scipy.org/doc/scipy/tutorial/interpolate.html)
- 📺 Foundations of Computational Economics, lectures 30–32 on cake eating and function approximation [fedor.iskh.me/compecon](https://fedor.iskh.me/compecon)
````
