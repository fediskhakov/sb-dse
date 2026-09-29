---
title: "🔬 Practice: NFXP under the hood"
short_title: 🔬 NFXP practice
subtitle: Class 11 — Tuesday, September 29
downloads:
  - file: 11_practical_nfxp.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

The [NFXP estimator of Class 10](10_nfxp.md) reproduces Rust (1987) to the last printed digit. Today we
take a closer look under the hood: how main components interact, where the time goes, what the inner loop contributes, what $\beta$ and the mileage grid do to the estimates, and whether it recovers parameters we generated ourselves.

````{hint} Running the code for this lecture
:class: dropdown

The code for this class is in the folder `session10-11/` of the course **code
repository**, shared with [Class 10](10_nfxp.md). Today is a practical, so that folder
now also holds the notebook `nfxp_practice.ipynb` with the exercises we work on together,
next to `zurcher.py`, `nfxp.py`, the [bus data](10_nfxp.md#nfxp-data) and the
simulator `sim_zurcher.py`. The completed notebook appears there afterwards.

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


(task-nfxp-underhood)=
````{warning} Practical task: NFXP under the hood

Today we study the details of implementation of the Zurcher model and the NFXP estimator.

The files we will examine are in the folder `session10-11` of the course **code repository**
`zurcher.py` and `nfxp.py`.

````

Python can describe its own code: `inspect` lists what a
module defines, with signatures and docstrings.

```{code-cell} python3
:tags: [hide-input]

import sys, pathlib, tempfile, urllib.request
CODE = pathlib.Path('../gh-code/session10-11')       # the code folder, when building next to it
if not CODE.exists():                                 # otherwise fetch the modules from the code repository
    CODE = pathlib.Path(tempfile.mkdtemp())
    url = 'https://raw.githubusercontent.com/fediskhakov/sb-dse-code/main/session10-11/'
    for f in ('zurcher.py', 'nfxp.py'):
        urllib.request.urlretrieve(url + f, CODE / f)
sys.path.insert(0, str(CODE))

import inspect, re, dataclasses

def overview(module, full=False):
    '''Print what a module defines: constants, functions and classes, and for every
    class its attributes, properties and methods, each with its signature and the
    first line of its docstring, or the whole docstring with full=True'''
    def doc(obj, pad):                               # the docstring, indented under the name
        d = inspect.getdoc(obj)                      # getdoc strips the common indentation
        lines = d.split('\n') if d else []
        return ''.join(f'\n{pad}{l}'.rstrip() for l in (lines if full else lines[:1]))
    def sig(f):                                      # signature without self and local paths
        s = str(inspect.signature(f)).replace('(self, ', '(').replace('(self)', '()')
        return re.sub(r"PosixPath\('[^']*/([^/']+)'\)", r"'\1'", s)
    print(f'{module.__name__}.py{doc(module, "│   ")}')
    for name, obj in vars(module).items():
        if name.isupper():                           # module-level constants
            print(f'├── {name} ({type(obj).__name__})')
        elif getattr(obj, '__module__', None) != module.__name__:
            continue                                 # imported, not defined here
        elif inspect.isfunction(obj):
            print(f'├── {name}{sig(obj)}{doc(obj, "│       ")}')
        elif inspect.isclass(obj):
            bases = [b.__name__ for b in obj.__bases__ if b is not object]
            print(f'└── class {name}' + (f'({", ".join(bases)})' if bases else '') + doc(obj, '    │   '))
            src = inspect.getsource(obj)
            lhs = re.findall(r'^\s*((?:self\.\w+\s*,\s*)*self\.\w+)\s*=(?!=)', src, re.M)
            attrs = {a for l in lhs for a in re.findall(r'self\.(\w+)', l)}
            props = [m for m, f in vars(obj).items() if isinstance(f, property)]
            inherited = {m for m, f in inspect.getmembers(obj) if isinstance(f, property)}
            attrs = sorted(a for a in attrs - inherited if not a.startswith('_'))
            if dataclasses.is_dataclass(obj):            # fields of a dataclass
                attrs = [f.name for f in dataclasses.fields(obj)]
            if attrs:                                # attributes: every self.x = ... in the class
                print(f'    ├┈┈ {", ".join(attrs)}')
            for m in props:
                print(f'    ├╌╌ {m}{doc(vars(obj)[m], "    │       ")}')
            for m, f in vars(obj).items():
                if inspect.isfunction(f) and (m == '__init__' or not m.startswith('__')) \
                        and '<string>' not in (inspect.getsourcefile(f) or '<string>'):
                    shown = m.replace(f'_{name}__', '__')   # undo name mangling of private methods
                    print(f'    ╞══ {shown}{sig(f)}{doc(f, "    │       ")}')

import zurcher, nfxp
overview(zurcher)
print()
overview(nfxp)
```


### Representation of the state space and transition matrices

- The state is not mileage but the **index of a mileage bin**
- Assigning `n` builds the grid and, once the transition probabilities are known, the transition matrix:

```python
@n.setter
def n(self, value):
    '''Attribute n setter'''
    if hasattr(self, '_zurcher__p'):   # p may not be set yet when n is assigned first
        assert len(self.__p) < value, 'More transition probability parameters than grid points'
    self.__n = value
    self.grid = np.arange(self.__n)
    if hasattr(self, '_zurcher__p'):
        self.trpr = self.__transition_probs()
```

```python
def __transition_probs(self):
    '''Computing the transision probability matrix'''
    trpr = np.zeros((self.__n,self.__n))  # init
    probs = self.p + [1-sum(self.p)]  # ensure sum up to 1
    for i,p in enumerate(probs):
        trpr += np.diag([p]*(self.__n-i),k=i)
    trpr[:,-1] = 1.-np.sum(trpr[:,:-1],axis=1)  # last column absorbs the mass beyond the grid
    return trpr
```

`grid` is `np.arange(n)`, the bins $0,\dots,n-1$, and the maintenance cost
$0.001 \cdot c \cdot x$ is evaluated at these indices, so $c$ is measured per bin. Both
`n` and `p` are properties whose setters rebuild `trpr`, so the model object can never
hold a grid and a matrix that disagree. The `hasattr` checks are there because `__init__`
assigns `p` before `n`, and the double underscore makes Python store `self.__p` as
`self._zurcher__p`.

`__transition_probs()` builds $\Pi(d=0)$ of the
[motion rules in Class 9](9_zurcher.md#zurcher-transitions) as a sum of shifted
diagonals: probability $\theta_{2j}$ of moving up $j$ bins sits on the $j$-th
superdiagonal, with the residual $1-\sum_j \theta_{2j}$ as the last one. The rows near
the top of the grid lose the mass that would fall beyond it, and the last line puts it
back into the last column, which makes the top bin absorbing. 

$\Pi(d=1)$ is not built: every row of it equals the first row of $\Pi(d=0)$.

### Structure of Bellman operator

```python
def bellman(self,ev0,deriv=False):
    '''Bellman operator for the model
       Depending on deriv argument, returns 2 or 3 outputs (Fréchet derivative)
    '''
    x = self.grid  # points in the next period state
    mcost = -0.001*x*self.c                         # 1-dim array of maintenance costs
    vx0 = mcost + self.beta * ev0                   # 1-dim array v(x,0), keep
    vx1 = mcost[0] - self.RC + self.beta * ev0[0]   # 1-dim array v(x,1), replace
    M = np.maximum(vx0,vx1)                         # de-max values to avoid exp(large number)
    logsum = M + np.log(np.exp(vx0-M) + np.exp(vx1-M))
    ev1 = self.trpr @ logsum                        # 1-dim array after matrix multiplication
    pk = 1/( np.exp(vx1-vx0)+1 )                    # choice prob to keep
    if not deriv:
        return ev1, pk
```

This is the [matrix form of the Bellman operator](9_zurcher.md#zurcher-bellman-matrix),
$\Gamma(EV) = \Pi \cdot L\big(U(\text{keep}) + \beta EV,\; U(\text{replace}) + \beta EV[0]\big)$,
in the [expected value function space](9_zurcher.md#zurcher-ev-bellman). The vector `ev0`
has one element per bin, $EV(x,\text{keep})$, and $EV(x,\text{replace})$ is not stored
at all: replacing resets the mileage, so it equals $EV(0,\text{keep})$ = `ev0[0]` for
every $x$, and `vx1` is a scalar. The logsum is computed with the
[de-maxing trick of Class 6](6_logit.md#demaxing), and `pk` is the
[logit probability of keeping](9_zurcher.md#zurcher-choice-probabilities), written so
that it stays in $[0,1]$ even when the exponent overflows.

:::{div}
:class: discussion

- Why is it enough to store one number per bin when the Bellman equation is written for
  $EV(x,d)$ with two choices?
- `trpr` is dense, $n \times n$, although at most $J+1$ elements per row are non-zero.
  Where would a sparse matrix pay off, and where would it not?
:::

### Computation of the Fréchet derivative

The derivative is the last three lines of `bellman()`, computed only when
`deriv=True`:

```python
# Fréchet derivative
dev1 = self.beta * self.trpr * pk[np.newaxis,:] # element-wise, pk in rows
dev1[:,0] += self.beta * self.trpr @ (1-pk)     # w.r.t. EV[0] special case
return ev1, pk, dev1
```

They are the two terms of the
[Fréchet derivative derived in Class 9](9_zurcher.md#zurcher-frechet),

$$
\frac{\partial \Gamma}{\partial EV} = \beta\, \Pi \,\text{diag}(P) + \beta \big( \Pi \bar P, 0, \dots, 0 \big)
$$

The first term scales column $j$ of $\Pi$ by the probability of keeping at bin $j$, and
`self.trpr * pk[np.newaxis,:]` does exactly that by broadcasting: `pk` becomes a row
vector and multiplies every row of `trpr` element by element. Nothing $n \times n$ is
multiplied, so the cost is $O(n^2)$ rather than the $O(n^3)$ of `trpr @ np.diag(pk)`.
The second term is non-zero in the first column only, because $EV[0]$ enters the
replacement value in every bin, and it is added in place.

The matrix is used twice. The [Newton–Kantorovich step](9_zurcher.md#nk-iterations)
solves a linear system with $I - \Gamma'$:

```python
ev1,pk,dev = self.bellman(ev0,deriv=True) # compute with Fréchet derivative
ev1 = ev0 - np.linalg.solve(np.eye(self.n)-dev,ev0 - ev1)  # NK step
```

and the [analytical gradient of the likelihood](10_nfxp.md#nfxp-score) solves another
system with the same matrix, see `score()` below.

### Main structure of the NFXP estimator loops

`estimate()` is the whole estimator in one method: Rust's three stages, each a call to
the outer loop.

```python
def estimate(self, theta0=(0.0, 0.0), method='bhhh', full=True, verbose=False):
    '''Rust's three stages: transition probabilities from frequencies, then (RC, c)
    by partial likelihood with p fixed, then all parameters jointly.  The object
    ends up at the estimates; theta0 is the starting point for (RC, c).'''
    t0 = time.perf_counter()
    self.reset()
    stages = []
    # stage 1: closed form
    self.p = self.transition_freq()
    # stage 2: partial likelihood
    theta, ok, res = self.maximize(np.asarray(theta0, dtype=float)[:2], method, verbose)
    res['loglik_choice'] = self.loglik_choice(theta)   # the value of the partial likelihood
    stages.append(res)
    # stage 3: full likelihood, from the stage 1 and 2 estimates
    if full:
        theta, ok, res = self.maximize(np.append(theta, self.p), method, verbose)
        stages.append(res)
    self.theta = theta
    S = self.score(theta)
    vcov = np.linalg.inv(S.T @ S)                      # the inverse is wanted for itself here
    return Result(names=self.names[:theta.size], theta=theta, se=np.sqrt(np.diag(vcov)),
                  vcov=vcov, loglik=self.loglik(theta), N=self.N, method=method,
                  converged=ok, time=time.perf_counter() - t0,
                  n_solver_calls=self.n_solver_calls, n_inner_iter=self.n_inner_iter,
                  beta=self.beta, n=self.n, stages=stages)
```

Stage 1 sets the transition probabilities to their frequencies, a closed form. Stage 2
maximizes over $(RC, c)$ with $p$ held there, stage 3 over all parameters from the
stage 2 estimates, and the standard errors come from the outer product of the scores at
the end. `maximize()` only dispatches: to our own `bhhh()` or to `scipy.optimize.minimize`
with the objective, gradient and Hessian methods of the class. Underneath, the calls
nest as in the [nested loop of Class 10](10_nfxp.md#nfxp-nested-loop):

```text
estimate()                                 three stages, standard errors
└── maximize(theta0, method)               outer loop: bhhh() or scipy.optimize.minimize
    ├── loglik(theta)                      objective and line search
    │   └── loglik_obs(theta)
    │       └── loglik_choice_obs(theta)
    │           └── solve(theta)           moves the object to theta
    │               └── solve_fixed_point  inner loop: NK, or poly-algorithm + NK
    └── score(theta)                       gradient; its outer product is the BHHH Hessian
        └── solve(theta)                   same theta: warm start, converges at once
```

The [BHHH iterations of Class 10](10_nfxp.md#bhhh) add a line search on top, so one
outer iteration costs one `score()` and one or more `loglik()` calls, each with a solve
of the model.

:::{div}
:class: discussion

- One call of `loglik()` solves the model once. How many does one BHHH iteration need,
  counting the line search?
- Why does `estimate()` run a partial likelihood stage first, instead of maximizing the
  full likelihood from the start?
:::


### Why functions needed?

```
transition_freq(self)
feasible(self, theta)
snapshot(self)
restore(self, snap)
thousand_miles(self, x)
```

```python
def transition_freq(self):
    '''Transition probabilities from the increment frequencies: the closed form MLE
    away from the top of the grid, and the starting point of the full likelihood.
    An unobserved category gets half an observation so the start is inside the simplex.'''
    counts = np.bincount(self.dx, minlength=self.J + 1).astype(float)
    counts = np.maximum(counts, 0.5)
    return counts[:self.J] / counts.sum()

def feasible(self, theta):
    '''The transition probabilities in theta must form a distribution: the likelihood
    is -inf outside, and the outer loop must not step there'''
    p = theta[2:]
    return bool(np.all(p > 0) and p.sum() < 1)

def snapshot(self):
    '''Parameters and warm start, so a diagnostic can leave the estimator as it found it'''
    return self.theta, None if self.ev is None else self.ev.copy()

def restore(self, snap):
    '''Undo what happened since snapshot()'''
    self.theta, self.ev = snap

def thousand_miles(self, x):
    '''Grid value x in thousands of miles, for tables and plots'''
    return x * self.max_miles / self.n / 1000
```

- `transition_freq()` is stage 1. The [transition part of the likelihood](10_nfxp.md#nfxp-likelihood)
  depends on $\theta_2$ only, and away from the top bin its maximizer is the vector of
  frequencies of the mileage increments. It also sets the default `p` when the object is
  created, so the model has a valid transition matrix before `estimate()` is called.
  The half observation for an increment that never occurs keeps the starting point
  strictly inside the simplex, where the log-likelihood is finite.
- `feasible()` guards that simplex. Outside it some probability is negative or the
  residual $1 - \sum_j p_j$ is, `loglik()` returns $-\infty$ and `score()` returns
  zeros without solving the model, and the line search rejects the step.
- `snapshot()` and `restore()` exist because the estimator is **stateful**: its
  parameters and the warm start `ev` change with every evaluation. A diagnostic that
  evaluates the likelihood elsewhere, `numerical_hessian()` for instance, takes a
  snapshot first and restores it after, so the object is left at the estimates.
- `thousand_miles()` converts a bin index back to mileage for tables and plots, the
  inverse of the discretization in `read_busdata()`.

(warm-start)=
### Warm start mechanism

Why need functions `solve_fixed_point(self, ev0=None, callback=None)` and `solve(self, theta=None)`?

```python
# Inner loop settings.  The fixed point is always finished with Newton-Kantorovich
# steps, whose stopping rule is trustworthy: NK converges quadratically, so a step
# smaller than tol means the true error is far smaller still.  A contraction step
# of size tol, in contrast, can still be tol/(1-beta) away from the fixed point.
# tol is relative to the scale of EV, because the NK linear system has condition
# number of order 1/(1-beta) and its precision floor scales with |EV|.
SOLVER = dict(
    tol=1e-10,          # relative tolerance of the final NK steps
    nk_maxiter=50,      # NK steps before declaring a divergence
    cold=dict(tol=1e-6, maxiter=2000, sa_min=5, sa_max=50, switch_tol=1e-3),  # poly-algorithm from zero
)
```

```python
def solve_fixed_point(self, ev0=None, callback=None):
    '''Solve the model at its current parameters with the settings in self.solver:
    NK steps from ev0 if given, otherwise the poly-algorithm from zero followed by
    NK steps; if NK diverges from a poor warm start, start cold.  The NK tolerance
    is relative to the scale of EV: set from the starting point and, if the solution
    lives on a smaller scale, the last steps are repeated at the tolerance that
    scale demands.  Returns EV and P(keep|x).'''
    tol, nk_maxiter, cold = self.solver['tol'], self.solver['nk_maxiter'], self.solver['cold']
    if ev0 is None:
        ev0, _ = self.solve_poly(callback=callback, **cold)
    for attempt in range(2):
        try:
            with np.errstate(all='ignore'):       # a diverging NK step overflows before it raises
                tol0 = tol * max(1.0, np.abs(ev0).max())
                ev, pk = self.solve_nk(ev0=ev0, tol=tol0, maxiter=nk_maxiter, callback=callback)
                tol1 = tol * max(1.0, np.abs(ev).max())
                if tol1 < tol0:
                    ev, pk = self.solve_nk(ev0=ev, tol=tol1, maxiter=nk_maxiter, callback=callback)
            return ev, pk
        except RuntimeError:
            if attempt == 1:
                raise
            ev0, _ = self.solve_poly(callback=callback, **cold)   # cold restart

def solve(self, theta=None):
    '''Move to theta if given and solve the model, warm started from the previous
    solution when allowed.  Returns EV, P(keep|x) and the Fréchet derivative.'''
    if theta is not None:
        self.theta = theta
    ev0 = self.ev if (self.warm_start and self.ev is not None and self.ev.size == self.n) else None

    def count(**kw):
        self.n_inner_iter += 1

    ev, pk = self.solve_fixed_point(ev0=ev0, callback=count)
    self.n_solver_calls += 1
    self.ev = ev
    return self.bellman(ev, deriv=True)   # ev, pk, dev at the fixed point
```

The two methods split the work between *how* to solve and *from where*.
`solve_fixed_point()` is the inner loop proper. Given a starting point it runs
[Newton–Kantorovich iterations](9_zurcher.md#nk-iterations) straight away; without one it
first runs the [poly-algorithm of Class 9](9_zurcher.md#poly-algorithm) from zero with
the `cold` settings, loose enough to stop as soon as NK can take over. If NK diverges
from a poor starting point, `solve_nk()` raises `RuntimeError` and the method restarts
cold. The tolerance is relative to the size of $EV$, which is of order $10^3$ here, as
the comment above `SOLVER` explains.

`solve()` decides the starting point. With `warm_start=True` it hands over `self.ev`,
the solution at the previous parameter vector, stores the new solution for the next
call and counts the Bellman operator evaluations through the callback. It returns
`bellman(ev, deriv=True)`, so every caller also gets the choice probabilities and the
Fréchet derivative at the fixed point. Between two evaluations of the outer loop the
parameters move very little, $EV_\theta$ moves very little with them, and NK from the
previous solution converges in about three steps, where the poly-algorithm from zero
needs about fifteen (at the estimates, $\beta = 0.9999$).

:::{div}
:class: discussion

- `score()` calls `solve()` at the same $\theta$ that `loglik()` has just solved. What
  does that cost with the warm start, and what without it?
- After a failed line search step far from the optimum, `self.ev` holds the solution
  at a rejected $\theta$. Is that a problem for the next evaluation?
:::

### Collection of log-likelihood evaluators

```python
def choice_values(self, ev):
    '''Choice specific values v(x, keep), v(x, replace) and their logsum at a solution.
    The Bellman operator of the parent computes them but does not return them.'''
    cost = 0.001 * self.c * self.grid
    v0 = -cost + self.beta * ev
    v1 = -cost[0] - self.RC + self.beta * ev[0]
    return v0, v1, np.logaddexp(v0, v1)

# ---------------------------------------------------------------- likelihood
def loglik_choice_obs(self, theta):
    '''Log choice probability log P(d | x) of every bus-month at theta'''
    theta = np.asarray(theta, dtype=float)
    if not self.feasible(theta):
        return np.full(self.N, -np.inf)
    ev, _, _ = self.solve(theta)
    v0, v1, L = self.choice_values(ev)
    logP0, logP1 = v0 - L, v1 - L           # log probabilities directly: never log of a probability
    return np.where(self.d == 0, logP0[self.x], logP1[self.x])

def loglik_transition_obs(self):
    '''Log probability of the observed mileage increment of every bus-month, read off
    the model's own transition matrix: p_dx away from the top of the grid, and the
    cumulative probability where the top bin absorbs'''
    with np.errstate(divide='ignore'):     # a zero probability is a legitimate -inf
        return np.log(self.trpr[self.x_from, self.x_to])

def loglik_obs(self, theta):
    '''Log-likelihood contribution of every bus-month, choice part plus transition part'''
    choice = self.loglik_choice_obs(theta)     # moves to theta first, so trpr is at theta below
    return choice + self.loglik_transition_obs()

def loglik(self, theta):
    '''Log-likelihood, summed over bus-months'''
    return float(np.sum(self.loglik_obs(theta)))

def loglik_choice(self, theta):
    '''The choice part of the log-likelihood alone.  With p held fixed this is the
    partial likelihood of stage 2 up to a constant, and it is the number the
    reference implementations report for that stage.'''
    return float(np.sum(self.loglik_choice_obs(theta)))
```

The [likelihood of Class 10](10_nfxp.md#nfxp-likelihood) is a sum over bus-months of a
choice part and a transition part, and each method returns one piece of it:

- `loglik_choice_obs()` and `loglik_transition_obs()` return one number **per
  bus-month**. BHHH needs the likelihood observation by observation, because its
  Hessian is the outer product of per-observation scores.
- `loglik_obs()` adds the two parts and `loglik()` sums them, the objective of the outer
  loop. The choice part is computed first because it moves the object to $\theta$,
  which rebuilds `trpr` when $p$ changes, and the transition part is then read off the
  matrix at the new $\theta$.
- `loglik_choice()` is the partial likelihood of stage 2: with $p$ fixed the transition
  part is a constant, and this is the number the reference implementations report.
- `choice_values()` recomputes $v(x,\text{keep})$, $v(x,\text{replace})$ and the logsum
  at the solution, because `bellman()` does not return them. The log choice probability
  is then `v0 - L`, a difference, following
  [never take the log of a probability](6_logit.md#log-of-probability).

The transition part reads the model's own matrix at `[x_from, x_to]`, which gives
$p_{\Delta x}$ in the middle of the grid and the cumulative probability in the top bin,
where the matrix absorbs everything beyond it. The `objective*` methods further down
wrap the same numbers for `scipy`, which minimizes: minus the mean log-likelihood, its
gradient, and the outer product of the scores.

### How are gradient of the likelihood and individual scores computed?

```python
def score(self, theta):
    '''N x K matrix of scores, K = theta.size, with the derivative of the fixed
    point from the implicit function theorem'''
    theta = np.asarray(theta, dtype=float)
    K = theta.size
    if not self.feasible(theta):             # the likelihood is -inf there: no direction to report
        return np.zeros((self.N, K))
    ev, pk, dev = self.solve(theta)
    _, _, L = self.choice_values(ev)
    # implicit function theorem on the fixed point EV = Gamma(EV)
    dEV = np.linalg.solve(np.eye(self.n) - dev, self.dbellman_dtheta(pk, L, K))
    # choice part: (1 - d - P(keep|x)) times the derivative of v(x, keep) - v(x, replace)
    dv = self.beta * (dEV[self.x, :] - dEV[0, :])
    dv[:, 0] += 1.0
    dv[:, 1] += -0.001 * self.x
    S = (1 - self.d - pk[self.x])[:, None] * dv
    if K > 2:
        S[:, 2:] += self.dlog_transition()
    return S
```

```python
def dbellman_dtheta(self, pk, L, K):
    '''Derivative of the Bellman operator w.r.t. the first K parameters, EV held fixed, n x K'''
    n, grid, J = self.n, self.grid, self.J
    dG = np.zeros((n, K))
    dG[:, 0] = -(self.trpr @ (1 - pk))                     # RC: replace value moves by -1
    dG[:, 1] = -0.001 * (self.trpr @ (pk * grid))          # c: keep value moves by -0.001 x
    for j in range(K - 2):
        # dPi/dp_j: +1 in column min(i+j, n-1), -1 in column min(i+J, n-1) of row i
        dG[:, 2 + j] = L[np.minimum(grid + j, n - 1)] - L[np.minimum(grid + J, n - 1)]
    return dG

def dlog_transition(self):
    '''Derivative of log Pi[x_from, x_to] w.r.t. p_0, ..., p_{J-1}, N x J: the derivative
    D_j of the matrix over its entry.  Away from the top bin this is +1/p_dx when
    dx == j and -1/p_J when dx == J.'''
    n, J = self.n, self.J
    pr = self.trpr[self.x_from, self.x_to]
    col_J = np.minimum(self.x_from + J, n - 1)
    D = np.empty((self.N, J))
    for j in range(J):
        col_j = np.minimum(self.x_from + j, n - 1)
        D[:, j] = ((self.x_to == col_j).astype(float) - (self.x_to == col_J)) / pr
    return D
```

`score()` returns an $N \times K$ matrix, one row of $\partial \ell_i / \partial \theta$
per bus-month, which is what [BHHH](10_nfxp.md#bhhh) needs. It follows the
[analytical gradient of Class 10](10_nfxp.md#nfxp-score) step by step.

1. **Derivative of the fixed point.** By the implicit function theorem,
   $\partial EV/\partial\theta = (I - \Gamma')^{-1}\, \partial\Gamma/\partial\theta$.
   `dev` is the [Fréchet derivative](9_zurcher.md#zurcher-frechet) that `solve()`
   returned, and the linear solve uses the same matrix $I-\Gamma'$ as the NK step.
2. **Derivative of the Bellman operator with $EV$ held fixed**, in
   `dbellman_dtheta()`. By the derivative of the logsum it is $\Pi$ applied to the
   probability-weighted derivatives of the choice values: $-\Pi(1-P)$ for $RC$, which
   enters the replacement value only, and $-0.001\,\Pi(P \cdot x)$ for $c$. For $p_j$ the
   matrix $\Pi$ itself changes, by $+1$ in the column $j$ bins ahead and $-1$ in the
   column of the residual category $J$, applied to the logsum `L`.
3. **Choice part.** $\log P(d|x)$ is a binary logit in
   $\Delta v(x) = v(x,\text{keep}) - v(x,\text{replace})$, so its derivative is
   $(1 - d - P(\text{keep}|x))\, \partial \Delta v(x) / \partial\theta$, and
   $\partial \Delta v(x)/\partial\theta = \beta\big(\partial EV[x] - \partial EV[0]\big)$
   plus the direct effects, $+1$ for $RC$ and $-0.001\,x$ for $c$.
4. **Transition part**, in `dlog_transition()`: $+1/p_{\Delta x}$ for the category that
   was observed and $-1/p_J$ through the residual probability, with the top bin
   handled by comparing columns rather than increments.

The Fréchet derivative, the implicit function theorem and the choice probabilities are
all by-products of solving the model, which is why the gradient costs about one more
linear solve than the likelihood itself. The replication notebook checks the result
against finite differences of `loglik()`.

:::{div}
:class: discussion

- Which part of `score()` would change if the maintenance cost were quadratic in
  mileage?
- Why is the score of the transition probabilities not zero at the stage 2 estimates,
  where $p$ is at its closed-form maximizer?
:::



## Exercises

The four exercises below are taken from Exercise set 3 of the *Dynamic Programming*
course at the University of Copenhagen (Spring 2020) by Patrick Kofod Mogensen and
Maria Juul Hansen. The questions in the discussion blocks are quoted from it; the code
is rewritten for the model and estimator classes of this course. All of them start from
the data and the estimator of [Class 10](10_nfxp.md):

```python
from nfxp import read_busdata, describe_data, estim_zurcher

data = read_busdata()                        # bus groups 1-4, 175 bins, 450 000 miles
describe_data(data)
result = estim_zurcher(data).estimate()      # BHHH, beta = 0.9999, warm start
print(result)
```

### Run time profiler

A profiler answers "which function is slow" by measuring run times. The standard
library one, `cProfile`, reports time per function; in a notebook `%prun` is a shortcut
for it:

```python
import cProfile, pstats

cProfile.run('estim_zurcher(data).estimate()', 'nfxp.prof')
pstats.Stats('nfxp.prof').sort_stats('cumulative').print_stats(15)
```

Read the cumulative column top down: the outer loop, the score and the inner loop are
nested, so each line includes the ones below it. For line-by-line timing of a single
function install `line_profiler` (`uv pip install line_profiler`), load it with
`%load_ext line_profiler` and run `%lprun -f est.score est.estimate()`.

The optimizer options are the `method` argument of `estimate()`. `bhhh` and `trust-ncg`
use the analytical score and the BHHH approximation of the Hessian, `bfgs` uses the
analytical score and no Hessian, `bfgs-numerical` replaces the score by finite
differences, and `nelder-mead` uses function values only:

```python
for method in ('bhhh', 'trust-ncg', 'bfgs', 'bfgs-numerical', 'nelder-mead'):
    r = estim_zurcher(data).estimate(method=method)
    print(f'{method:<15} RC {r.theta[0]:8.4f}  c {r.theta[1]:7.4f}  loglik {r.loglik:11.4f}  '
          f'{r.time:6.2f} s  {r.n_solver_calls:5d} solves  converged {r.converged}')
```

:::{div}
:class: discussion

Try using line_profiler in python. This gives you a lot of information about the
performance of your code.

- Now try changing the optimizer options, and turn the use of the non-numerical Hessian
  off. What happens?
- Now also try it with the analytical gradient off, what happens?
:::

### The effect of the warm start

`solve()` starts the fixed point iterations from the previous solution, `self.ev`, and
the switch is the `warm_start` argument of `estim_zurcher`. The `Result` counts both
loops: `n_solver_calls` is the number of times the model was solved, `n_inner_iter`
the number of Bellman operator evaluations those solves took in total.

```python
for warm in (True, False):
    r = estim_zurcher(data, warm_start=warm).estimate()
    print(f'warm start {warm!s:<5}  {r.time:5.2f} s  {r.n_solver_calls} solves  '
          f'{r.n_inner_iter} inner iterations  RC {r.theta[0]:.4f}  c {r.theta[1]:.4f}')
```

The mechanism is explained in the [warm start section](#warm-start) above.

:::{div}
:class: discussion

We use the latest EV guess to start the nfxp.solve-procedure even though we change
$\theta$ from one likelihood iteration to another. Why do you think we do that?

- What if we started with EV = 0 in each iteration? Try that and see what happens with
  the parameters and the numerical performance.
:::

### Estimation under different $\beta$

$\beta$ is an argument of `estim_zurcher`, not an element of `theta`, so it is fixed in
estimation. Estimate the model on a grid of discount factors and compare the cost
parameters and the log-likelihood:

```python
for beta in (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999):
    r = estim_zurcher(data, beta=beta).estimate()
    print(f'beta {beta:<7} RC {r.theta[0]:8.4f}  c {r.theta[1]:8.4f}  loglik {r.loglik:11.4f}')
```

:::{div}
:class: discussion

Try estimate the model for different values of $\beta$.

- Why can we not estimate $\beta$?
- When estimating with different $\beta$, do the changes in the estimates of $c$ and/or
  $RC$ make intuitively sense?
- Can you think of some data/variation, which could allow us to identify $\beta$?
:::

### Alternative specification of the mileage process

The discretization is part of the model. `read_busdata(n, max_miles)` chooses the number
of bins and where the grid ends, and every bus with more mileage than `max_miles` lands
in the top bin, which is absorbing unless the engine is replaced. The cost function is
$0.001 \cdot c \cdot x$ with $x$ the bin index, and the transition probabilities are
probabilities of moving a number of bins, so both depend on the width of a bin. The
three specifications of the exercise, next to the baseline:

```python
for n, max_miles in ((175, 450_000), (350, 900_000), (175, 900_000), (88, 225_000)):
    d = read_busdata(n=n, max_miles=max_miles)
    r = estim_zurcher(d).estimate()
    print(f'n {n:3d}  max {max_miles:7,d}  bin {max_miles / n:6.0f} miles  J {r.theta.size - 2}  '
          f'RC {r.theta[0]:7.4f}  c {r.theta[1]:7.4f}  loglik {r.loglik:10.3f}  '
          f'top bin {(d.x == n - 1).mean():.1%}')
```

:::{div}
:class: discussion

Try setting the maximum number of miles (odometer reading) to 900. Now the absorbing
state is much higher.

- If we adjust the number of grid points as well, so that we have a comparable model
  (multiply the number of grids by 2), do we get a better fit?
- Try to lower the number of grid points to 175 again. How do the parameters change?
  Are the changes intuitive?
- What if you change the max to 225 and half the number of grids (hint: what goes
  wrong?)?
:::

````{warning} Practical task: NFXP under the hood

(task-nfxp-practice)=
**Pull the code repository before the class:**

```bash
cd sb-dse-code && git pull
```

Open `session10-11/nfxp_practice.ipynb` and work through the four exercises above:

1. Profile one call of `estimate()`, and compare the optimizers with and without the
   Hessian and the analytical gradient.
2. Measure the effect of the warm start on the inner loop and on the estimates.
3. Estimate the model on the grid of $\beta$ and explain the changes in $RC$ and $c$.
4. Run the alternative specifications of the mileage process, predict each row before
   running it, and explain the last one.

Whatever is left unfinished is finished at home; nothing is collected. Use of AI
assistance is allowed and encouraged, subject to the
[course AI policy](index.md), but you must be able to explain every
number you report.
````

````{note} References and additional resources

(11_practical_nfxp_references)=
- 📖 {cite:t}`rustOptimalReplacementGMC1987` "Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher"
- 📖 {cite:t}`rustNestedFixedPoint2000` "Nested Fixed Point Algorithm Documentation Manual", version 6 {download}`Download pdf <_static/pdf/nfxp_man_2000.pdf>`
- 📖 {cite:t}`suConstrainedOptimizationApproaches2012` "Constrained Optimization Approaches to Estimation of Structural Models"
- 📖 {cite:t}`ecma_comment` "Constrained optimization approaches to estimation of structural models: Comment"
- 📖 {cite:t}`abbringIdentifyingDiscountFactor2020` "Identifying the discount factor in dynamic discrete choice models"
- 📄 Python profilers, `cProfile` and `pstats` <https://docs.python.org/3/library/profile.html>
- 📄 `line_profiler` <https://github.com/pyutils/line_profiler>

````
