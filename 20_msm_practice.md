---
title: "🔬 Practice: MSM estimation of the retirement model"
short_title: 🔬 MSM practice
subtitle: Class 20 — Tuesday, November 3
downloads:
  - file: 20_msm_practice.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

Today is a journey through the [method of simulated moments of Class 19](19_msm.md) on
three models we have solved: the Deaton model, the Deaton model with a consumption
floor, and the retirement model. Each stop uses the same machinery, and each one adds a
difficulty: the choice of moments, a non-convex model with a parameter that is only
identified where it binds, and a discrete choice that makes the criterion a staircase.

````{hint} Running the code for this lecture
:class: dropdown

The code for this class is in the folder `session20-nov3/` of the course **code
repository**.

The complete notebook `msm_practice.ipynb` has all the code of
this page.

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

````{danger} Homework: project proposal due Nov 5
:class: dropdown

Project proposals are due **Thu Nov 5**. 

Prepare a one-two page pdf proposal, following the [guidelines](0_project.md)

Follow the usual process to submit your work using your homework repository using `project_proposal` as the assignment name.

Acceptable file formats: `markdown`, `pdf` (produced from markdown, LaTeX, etc), or `ipynb` (Jupyter notebook).

Feel free to prepare your document right in the homework repository.

````


## Taking stock of existing code in Python modules

Since the [cake eating class](15_cake_eating.md) the code of each class has been
collected into a Python module, each in the
folder of the class that first imported it in the course **code repository**:

| Module | Folder | What it holds |
| :-- | :-- | :-- |
| `deaton.py` | [`session17-oct22/`](https://github.com/fediskhakov/sb-dse-code/tree/main/session17-oct22) | the Deaton model object of [Class 16](16_cons_practice.md#cons-model-object) with its three solvers, VFI with discretized and with continuous choice and time iterations, the Euler residual accuracy check and a simulator |
| `deaton_egm.py` | [`session19-oct29/`](https://github.com/fediskhakov/sb-dse-code/tree/main/session19-oct29) | the same model object with the EGM solver of [Class 17](17_egm.md) and a simulator, the pieces the [MSM example of Class 19](19_msm.md#msm-comparative-statics) runs on |
| `retirement.py` | [`session20-nov3/`](https://github.com/fediskhakov/sb-dse-code/tree/main/session20-nov3) | the retirement model of [Class 18](18_dcegm.md#retirement-model) with the DC-EGM solver and a simulator |
| `cons_floor.py` | [`session20-nov3/`](https://github.com/fediskhakov/sb-dse-code/tree/main/session20-nov3) | the consumption-floor model of the [homework kappa_floor](17_egm.md#homework-kappa_floor), solved by EGM with the upper envelope step of DC-EGM, next to the VFI benchmark and a simulator |
| `msm.py` | [`session20-nov3/`](https://github.com/fediskhakov/sb-dse-code/tree/main/session20-nov3) | the model-free part of the MSM estimator of [Class 19](19_msm.md#msm-weighting), the minimizer, the Jacobian, the sandwich variance and the $J$-test, and the machinery of this class that connects a model to it |

The modules of the last two classes and of the homework are new. Each opens with a
docstring that shows how it is meant to be called, and the notebooks of the classes
that wrote the code remain the place to read it with its explanations. Every module is
imported by name, so every call below says which module it runs.

```{code-cell} python3
:tags: [hide-input]

# Import all the modules of the table above, by name: every call below says which module it runs
import sys, time, pathlib, tempfile, urllib.request
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
MODULES = {'session17-oct22': ('deaton.py',),               # the folder of every module in the code repository
           'session19-oct29': ('deaton_egm.py',),
           'session20-nov3': ('retirement.py', 'msm.py', 'cons_floor.py')}
url = 'https://raw.githubusercontent.com/fediskhakov/sb-dse-code/main/'
for folder, files in MODULES.items():
    CODE = pathlib.Path('../gh-code') / folder                # the local clone of the code repository
    if not CODE.exists():                                     # otherwise fetch the modules from GitHub
        CODE = pathlib.Path(tempfile.mkdtemp())
        for f in files:
            urllib.request.urlretrieve(f'{url}{folder}/{f}', CODE / f)
    sys.path.insert(0, str(CODE))
import deaton                                                 # Deaton model, VFI and time iterations (Class 16)
import deaton_egm                                             # Deaton model with EGM and a simulator (Classes 17, 19)
import retirement                                             # retirement model, DC-EGM and a simulator (Class 18)
import cons_floor                                             # consumption floor model, EGM with upper envelope (homework)
import msm                                                    # the model-free MSM estimator (Class 19)
```

The estimator in `msm.py` takes the criterion and the simulated moments as function
arguments. What changes from model to model:

- a **library** of moment functions,
- a **simulator** that produces paths like the data at any $\theta$,
- and the **bounds** of the parameter region, the part of the parameter space where the
  solver works: outside them the simulator returns nothing, the criterion returns a huge
  number, and a derivative-free optimizer turns back.

The machinery of `msm.py` turns those three things into everything else:

| Function | What it does |
| :-- | :-- |
| `msm.moments(lib, names, *paths)` | the contributions $h(z_i)$, one row per record, for a set of moments named from the library |
| `msm.setup(lib, names, data, simulate_at)` | the data moments, their covariance $\hat\Omega$, the simulated moments on fixed draws and the criterion, the last two as functions of $\theta$ |
| `msm.wdiag(Omega)` | the diagonal weighting matrix of inverse variances, with no weight on a moment without variation |
| `msm.fit(crit, simmom, x0, W, Omega, N)` | the estimator under one weighting matrix, with a restart from its own solution, and the standard errors |
| `msm.table(results, pnames)` | the results under several weighting matrices side by side |
| `msm.diagnostics(lib, names, data, simulate_at, N, theta0, grids, pnames)` | the sweep of every moment along each parameter in standard-error units and the map of the criterion, the two figures of [Class 19](19_msm.md#msm-comparative-statics) |
| `msm.montecarlo(...)`, `msm.mc_report(...)` | replications of the estimator with the diagonal weighting matrix on fresh data sets, and their table and histograms |

A set of moments is a list of names, so it can be changed in one line and every
diagnostic rerun. The docstring of every function in the module says what goes in and
what comes out, and the functions are read in class as they come up.

(msm-practice-deaton)=
## Part 1: the Deaton model

The data are those of [Class 19](19_msm.md#msm-comparative-statics): 500 consumers of
the Deaton model with $\beta = 0.92$ and $\sigma = 0.5$, simulated for 40 periods from
log-normal initial wealth, with the last ten periods observed. The simulator draws
fresh initial wealth from the same distribution, which the estimator treats as known.

```{code-cell} python3
theta0 = np.array([0.92, 0.5])                            # true beta and sigma
FIXED_D = dict(R=1.05, Mbar=20, ngrid=100, nquad=10)      # the rest of the model, fixed and known: return, wealth grid on [0, Mbar] with ngrid points, quadrature nodes
N, T, tail = 500, 40, 10                                  # individuals, simulated periods, observed periods
m0 = deaton_egm.deaton(beta=theta0[0], sigma=theta0[1], **FIXED_D)
pol0 = deaton_egm.solve_egm(m0)
def deaton_data(rng):
    '''A data set: N paths from the true model, initial wealth log-normal, the last `tail` periods observed'''
    return deaton_egm.simulate(m0, pol0, np.exp(rng.standard_normal(N)), T, seed=rng)
Wd, Cd = deaton_data(np.random.default_rng(2026))
print(f'wealth {Wd.shape}, consumption {Cd.shape}; observed: the last {tail} periods')

DEATON = {'mean wealth':            lambda W, C: W[:, -tail:].mean(axis=1),
          'std wealth':             lambda W, C: W[:, -tail:].std(axis=1),
          'std consumption growth': lambda W, C: np.diff(np.log(C[:, -tail:]), axis=1).std(axis=1),
          'mean consumption':       lambda W, C: C[:, -tail:].mean(axis=1),
          'mean saving rate':       lambda W, C: (1 - C[:, -tail:] / W[:, -tail:]).mean(axis=1)}

def deaton_simulate_at(theta, S=5, seed=1):
    '''Simulate S copies of the data at theta on fixed draws, initial wealth included; None outside the bounds'''
    beta, sigma = theta
    if not (0 < beta < 1 / FIXED_D['R'] and sigma > 0): return None   # the bounds: EGM does not converge for beta R >= 1
    m = deaton_egm.deaton(beta=beta, sigma=sigma, **FIXED_D)
    rng = np.random.default_rng(seed)                         # the same generator state at every theta
    return deaton_egm.simulate(m, deaton_egm.solve_egm(m), np.exp(rng.standard_normal(S * N)), T, seed=rng)
```

### Which moments

The library holds five moment functions of one path, and a set of moments is a list of
their names. Three sets, each with its sweeps and its criterion map:

- **A**, the three moments of Class 19: mean wealth, the spread of wealth, the spread of
  consumption growth
- **B**, two means only: mean wealth and mean consumption
- **C**, all five

```{code-cell} python3
grids_d, pnames_d = (np.linspace(0.86, 0.95, 31), np.linspace(0.3, 0.7, 31)), ('β', 'σ')
A = ['mean wealth', 'std wealth', 'std consumption growth']
B = ['mean wealth', 'mean consumption']
C = list(DEATON)
mhat, Omega, simmom, crit = msm.diagnostics(DEATON, A, (Wd, Cd), deaton_simulate_at, N, theta0, grids_d, pnames_d)
```

```{code-cell} python3
_ = msm.diagnostics(DEATON, B, (Wd, Cd), deaton_simulate_at, N, theta0, grids_d, pnames_d)
```

```{code-cell} python3
mhatC, OmegaC, simmomC, critC = msm.diagnostics(DEATON, C, (Wd, Cd), deaton_simulate_at, N, theta0, grids_d, pnames_d)
print(f'condition number of Omega: {np.linalg.cond(Omega):.0f} with set A, {np.linalg.cond(OmegaC):.0f} with set C')
```

- **Set A** has one basin around the truth. $\beta$ moves mean wealth up and
  consumption growth down, $\sigma$ moves the two wealth moments together: two
  directions, two parameters.
- **Set B** has a valley. Both means rise with $\beta$ and rise with $\sigma$, so a
  lower $\beta$ with a higher $\sigma$ reproduces them: the two columns of $D$ are nearly
  collinear, and the criterion has a floor running diagonally through the truth.
- **Set C** has the basin of A and little more. The two extra means respond to both
  parameters in the same way as mean wealth, so they repeat information already there
  while raising the condition number of $\hat\Omega$ by a factor of seventy, which the
  optimal weighting matrix will feel.

:::{div}
:class: discussion

- The sweeps are in units of the standard error of the data moment. Why is that the
  right scale for judging whether a moment is informative?
- Set B has two moments and two parameters: the order condition holds. What exactly
  fails?
:::

### Fixed draws against fresh draws

The simulator draws the shocks from a generator seeded with the same number at every
$\theta$. The criterion with fresh draws at every call is a different object, and the
figure shows the difference along $\beta$: the fixed-draw criterion with one and with
five copies of the data, and the fresh-draw criterion with one.

```{code-cell} python3
:tags: [hide-input]

Wdiag = msm.wdiag(Omega)                                          # moments scaled by their variances
counter = iter(range(10**6))
def crit_free(theta, W):
    '''The criterion with fresh draws at every call'''
    return crit(theta, W, S=1, seed=1000 + next(counter))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, (lo, hi) in zip(ax, ((0.86, 0.95), (0.912, 0.928))):
    bg = np.linspace(lo, hi, 61)
    a.plot(bg, [crit([b, theta0[1]], Wdiag, S=1) for b in bg], lw=1, label='fixed draws, S = 1')
    a.plot(bg, [crit([b, theta0[1]], Wdiag, S=5) for b in bg], lw=1, label='fixed draws, S = 5')
    a.plot(bg, [crit_free([b, theta0[1]], Wdiag) for b in bg], '.', ms=3, label='fresh draws at every call, S = 1')
    a.axvline(theta0[0], color='k', lw=0.5, ls=':'); a.grid(color='0.9')
ax[0].set(xlabel='β', ylabel='MSM criterion, log scale', yscale='log', title=f'Criterion along β on [0.86, 0.95], σ = {theta0[1]}')
ax[1].set(xlabel='β', ylabel='MSM criterion', title=f'Zoom on [0.912, 0.928] around the truth β = {theta0[0]}')
ax[0].legend()
plt.show()
```

What an optimizer makes of the two: Nelder–Mead on the fresh-draw criterion three
times, with a budget of 400 evaluations, and on the fixed-draw criterion once.

```{code-cell} python3
for k in range(3):                                            # Nelder-Mead on the fresh-draw criterion, three times
    res = minimize(crit_free, (0.9, 0.4), args=(Wdiag,), method='Nelder-Mead',
                   options={'xatol': 1e-5, 'fatol': 1e-9, 'maxfev': 400})
    print(f'fresh draws: stopped at beta {res.x[0]:.4f} sigma {res.x[1]:.4f} after {res.nfev} evaluations; converged: {res.success}')
th_fix, Q_fix, nfev = msm.estimate(crit, x0=(0.9, 0.4), args=(Wdiag,))
print(f'fixed draws: beta {th_fix[0]:.4f} sigma {th_fix[1]:.4f} Q {Q_fix:.4f} after {nfev} evaluations')
```

With fresh draws the simplex never contracts: every evaluation sees a new random
function, the stopping rule on the criterion value is never met, and the optimizer runs
out of budget somewhere near the truth, in a different place each time. With fixed draws
it converges in under a hundred evaluations. The fixed draws are what makes the
criterion a function.

### Nelder-Mead gradient-free optimizer

`msm.estimate` minimizes by Nelder–Mead. The method works on a **simplex** of $K + 1$
points, a triangle for two parameters. One iteration:

- evaluate the criterion at the vertices and drop the worst one
- **reflect** it through the centroid of the others
- **expand** the step if the reflection paid off, **contract** it if it did not
- when nothing helps, **shrink** the whole simplex toward its best vertex

No derivative is used, so the steps of a simulated criterion do not stop it. The method
stops when the simplex is small: within `xatol` in the parameters and within `fatol` in
the criterion values. That is its weak point:

- a small simplex is reached at a minimum, but also around a point that is not one
- on a flat or stepped stretch the reflections keep failing and the contractions pile up
- the simplex **collapses** onto its current best vertex, with the slope still ahead of it
- the method cannot tell the two cases apart, because it never sees a gradient

The **restart** is the remedy: a second run started at the solution of the first.

- It rebuilds a full-size simplex around the solution, in scipy by moving each
  coordinate by five percent.
- At a minimum every new vertex is worse. The simplex contracts again within a few
  dozen evaluations, and the solution is confirmed.
- After an early stop the fresh simplex straddles the slope that the collapsed one could
  not see, and the run carries on from there.
- It costs one more run of about a hundred evaluations, a second here. Cheap insurance
  for a method that cannot check its own optimality.

### MSM with different weighting matrices

The estimator of Class 19 leaves one choice open, the weighting matrix. Three choices
with set A, each from the same poor start: the identity, the diagonal matrix of inverse
variances, and the optimal matrix, the inverse of $\hat\Omega$. The three runs are
independent of each other. The optimal matrix needs no preliminary estimate here,
because the moments are means over individuals and $\hat\Omega$ is the covariance of
their contributions, computed from the data once. The
[two-step algorithm of Class 19](19_msm.md#msm-weighting) is for moments that are not
means, where $\hat\Omega$ has to be simulated at a first estimate.

The operations of one run, written out once with the diagonal matrix: minimize, restart
from the solution, then the Jacobian of the moments by finite differences on the fixed
draws, and the sandwich.

```{code-cell} python3
S = 5
th, Q, n1 = msm.estimate(crit, x0=(0.88, 0.4), args=(Wdiag,))       # minimize from a poor start
th, Q, n2 = msm.estimate(crit, x0=th, args=(Wdiag,))                # restart from the solution
D = msm.jacobian(simmom, th, S)                                     # q x K, finite differences on fixed draws
se = np.sqrt(np.diag(msm.vcov(D, Wdiag, Omega, N, S)))              # standard errors from the sandwich
print(f'diagonal W   beta {th[0]:.4f} ({se[0]:.4f})  sigma {th[1]:.4f} ({se[1]:.4f})  Q {Q:.4f}  {n1 + n2} evaluations')
print('Jacobian D (rows: moments, columns: beta, sigma):'); print(np.round(D, 3))
```

The Jacobian is the sweep figure in numbers: its rows are the slopes of the three
lines at the truth, and its two columns point in different directions. These operations
are what `msm.fit` does in one call. The three weighting matrices side by side, and the
table every MSM paper carries, data moments against simulated moments at the estimate:

```{code-cell} python3
WS = {'identity': np.eye(len(A)), 'diagonal': Wdiag, 'optimal': np.linalg.inv(Omega)}   # the three weighting matrices
res = {name: msm.fit(crit, simmom, (0.88, 0.4), W, Omega, N, S) for name, W in WS.items()}   # the same start for all
msm.table(res, ('beta', 'sigma'))
th_opt = res['optimal'][0]                                           # the estimate with the optimal W, used below
print(f'\n{"moment":<24} {"data":>8} {"s.e.":>8} {"model, optimal W":>17}')
for name, mh, sd, mm in zip(A, mhat, np.sqrt(np.diag(Omega) / N), simmom(th_opt)):
    print(f'{name:<24} {mh:8.4f} {sd:8.4f} {mm:17.4f}')
```

All three land within a standard error of the truth, and the weighting matrix changes
little: with three nearly uncorrelated moments the optimal matrix has little to reweight,
and the identity loses some precision by letting the moment with the largest variance,
mean wealth, dominate the criterion. Every moment is matched within a standard error at
the optimal estimate. The row of criterion values is not comparable across the columns:
each weighting matrix sets its own scale.

:::{div}
:class: discussion

- A gradient method checks its own optimality through the gradient. What would the
  equivalent check be after Nelder–Mead stops, and what would it cost on a stepped
  criterion?
:::

(msm-practice-montecarlo)=
### Monte Carlo: the asymptotic distribution

The standard errors come from the sandwich with a numerical Jacobian. The check is a
Monte Carlo: draw data sets from the true model, run the estimator with the diagonal
weighting matrix on each, and compare
the spread of the estimates with the formula. The asymptotic theory says more than the
spread: the estimates standardized by their own formula standard errors should look
like draws from the standard normal, and a 95% confidence interval should cover the
truth in 95% of the replications. Every replication draws its own data set and its own
simulation draws, as two studies would, so both sources of noise in the $(1 + 1/S)$
formula are at work. `msm.montecarlo` does exactly that, with the diagonal weighting
matrix in every replication, and `msm.mc_report` prints the table and draws the
histograms.

```{code-cell} python3
t0 = time.perf_counter()
ests, ses = msm.montecarlo(deaton_data, DEATON, A, deaton_simulate_at, x0=(0.88, 0.4), N=N, R=50)
print(f'{len(ests)} replications in {time.perf_counter() - t0:.0f} s')
msm.mc_report(ests, ses, theta0, ('beta', 'sigma'))
```

The estimator is centered on the truth for both parameters, the spread of the
estimates is within 10% of the formula, a little below it for $\beta$ and a little above
for $\sigma$, and the 95% intervals cover the truth in 96% and 90% of the replications.
The histograms of the standardized estimates are what fifty draws from a standard normal
look like, no more and no less: fifty replications pin a standard deviation down to
about 10% and a coverage rate to about three percentage points, so this is a check of
the formula, not a measurement of it. The scatter in the last panel shows the two
estimates nearly uncorrelated, as the basin of set A suggests.


(msm-practice-floor)=
## Part 2: the consumption floor

The model of the [homework kappa_floor](17_egm.md#homework-kappa_floor): the Deaton model
without income risk, with next period wealth topped up to a floor $\underline{x}$. It is
solved by the EGM with the upper envelope step of `cons_floor.py`, which handles the
poverty trap that plain EGM misses. The model is deterministic, so there is no $\sigma$
to estimate; the parameters are $\beta$ and the floor, and the only randomness in the
data is the initial wealth, uniform on $[1, 12]$ for a thousand consumers over eight
periods.

```{code-cell} python3
theta0f = np.array([0.9, 1.5])                            # true beta and floor
FIXED_F = dict(R=1.05, y=1.0, Mbar=10, ngrid=200)         # the rest of the model, fixed and known: return, income, wealth grid on [0, Mbar] with ngrid points
Nf, Tf = 1000, 8                                          # consumers and periods
def floor_data(rng, theta=theta0f):
    '''A data set: Nf deterministic paths from the model at theta, initial wealth uniform on [1, 12]'''
    m = cons_floor.cons_floor(beta=theta[0], floor=theta[1], **FIXED_F)
    return cons_floor.simulate(m, cons_floor.solve_egm(m), rng.uniform(1, 12, Nf), Tf)
Mf, Cf, Ff = floor_data(np.random.default_rng(2026))
print(f'share on the floor by period  {np.round(Ff.mean(axis=0), 2)}')
print(f'mean wealth by period         {np.round(Mf.mean(axis=0), 2)}')
```

Everyone is on the floor by period five: the poverty trap of a floor at 1.5 ends at
wealth 3.87, and the richest consumer, at 12, saves for four periods before landing on
it. Moments of later periods have no variation in the data, so the library holds the
first three periods of wealth and of the floor indicator, and the first three of
consumption.

```{code-cell} python3
FLOOR = {**{f'mean wealth t={t}': (lambda M, C, F, t=t: M[:, t]) for t in (1, 2, 3)},
         **{f'share on floor t={t}': (lambda M, C, F, t=t: F[:, t].astype(float)) for t in (1, 2, 3)},
         **{f'mean consumption t={t}': (lambda M, C, F, t=t: C[:, t]) for t in (0, 1, 2)}}

def floor_simulate_at(theta, S=5, seed=1):
    '''Simulate S copies of the data at theta: initial wealth from the fixed draws, deterministic paths'''
    beta, kappa = theta
    if not (0.5 < beta < 1 / FIXED_F['R'] and 0 <= kappa < 4): return None   # the bounds: beta R < 1 for the solver, a floor below 4
    m = cons_floor.cons_floor(beta=beta, floor=kappa, **FIXED_F)
    return cons_floor.simulate(m, cons_floor.solve_egm(m), np.random.default_rng(seed).uniform(1, 12, S * Nf), Tf)
```

### Does the same recipe work

The diagnostics, with all nine moments:

```{code-cell} python3
grids_f, pnames_f = (np.linspace(0.85, 0.94, 31), np.linspace(1.0, 2.0, 31)), ('β', 'floor')
Af = list(FLOOR)
mhatf, Omegaf, simmomf, critf = msm.diagnostics(FLOOR, Af, (Mf, Cf, Ff), floor_simulate_at, Nf, theta0f, grids_f, pnames_f)
print(f'condition number of Omega: {np.linalg.cond(Omegaf):.2g}')
```

- The floor is strongly identified: the share on the floor in periods 2 and 3 climbs
  off the chart within a tenth of the true value, and mean wealth in those periods
  rises with the floor, because wealth on the floor *is* the floor.
- $\beta$ is identified through the savers: a more patient consumer saves more in period
  0, so consumption in period 0 falls and wealth in period 1 rises with $\beta$.
- The criterion map is a basin stretched along $\beta$, which is the less precisely
  identified of the two.
- $\hat\Omega$ is **singular**, with a condition number of $10^{15}$. The paths are
  deterministic given initial wealth, and the budget constraint
  $M_2 = R(M_1 - c_1) + y + (\underline{x} - y)\,\mathbf 1(\text{on the floor at } 2)$
  holds for every consumer, so wealth in period 2, wealth and consumption in period 1
  and the floor indicator of period 2 are exactly collinear in the data. The optimal
  weighting matrix does not exist for this set, and the diagonal one is used throughout.

The criterion through the truth along each parameter, with one and with five copies of
the data on fixed draws, as for the Deaton model:

```{code-cell} python3
:tags: [hide-input]

Wf = msm.wdiag(Omegaf)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, k, g in zip(ax, (0, 1), grids_f):
    for S_ in (1, 5):
        th = lambda v: [v, theta0f[1]] if k == 0 else [theta0f[0], v]   # move one parameter, the other at the truth
        a.plot(g, [critf(th(v), Wf, S_) for v in g], lw=1, label=f'fixed draws, S = {S_}')
    a.axvline(theta0f[k], color='k', lw=0.5, ls=':'); a.grid(color='0.9')
ax[0].set(xlabel='β', ylabel='MSM criterion, log scale', yscale='log', title=f'Criterion along β, floor = {theta0f[1]}')
ax[1].set(xlabel='floor', ylabel='MSM criterion, log scale', yscale='log', title=f'Criterion along the floor, β = {theta0f[0]}')
ax[0].legend()
plt.show()
```

- Both cuts have their minimum at the truth. The one along the floor is the deeper and
  the sharper: two orders of magnitude between a floor at income and the true one,
  against one order along $\beta$ over a comparable distance. This is the stretched
  basin of the map seen from its two sides.
- The two sample sizes almost coincide. The model is deterministic, so the only
  simulation noise is in the draws of initial wealth, and five thousand uniform draws
  average little better than one thousand against data of one thousand. At the minimum
  the smaller sample even sits lower, by the luck of its draws.
- Both cuts are faintly kinked. The share on the floor is an indicator of initial wealth
  against the trap boundary, which moves with both parameters, so the simulated moments
  step whenever a draw crosses it. The retirement model turns these kinks into a
  staircase.

```{code-cell} python3
:tags: [hide-input]

kg = np.linspace(0, 3, 61)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(kg, [critf([theta0f[0], k], Wf) for k in kg], lw=1.5)
ax[0].axvline(theta0f[1], color='k', lw=0.5, ls=':'); ax[0].axvline(1.0, color='C3', lw=0.5, ls='--')
ax[0].set(xlabel='floor', yscale='log', title=f'Criterion along the floor, data from floor {theta0f[1]}'); ax[0].grid(color='0.9')
for f in (1.1, 1.5, 2.0):
    Mx, Cx, Fx = floor_data(np.random.default_rng(2026), theta=(theta0f[0], f))
    mh_x, Om_x, sm_x, cr_x = msm.setup(FLOOR, Af, (Mx, Cx, Fx), floor_simulate_at)
    ax[1].plot(kg, [cr_x([theta0f[0], k], msm.wdiag(Om_x)) for k in kg], lw=1.5, label=f'data from floor {f}')
    ax[1].axvline(f, color='k', lw=0.5, ls=':')
ax[1].set(xlabel='floor', yscale='log', title='Criterion along the floor, three data sets'); ax[1].legend(); ax[1].grid(color='0.9')
plt.show()
```

The criterion along the floor, with $\beta$ at its true value, is **flat below
income**: a floor at or below $y = 1$ never binds, the model is the same for every such
value, and so are the simulated moments. Identification of the floor begins where it
binds. The right panel repeats the cut for data generated with floors at 1.1, 1.5 and 2:
each criterion has its minimum at its own floor, each is flat below one, and the higher
the floor the deeper and wider the basin, because more consumers are caught by it. With
a floor of 2 the moments of period 3 have no variation left in the data, and `msm.wdiag`
gives them no weight.

The estimator from the same poor start under the identity and the diagonal weighting
matrix. There is no optimal column: the inverse of a singular $\hat\Omega$ does not
exist. A third run starts with the floor below income, on the flat part of the
criterion.

```{code-cell} python3
t0 = time.perf_counter()
WSf = {'identity': np.eye(len(Af)), 'diagonal': Wf}               # no optimal matrix: Omega is singular
resf = {name: msm.fit(critf, simmomf, (0.88, 1.2), W, Omegaf, Nf) for name, W in WSf.items()}   # the same start for both
msm.table(resf, ('beta', 'floor'))
print(f'two runs in {time.perf_counter() - t0:.0f} s')
thb, Qb, nb = msm.estimate(critf, x0=(0.88, 0.5), args=(Wf,))        # a start with the floor below income: on the flat
print(f'\nstart on the flat  beta {thb[0]:.4f}  floor {thb[1]:.4f}  Q {Qb:.4f}  {nb} evaluations')
print('column of the Jacobian for the floor there:', np.round(msm.jacobian(simmomf, thb, 5)[:, 1], 4))
```

- Started inside the binding region, the diagonal matrix recovers both parameters
  within about a standard error, in a few hundred evaluations of a solver that takes 20
  milliseconds. The identity lets the wealth and consumption levels decide and all but
  ignores the shares on the floor: its standard errors are three to four times larger,
  and its estimate lands two standard errors from the truth in both parameters.
- It lands up the valley. The elongated basin of the map is the correlation of the two
  estimates, a more patient consumer with a higher floor reproducing nearly the same
  wealth profile, and the identity estimate has drifted along exactly that direction, a
  higher $\beta$ together with a higher floor.
- Started below income, Nelder–Mead **stops on the flat**: the criterion does not change
  along the floor there, the simplex contracts in that direction, and the solution
  reports a floor of 0.55 with a criterion a hundred times the minimum. The column of
  the Jacobian for the floor is exactly zero, so the rank condition fails and no
  standard error exists. The lesson is the one of [Class 19](19_msm.md#msm-moments): a
  parameter is identified only where the moments respond to it, and an optimizer cannot
  find a region it cannot see.

A short Monte Carlo, eight replications with both parameters and the diagonal matrix:
enough for the table, not for histograms.

```{code-cell} python3
t0 = time.perf_counter()
estf, sesf = msm.montecarlo(floor_data, FLOOR, Af, floor_simulate_at, x0=(0.88, 1.2), N=Nf, R=8)
print(f'{len(estf)} replications in {time.perf_counter() - t0:.0f} s')
msm.mc_report(estf, sesf, theta0f, ('beta', 'floor'), plot=False)
```

The estimates are centered on the truth and the formula standard errors are of the
right size, somewhat above the spread. Eight replications cannot tell a 0.6% standard
error from a 0.8% one, nor a coverage of 100% from one of 95%.

:::{div}
:class: discussion

- The singular $\hat\Omega$ comes from the budget constraint. Which moment would you drop
  to make the optimal weighting matrix exist, and would the estimate change?
- With income risk, as in the Deaton model, would the criterion still be flat below
  $y$? What does the answer say about which models can identify a safety net?
- The poverty trap ends at wealth 3.87 for a floor of 1.5. Which moment carries that
  number, and how would you add one that measures it directly?
:::

(msm-practice-retirement)=
## Part 3: the retirement model

The [retirement model of Class 18](18_dcegm.md#retirement-model) adds a discrete
choice: every period a worker decides whether to keep working, and retirement is
absorbing. The data are 1000 workers with initial wealth uniform on $[2, 20]$, taste
shocks $\sigma = 0.1$ and income shocks $\sigma_y = 0.2$, over $T = 10$ periods. The
parameters we pretend not to know:

| | true value | estimated |
| :-- | :-- | :-- |
| disutility of work $\delta$ | $0.5$ | yes |
| discount factor $\beta$ | $0.95$ | yes |
| taste shock scale $\sigma$ | $0.1$ | fixed, see [what breaks](#msm-practice-sigma) |
| income shock $\sigma_y$ | $0.2$ | fixed |
| return $R$, wage $y$ | $1.05$ | fixed |
| wage $y$ | $1$ | fixed |

### Simulations with a discrete choice

```{code-cell} python3
theta0r = np.array([0.5, 0.95])                           # true delta and beta
FIXED_R = dict(sigma=0.1, sigma_y=0.2, R=1.05, y=1.0, T=10, Mbar=20, ngrid=500, nquad=10)   # the rest of the model, fixed and known: taste shock scale, income shock std, return, wage, periods, savings grid on [0, Mbar] with ngrid points, quadrature nodes
def model(theta):
    '''The retirement model at the parameters we estimate; everything else fixed at the values above'''
    delta, beta = theta                                   # the two parameters, in this order everywhere below
    return retirement.retirement(beta=beta, delta=delta, **FIXED_R)   # the model object of Class 18
Nr = 1000                                                 # workers in the data
rng = np.random.default_rng(3)                            # the seed of the data: initial wealth and all shocks
M0r = rng.uniform(2, 20, Nr)                              # initial wealth of the sample
mr = model(theta0r)                                       # the true model
solr = retirement.solve_dcegm(mr)                         # its solution: sol[t][d] for every period and choice
Md, Cd_r, Dd, VWd, VRd = retirement.simulate(mr, solr, M0r, seed=rng, values=True)   # the data: wealth, consumption, work status, Nr x T each, and the two values behind every decision
print(f'share working by period  {np.round(Dd.mean(axis=0), 2)}')   # the profile of work status, one number per period
print(f'mean wealth by period    {np.round(Md.mean(axis=0), 1)}')   # the profile of wealth
```

```{code-cell} python3
:tags: [hide-input]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
t = np.arange(mr.T)
for i in range(12):
    ax[0].plot(t, Md[i], '-', lw=0.8, color=f'C{i % 10}'); ax[0].plot(t[Dd[i] == 1], Md[i, Dd[i] == 1], 'o', ms=4, color=f'C{i % 10}')
    ax[1].plot(t, Cd_r[i], '-', lw=0.8, color=f'C{i % 10}'); ax[1].plot(t[Dd[i] == 1], Cd_r[i, Dd[i] == 1], 'o', ms=4, color=f'C{i % 10}')
ax[0].set(xlabel='period', ylabel='wealth', title='12 workers: wealth, dots while working')
ax[1].set(xlabel='period', ylabel='consumption', title='the same workers: consumption')
ax[2].plot(t, Dd.mean(axis=0), 'o-', label='share working'); ax[2].plot(t, Md.mean(axis=0) / Md[:, 0].mean(), 's-', label='mean wealth / initial')
ax[2].set(xlabel='period', title='the profiles of the sample'); ax[2].legend()
for a in ax: a.grid(color='0.9')
plt.show()
```

Twelve workers on the left: wealth runs down along straight-looking lines while they
work, with dots, and faster once they retire and live off savings. Consumption in the
middle is nearly flat for each worker, with the income shocks on top, and jumps at
retirement when the disutility of work disappears from the trade-off. The two profiles
on the right are the sample averages: the share working falls from 83% to zero in the
last period, where no one works, and mean wealth is run down to a sixth of its initial
level.

### Playing with the parameters

The same initial wealth and the same draws, with the disutility of work at a quarter,
a half and one, and with the discount factor at 0.9:

```{code-cell} python3
:tags: [hide-input]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for d in (0.25, 0.5, 1.0):
    mx = model((d, theta0r[1]))
    Mx, Cx, Dx = retirement.simulate(mx, retirement.solve_dcegm(mx), M0r, seed=np.random.default_rng(3))
    ax[0].plot(t, Dx.mean(axis=0), 'o-', label=f'δ = {d}'); ax[1].plot(t, Mx.mean(axis=0), 'o-', label=f'δ = {d}')
mx = model((theta0r[0], 0.9))
Mx, Cx, Dx = retirement.simulate(mx, retirement.solve_dcegm(mx), M0r, seed=np.random.default_rng(3))
ax[0].plot(t, Dx.mean(axis=0), 's--', label=f'δ = {theta0r[0]}, β = 0.9'); ax[1].plot(t, Mx.mean(axis=0), 's--', label=f'δ = {theta0r[0]}, β = 0.9')
ax[0].set(xlabel='period', title='share working by period'); ax[1].set(xlabel='period', title='mean wealth by period')
for a in ax: a.legend(); a.grid(color='0.9')
plt.show()
```

- With $\delta = 0.25$ everyone works until the last period; with $\delta = 1$ two
  thirds retire at once. The share working is the moment that sees $\delta$, and the
  wealth profile barely moves with it.
- With $\beta = 0.9$ the share working starts lower and falls more slowly, while wealth
  is run down visibly faster. The wealth profile is the moment that sees $\beta$, and the
  work shares see it too, in a different pattern from $\delta$.

:::{div}
:class: discussion

- Why does the share working fall more slowly with $\beta = 0.9$ even though it starts
  lower?
- Which of the two profiles would move if $\sigma$ doubled, and in which direction?
:::

(msm-practice-moments)=
### Moments

The two profiles *are* the moments: the share working in periods $0, \dots, T-2$
and mean wealth in periods $1, \dots, T-1$, $q = 18$ numbers. The last period of work
status is left out because it is zero whatever the parameters, and the first period of
wealth because it is the observed initial condition. The simulator draws fresh initial
wealth from the known distribution, with the same seed at every $\theta$, and the bounds
keep the optimizer where the solver works.

```{code-cell} python3
RET = {**{f'share working t={t}': (lambda M, C, D, t=t: D[:, t]) for t in range(mr.T - 1)},
       **{f'mean wealth t={t}': (lambda M, C, D, t=t: M[:, t]) for t in range(1, mr.T)}}
Ar = list(RET)
def retirement_simulate_at(theta, S=5, seed=1):
    '''Simulate S copies of the data at theta, initial wealth included, on fixed draws'''
    delta, beta = theta
    if not (0 < delta < 5 and 0.5 < beta < 1): return None   # outside: no solution, or no retirement at all
    m = model(theta)
    rng = np.random.default_rng(seed)
    return retirement.simulate(m, retirement.solve_dcegm(m), rng.uniform(2, 20, S * Nr), seed=rng)
mhatr, Omegar, simmomr, critr = msm.setup(RET, Ar, (Md, Cd_r, Dd), retirement_simulate_at)
q, K = mhatr.size, 2
Wr = msm.wdiag(Omegar)
print(f'{q} moments, {K} parameters; standard errors of the data moments:')
print(f'  work status  {np.round(np.sqrt(np.diag(Omegar)[:9] / Nr), 3)}')
print(f'  wealth       {np.round(np.sqrt(np.diag(Omegar)[9:] / Nr), 2)}')
t0 = time.perf_counter(); Q0 = critr(theta0r, Wr)
print(f'criterion at the truth {Q0:.4f}, one evaluation with S = 5 in {time.perf_counter() - t0:.3f} seconds')
```

The two kinds of moments live on different scales: a residual in a work share is of
order $0.01$, a residual in mean wealth of order $0.1$, and the inverse variances put
them on the same footing. With the identity matrix the wealth residuals would decide
everything and the work shares nothing.

(msm-practice-simulator)=
### The stepped criterion

The figure cuts the criterion along $\delta$ and along $\beta$ through the truth, and
zooms in on a window of $\delta$ one thousandth wide.

```{code-cell} python3
:tags: [hide-input]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
dg, bg, dz = np.linspace(0.3, 0.7, 41), np.linspace(0.9, 0.99, 37), np.linspace(0.4995, 0.5005, 51)
for S_, col in ((1, 'C0'), (5, 'C1')):
    ax[0].plot(dg, [critr([d, theta0r[1]], Wr, S_) for d in dg], color=col, lw=1, label=f'S = {S_}')
    ax[1].plot(bg, [critr([theta0r[0], b], Wr, S_) for b in bg], color=col, lw=1)
    ax[2].plot(dz, [critr([d, theta0r[1]], Wr, S_) for d in dz], '.-', color=col, ms=3, lw=0.7)
ax[0].set(xlabel='δ', yscale='log', title=f'Criterion along δ, β = {theta0r[1]}')
ax[1].set(xlabel='β', yscale='log', title=f'Criterion along β, δ = {theta0r[0]}')
ax[2].set(xlabel='δ', title='Zoom: the steps of the indicator moments')
ax[0].legend()
for a in ax: a.axvline(theta0r[0] if a is not ax[1] else theta0r[1], color='k', lw=0.5, ls=':'); a.grid(color='0.9')
plt.show()
```

- Both cuts have a clear minimum at the truth, three orders of magnitude deep: $\delta$
  and $\beta$ are identified by these moments.
- The zoom shows **steps**: the work status is an indicator $\mathbf 1(\tilde u < P)$, so
  a simulated share only changes when a fixed draw crosses the moving threshold, by
  $1/(SN)$ at a time. The criterion is **piecewise flat**, smaller steps with $S = 5$
  than with $S = 1$, and they never disappear.
- The steps are why `estimate` uses Nelder–Mead and why `jacobian` takes finite
  differences with a step $h = 10^{-2}$, wide enough to average over many of them: with
  $h = 10^{-3}$ the derivative of a work share is one step divided by $h$, a number
  that has nothing to do with the slope.

(msm-practice-estimation)=
### Estimation and standard errors

The estimator under the three weighting matrices, and the fit of the moments as a
figure: data moments with two standard error bands against the simulated moments at the
optimal estimate.

```{code-cell} python3
t0 = time.perf_counter()
WSr = {'identity': np.eye(q), 'diagonal': Wr, 'optimal': np.linalg.inv(Omegar)}   # the three weighting matrices
resr = {name: msm.fit(critr, simmomr, (0.3, 0.9), W, Omegar, Nr) for name, W in WSr.items()}   # the same poor start for all
msm.table(resr, ('delta', 'beta'))
print(f'three runs in {time.perf_counter() - t0:.0f} s')
th_r = resr['optimal'][0]                                         # the estimate with the optimal W, used below
```

```{code-cell} python3
:tags: [hide-input]

ms = simmomr(th_r); se_m = np.sqrt(np.diag(Omegar) / Nr)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, sl, off, lab in ((ax[0], slice(0, 9), 0, 'share working'), (ax[1], slice(9, 18), 1, 'mean wealth')):
    tt = np.arange(9) + off
    a.errorbar(tt, mhatr[sl], yerr=2 * se_m[sl], fmt='o', ms=4, capsize=3, label='data, ± 2 s.e.')
    a.plot(tt, ms[sl], 's--', ms=4, lw=0.8, color='C3', label='model at the estimate')
    a.set(xlabel='period', title=f'Fit of the moments: {lab}'); a.grid(color='0.9'); a.legend()
plt.show()
```

With the diagonal matrix the estimate lands within a tenth of a standard error of the
truth in both parameters. The optimal matrix has standard errors three times smaller
for $\delta$ and five times smaller for $\beta$. That is what the optimal $W$ buys when
the moments are correlated, as eighteen moments from an absorbing retirement process and
a persistent wealth path are: it weights the *differences* between neighboring periods,
which the diagonal $W$ cannot see. The identity lets the wealth residuals, a hundred
times larger than those of the work shares, decide on their own, and $\delta$ is the
parameter that suffers. The figure shows both profiles matched within the bands in
every period at the optimal estimate. A caution from {cite:t}`altonji1996SmallSampleBias`
applies to that column: $\hat\Omega$ has a condition number of $2 \cdot 10^4$ and is
estimated from one thousand individuals, and its inverse is estimated with error that
is correlated with the moment conditions.

### Moments from choice-specific values

The steps come from the simulated work status, an indicator. The model also produces,
at every state, the two values behind the retirement decision, and the **value of
working** is a smooth function of the parameters given the state: it carries $\delta$
directly, through the disutility of work, and $\beta$ through the continuation. The
experiment below replaces every work share in the library by the mean value of working
in that period, which the simulator returns for every worker at the wealth she holds,
retirees included, and keeps the wealth moments. Such moments are not observable, a
value is a model object, so the data moments come from the simulation of the data by the
true model, which only synthetic data allow. The question is what a smooth contribution does to the steps and to the
precision.

```{code-cell} python3
RETV = {**{f'value of working t={t}': (lambda M, C, V, t=t: V[:, t]) for t in range(mr.T - 1)},   # in place of the work shares
        **{f'mean wealth t={t}': (lambda M, C, V, t=t: M[:, t]) for t in range(1, mr.T)}}        # the wealth moments as before
Av = list(RETV)
def retirement_simulate_values(theta, S=5, seed=1):
    '''Like retirement_simulate_at, with the value of working on the simulated paths in place of the work status'''
    delta, beta = theta
    if not (0 < delta < 5 and 0.5 < beta < 1): return None   # the bounds
    m = model(theta)                                      # the model at theta
    rng = np.random.default_rng(seed)                     # fixed draws
    Ms, Cs, Ds, VW, VR = retirement.simulate(m, retirement.solve_dcegm(m), rng.uniform(2, 20, S * Nr), seed=rng, values=True)   # the paths, with the values on them
    return Ms, Cs, VW                                     # the value of working at every point of every path
mhatv, Omegav, simmomv, critv = msm.setup(RETV, Av, (Md, Cd_r, VWd), retirement_simulate_values)
Wv = msm.wdiag(Omegav)
print(f'condition number of Omega: {np.linalg.cond(Omegav):.2g}, against {np.linalg.cond(Omegar):.2g} with the work shares')
```

```{code-cell} python3
:tags: [hide-input]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for S_, col in ((1, 'C0'), (5, 'C1')):
    ax[0].plot(dg, [critv([d, theta0r[1]], Wv, S_) for d in dg], color=col, lw=1, label=f'S = {S_}')
    ax[1].plot(bg, [critv([theta0r[0], b], Wv, S_) for b in bg], color=col, lw=1)
    ax[2].plot(dz, [critv([d, theta0r[1]], Wv, S_) for d in dz], '.-', color=col, ms=3, lw=0.7)
ax[0].set(xlabel='δ', yscale='log', title=f'Criterion along δ, β = {theta0r[1]}, values of working')
ax[1].set(xlabel='β', yscale='log', title=f'Criterion along β, δ = {theta0r[0]}, values of working')
ax[2].set(xlabel='δ', title='Zoom: the same window as above')
ax[0].legend()
for a in ax: a.axvline(theta0r[0] if a is not ax[1] else theta0r[1], color='k', lw=0.5, ls=':'); a.grid(color='0.9')
plt.show()
```

```{code-cell} python3
t0 = time.perf_counter()
WSv = {'identity': np.eye(len(Av)), 'diagonal': Wv, 'optimal': np.linalg.inv(Omegav)}   # the three weighting matrices
resv = {name: msm.fit(critv, simmomv, (0.3, 0.9), W, Omegav, Nr) for name, W in WSv.items()}   # the same poor start as before
msm.table(resv, ('delta', 'beta'))
print(f'three runs in {time.perf_counter() - t0:.0f} s')
D_shares, D_values = msm.jacobian(simmomr, th_r, 5, h=1e-3), msm.jacobian(simmomv, resv['optimal'][0], 5, h=1e-3)
print('Jacobian with h = 1e-3, first three rows, work shares (left) against values of working (right):')
print(np.round(np.column_stack([D_shares[:3], D_values[:3]]), 2))
```

- The staircase is **still there**. In the zoom the criterion with one copy of the data
  moves in steps as large as with the work shares, and five copies shrink them without
  removing them. The value of working is smooth in the parameters at a given state, but
  the state is a simulated wealth path, and every draw that crosses its threshold still
  moves the path and every value along it.
- The values carry more information about $\delta$. The Jacobian entries of the value
  moments are an order of magnitude larger than those of the shares, and of one sign
  across periods, because the disutility of work enters every value directly. Under the
  identity the standard error of $\delta$ falls from 0.016 to 0.004, under the optimal
  matrix from 0.003 to 0.002, and $\beta$ is as precise as before.
- The price is in $\hat\Omega$. The values of working of one worker in neighboring
  periods are nearly the same number, the condition number rises from $2 \cdot 10^4$ to
  $10^6$, and the optimal matrix leans on an inverse that a thousand workers determine
  poorly. And none of it exists outside a simulation: the values belong to the model, not
  to the data. The remedy that applies to real data is the smoothing of the simulated
  choices themselves, the generalized indirect inference of
  {cite:t}`bruins2018GeneralizedIndirectInference`.

:::{div}
:class: discussion

- Where exactly along a wealth path does one draw crossing the threshold change the
  simulated moments, and does the effect stay in one period?
- The value of working is not in any data set. Which observable moment comes closest
  to it, and what would it cost in precision?
:::

(msm-practice-sigma)=
### What breaks: the taste shock scale

The scale $\sigma$ of the taste shocks was fixed at its true value. Adding it to the
parameter vector is one line in `model` and one in the bounds, and it is where the
method shows its limits. The criterion along $\sigma$ through the truth:

```{code-cell} python3
def model3(theta):
    '''The retirement model with the taste shock scale as a third parameter'''
    delta, beta, sigma = theta
    return retirement.retirement(beta=beta, delta=delta, **{**FIXED_R, 'sigma': sigma})   # the estimated scale replaces the fixed one
def retirement3_simulate_at(theta, S=5, seed=1):
    '''The simulator with three parameters'''
    if not (0 < theta[0] < 5 and 0.5 < theta[1] < 1 and theta[2] > 0): return None
    m = model3(theta)
    rng = np.random.default_rng(seed)
    return retirement.simulate(m, retirement.solve_dcegm(m), rng.uniform(2, 20, S * Nr), seed=rng)
_, _, simmom3, crit3 = msm.setup(RET, Ar, (Md, Cd_r, Dd), retirement3_simulate_at)
for s in (0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2):
    print(f'sigma {s:<5} criterion {crit3([theta0r[0], theta0r[1], s], Wr):.4f}')
```

Compare with the cuts along $\delta$ and $\beta$ above, which climb more than an order of
magnitude within a few hundredths of the truth. Along $\sigma$ the minimum is at the
truth, $0.007$, but the criterion is **flat**: it rises to $0.02$ at $\sigma = 0.08$ and
$\sigma = 0.15$, and to $0.06$–$0.07$ only at $\sigma = 0.05$ and $\sigma = 0.2$. The moments
barely respond to $\sigma$, which is the small entry in $D$ that the
[asymptotics of Class 19](19_msm.md#msm-asymptotics) warned about:

- the standard error of $\sigma$ is large, and the finite-difference Jacobian in that
  direction is dominated by the steps, so even the reported standard error is unreliable
- Nelder–Mead estimating all three parameters stalls on the flat, stepped criterion:
  started from $\sigma = 0.05$ it stops at $\sigma = 0.068$, from $\sigma = 0.2$ at
  $\sigma = 0.275$, both with a criterion several times above the minimum, and a restart
  does not help; only a start near the truth finds $\sigma \approx 0.10$
- on other data sets the sampling noise of eighteen moments can move the flat minimum
  away from the truth, because the weak signal of $\sigma$ cannot pull it back

What would identify $\sigma$ is a moment that measures how retirement varies *given*
wealth: the dispersion of the retirement age among workers with similar initial wealth,
for instance.

<!-- ````{warning} Practical task: MSM estimation of the retirement model

(task-msm-practice)=
Pull the class code and open the notebook `msm_practice.ipynb` in `session20-nov3/`:

```bash
cd sb-dse-code && git pull
# first time: git clone https://github.com/fediskhakov/sb-dse-code.git
```

1. Fill in `moments`, `simmoments` and `criterion` and reproduce the estimates under the
   three weighting matrices and the standard errors of this page.
2. Run the estimator with the diagonal weighting matrix and $S = 1$, $5$ and $20$. Report the estimates, the standard
   errors and the run time, and compare the change in the standard errors with the
   factor $\sqrt{1 + 1/S}$.
3. With the identity weighting matrix, which moments drive the estimate, and why does
   $\delta$ suffer more than $\beta$?
4. Add $\sigma$ as a third parameter with `model3`. Start Nelder–Mead from
   $(0.3, 0.9, 0.2)$ and from $(0.7, 0.98, 0.05)$, with a restart each, and compare the
   two solutions and their standard errors with the Jacobian step $h = 10^{-3}$ and
   $h = 10^{-2}$.
5. Add moments that respond to $\sigma$: the mean and the standard deviation of the
   first period of retirement within each quartile of initial wealth, eight numbers,
   written as contributions whose mean over all individuals is the quartile statistic.
   Does the criterion along $\sigma$ become steeper, and does the estimate of $\sigma$
   settle?
6. [optional] Replace the simulated work shares by the choice probabilities, built with
   `m.choice_prob` from the values of working and of retiring that
   `retirement.simulate(..., values=True)` returns, keeping the data moments as they are. Do the steps
   in the zoomed criterion disappear, and does the estimate change?
7. [optional] Condition on the data: simulate from the *observed* initial wealth,
   `np.tile(M0, S)`, instead of fresh draws, and keep `M0` fixed across the Monte Carlo
   replications. The Monte Carlo spread falls well below the formula. Which variation
   does $\hat\Omega$ count that the conditioning has removed, and how would you
   estimate the covariance of the contributions *given* initial wealth?

Whatever is left unfinished is finished at home; nothing is collected. Use of AI
assistance is allowed and encouraged, subject to the
[course AI policy](index.md), but you must be able to explain every
number you report.
```` -->

````{note} References and additional resources

(20_msm_practice_references)=
- 📖 {cite:t}`egm` "The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks" — the retirement model, and its Monte Carlo estimation experiments in Section 5
- 📖 {cite:t}`bruins2018GeneralizedIndirectInference` "Generalized indirect inference for discrete choice models" — smoothing the simulated discrete choices so that the criterion becomes differentiable: the fix for the staircase of this class
- 📄 `scipy.optimize.minimize`, method Nelder–Mead <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html>
````
