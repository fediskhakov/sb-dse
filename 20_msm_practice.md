---
title: "🔬 Practice: MSM estimation of the continuous choice models"
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
import msm                                                    # the MSM estimator and its machinery (Classes 19, 20)
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

`msm.estimate` minimizes by the method of {cite:t}`nelder1965SimplexMethodFunction`. It
works on a **simplex** of $K + 1$ points, a triangle for two parameters. One iteration:

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
absorbing. The data are 100 workers with initial wealth uniform on $[2, 20]$, taste
shocks $\sigma = 0.1$ and income shocks $\sigma_y = 0.2$, over $T = 10$ periods. The
parameters we pretend not to know:

| | true value | estimated |
| :-- | :-- | :-- |
| disutility of work $\delta$ | $0.5$ | yes |
| discount factor $\beta$ | $0.95$ | yes |
| taste shock scale $\sigma$ | $0.1$ | fixed, see [the taste shock scale](#msm-practice-sigma) |
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
Nr = 100                                                  # workers in the data
rng = np.random.default_rng(3)                            # the seed of the data: initial wealth and all shocks
M0r = rng.uniform(2, 20, Nr)                              # initial wealth of the sample
mr = model(theta0r)                                       # the true model
solr = retirement.solve_dcegm(mr)                         # its solution: sol[t][d] for every period and choice
Md, Cd_r, Dd, VWd = retirement.simulate(mr, solr, M0r, seed=rng)   # the data: wealth, consumption, work status and value of working, Nr x T each
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
    Mx, Cx, Dx, _ = retirement.simulate(mx, retirement.solve_dcegm(mx), M0r, seed=np.random.default_rng(3))
    ax[0].plot(t, Dx.mean(axis=0), 'o-', label=f'δ = {d}'); ax[1].plot(t, Mx.mean(axis=0), 'o-', label=f'δ = {d}')
mx = model((theta0r[0], 0.9))
Mx, Cx, Dx, _ = retirement.simulate(mx, retirement.solve_dcegm(mx), M0r, seed=np.random.default_rng(3))
ax[0].plot(t, Dx.mean(axis=0), 's--', label=f'δ = {theta0r[0]}, β = 0.9'); ax[1].plot(t, Mx.mean(axis=0), 's--', label=f'δ = {theta0r[0]}, β = 0.9')
ax[0].set(xlabel='period', title='share working by period'); ax[1].set(xlabel='period', title='mean wealth by period')
for a in ax: a.legend(); a.grid(color='0.9')
plt.show()
```

- With $\delta = 0.25$ nearly everyone works until the last two periods; with
  $\delta = 1$ two thirds retire at once. The share working is the moment that sees $\delta$, and the
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
RET = {**{f'share working t={t}': (lambda M, C, D, V, t=t: D[:, t]) for t in range(mr.T - 1)},
       **{f'mean wealth t={t}': (lambda M, C, D, V, t=t: M[:, t]) for t in range(1, mr.T)}}
Ar = list(RET)
def retirement_simulate_at(theta, S=5, seed=1):
    '''Simulate S copies of the data at theta, initial wealth included, on fixed draws'''
    delta, beta = theta
    if not (0 < delta < 5 and 0.5 < beta < 1): return None   # outside: no solution, or no retirement at all
    m = model(theta)
    rng = np.random.default_rng(seed)
    return retirement.simulate(m, retirement.solve_dcegm(m), rng.uniform(2, 20, S * Nr), seed=rng)
mhatr, Omegar, simmomr, critr = msm.setup(RET, Ar, (Md, Cd_r, Dd, VWd), retirement_simulate_at)
q, K = mhatr.size, 2
Wr = msm.wdiag(Omegar)
print(f'{q} moments, {K} parameters; standard errors of the data moments:')
print(f'  work status  {np.round(np.sqrt(np.diag(Omegar)[:9] / Nr), 3)}')
print(f'  wealth       {np.round(np.sqrt(np.diag(Omegar)[9:] / Nr), 2)}')
t0 = time.perf_counter(); Q0 = critr(theta0r, Wr)
print(f'criterion at the truth {Q0:.4f}, one evaluation with S = 5 in {time.perf_counter() - t0:.3f} seconds')
```

The two kinds of moments live on different scales: the standard error of a work share is
about $0.05$, that of mean wealth up to $0.5$, and the inverse variances put them on the
same footing. With the identity matrix the wealth residuals would decide everything and
the work shares nothing.

(msm-practice-simulator)=
### The stepped criterion

The figure cuts the criterion along $\delta$ and along $\beta$ through the truth, and
zooms in on a window of $\delta$ one thousandth wide.

```{code-cell} python3
:tags: [hide-input]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
dg, bg, dz = np.linspace(0.4, 0.6, 151), np.linspace(0.94, 0.96, 151), np.linspace(0.4995, 0.5005, 71)
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

- Both cuts have a clear minimum: the criterion is ten to twenty times higher at the ends
  of the window in $\delta$, and two to eight times in the narrower one in $\beta$, so
  $\delta$ and $\beta$ are identified by these moments. With a hundred workers the minimum
  sits where the data put it, a few thousandths from the truth, not at the truth.
- The zoom shows **steps**: the work status is an indicator $\mathbf 1(\tilde u < P)$, so
  a simulated share only changes when a fixed draw crosses the moving threshold, by
  $1/(SN)$ at a time. The criterion is **piecewise flat**, smaller steps with $S = 5$
  than with $S = 1$, and they never disappear.
- The steps are why `estimate` uses Nelder–Mead and why `jacobian` takes finite
  differences with a step $h = 10^{-2}$, wide enough to average over many of them: with
  $h = 10^{-3}$ the derivative of a work share is one step divided by $h$, a number
  that has nothing to do with the slope. The [smooth simulator](#msm-practice-smooth)
  below removes the steps instead of averaging over them.

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

With the diagonal matrix the estimate lands within one standard error of the truth in
both parameters, and the figure shows both profiles matched within the bands in every
period at the optimal estimate.

- The optimal matrix has standard errors four times smaller for both parameters. That is
  what the optimal $W$ buys when the moments are correlated, as eighteen moments from an
  absorbing retirement process and a persistent wealth path are: it weights the
  *differences* between neighboring periods, which the diagonal $W$ cannot see.
- The identity lets the wealth residuals, up to ten times larger than those of the work
  shares, decide on their own, and $\delta$ is the parameter that suffers.
- A caution from {cite:t}`altonji1996SmallSampleBias` applies to the optimal column:
  $\hat\Omega$ has a condition number of $4 \cdot 10^4$ and is estimated from one hundred
  individuals, and its inverse is estimated with error that is correlated with the
  moment conditions.

(msm-practice-smooth)=
### Smooth simulator

The steps come from the simulator, not from the moments. A simulated work status is an
indicator $\mathbf 1(\tilde u < P)$, and a smooth function of the simulated paths does not
remove it:

- one draw that crosses its threshold moves the retirement date of one worker
- the retirement date stops the income, so her whole wealth path after it moves
- every later moment of that worker moves with the path, wealth and any value at that
  wealth alike

What is smooth in the parameters is the *expectation* over the taste shocks. Retirement
is absorbing and the choice probabilities are logit, so that expectation can be computed
exactly instead of simulated.

```{attention} Smooth simulator
Given the initial wealth $M_0$ and the income draws of a worker, her life is a set of
$T + 1$ branches: the **working branch**, with wealth $M^w_t$, and one **retired branch**
per retirement date $k$, with wealth $M^{(k)}_t$. With $P_t = P(\text{work} \mid M^w_t)$,
the probabilities of being still working after period $t$ and of retiring in period $k$ are

$$
p_t = p_{t-1} P_t, \quad p_{-1} = 1, \qquad \pi_k = p_{k-1} (1 - P_k).
$$

The **smooth simulator** replaces every simulated outcome by its expectation over the
branches: the work status in period $t$ by $p_t$, and a function $h$ of the state by

$$
\bar h_t = p_{t-1}\, h(M^w_t) + \sum_{k < t} \pi_k\, h(M^{(k)}_t).
$$
```

The branches of one worker at the true parameters, with initial wealth 14 and one draw
of her income, as a flow of probability from period to period:

```{code-cell} python3
:tags: [hide-input]

rng = np.random.default_rng(7)                            # one worker: initial wealth 14 and her income draws
y = mr.y * np.exp(mr.sigma_y * rng.standard_normal(mr.T) - mr.sigma_y**2 / 2)
Mw, p, pi = 14.0, [1.0], []                               # wealth of the working branch, p_t from p_{-1} = 1, pi_k
for t in range(mr.T):
    P = float(mr.choice_prob(mr.value(solr[t][1], 1, Mw), mr.value(solr[t][0], 0, Mw)))   # probability to keep working
    pi.append(p[-1] * (1 - P)); p.append(p[-1] * P)       # a new retired branch, and the working branch that continues
    if t < mr.T - 1: Mw = float(mr.next_wealth(Mw - mr.policy(solr[t][1], Mw), 1, y[t]))
top, gap = 1.25, 0.25                                     # the working stream hangs from top, the retired stack sits on zero
def band(x0, a0, b0, x1, a1, b1, color):
    '''A flow from the interval [a0, b0] at x0 to [a1, b1] at x1, along a smooth S-curve'''
    x = np.linspace(x0, x1, 40); s = (1 - np.cos(np.pi * (x - x0) / (x1 - x0))) / 2
    ax.fill_between(x, a0 + (a1 - a0) * s, b0 + (b1 - b0) * s, color=color, alpha=0.75, lw=0)
fig, ax = plt.subplots(figsize=(14, 5))
col = plt.cm.autumn(np.linspace(0, 0.85, mr.T))           # one color per retirement date
cum = np.r_[0, np.cumsum(pi)]                             # the retired stack: branch k on [cum[k], cum[k + 1]]
for t in range(mr.T):                                     # flows from the state after t - 1 (x = t - 1) to the state after t (x = t)
    band(t - 1, top - p[t + 1], top, t, top - p[t + 1], top, 'C0')            # those who keep working
    band(t - 1, top - p[t], top - p[t + 1], t, cum[t], cum[t + 1], col[t])      # those who retire in t: a new branch
    for k in range(t):                                                          # the branches retired before: they stay
        band(t - 1, cum[k], cum[k + 1], t, cum[k], cum[k + 1], col[k])
for t in range(mr.T):
    if pi[t] > 0.03: ax.text(mr.T - 0.9, (cum[t] + cum[t + 1]) / 2, f'retired in {t}: $\\pi_{t}$ = {pi[t]:.2f}', va='center', fontsize=9)
    ax.text(t, top + 0.04, f'{p[t + 1]:.2f}', ha='center', fontsize=9, color='C0')
ax.text(-1, top + 0.04, 'start', ha='center', fontsize=9, color='C0')
ax.text(-0.5, top - 0.3, 'working', ha='center', color='white', fontsize=11)
ax.set(xlim=(-1.2, mr.T + 1.3), ylim=(-0.05, top + 0.15), yticks=[], xticks=range(-1, mr.T),
       xticklabels=['start'] + [f't={t}' for t in range(mr.T)], title='The branches of one worker: working on top, with $p_t$ above, retired branches below')
for s in ('left', 'right', 'top'): ax.spines[s].set_visible(False)
plt.show()
```

The working stream on top shrinks by the share $1 - P_t$ every period, and what leaves
it opens a retired branch that stays retired to the end, with the probability $\pi_k$ of
the definition. Every column sums to one: the smooth simulator carries the whole
distribution of retirement dates, where the stepped one draws a single date.

Why does this remove the steps?

- The weights are products of logit probabilities, smooth in $\theta$, and every branch
  moves along the continuous policies of its own choice. There is no indicator left.
- The stepped simulator averages to the same numbers: $\bar h_t$ is the expectation of
  its outcome given $M_0$ and income, so the moment conditions are the same, with the
  noise of the taste shocks removed.

The algorithm runs the branches forward period by period, opening one new retired branch
every period:

```
Input: the solution of the model at θ; initial wealth M0 and income draws y of every worker
Algorithm:
  working branch: wealth Mw = M0, probability p = 1; no retired branches
  for t = 0, ..., T-1:
    record the expectations over the branches of wealth, consumption, value of working
    P = probability to keep working at Mw
    open a retired branch: probability p (1 - P), savings of retiring now
    p = p P, record it as the expected work status
    move Mw with the worker policy and income y_t, every retired branch with the retiree policy
Output: expected wealth, consumption, work status and value of working, N x T
```

`retirement.simulate(..., smooth=True)` implements it and returns the same four arrays as
the stepped simulator, so the moment library `RET` and the data stay as they are; only
the simulator changes. With the same seed both use the same initial wealth and income.

Where does it break?

- It needs the choice probabilities in closed form, logit here.
- It needs few branches. An absorbing choice gives $T + 1$ per worker, and a simulation
  costs about twice the stepped one. A choice that can be reversed every period gives
  $2^T$, and the remedy is then to smooth the simulated indicators instead, the
  generalized indirect inference of {cite:t}`bruins2018GeneralizedIndirectInference`.
- It needs continuous policies. With taste shocks close to zero the worker's consumption
  jumps at the kinks of the [upper envelope of Class 18](18_dcegm.md#dcegm-upper-envelope),
  and a branch whose wealth crosses a kink jumps with it.

```{code-cell} python3
def retirement_simulate_smooth(theta, S=5, seed=1):
    '''Like retirement_simulate_at with the taste shocks integrated out: the expectations of the same four arrays'''
    delta, beta = theta
    if not (0 < delta < 5 and 0.5 < beta < 1): return None   # the bounds
    m = model(theta)                                      # the model at theta
    rng = np.random.default_rng(seed)                     # fixed draws: initial wealth and income
    return retirement.simulate(m, retirement.solve_dcegm(m), rng.uniform(2, 20, S * Nr), seed=rng, smooth=True)
_, _, simmoms, crits = msm.setup(RET, Ar, (Md, Cd_r, Dd, VWd), retirement_simulate_smooth)   # the same moments and data
t0 = time.perf_counter(); ms_step = simmomr(theta0r, 50)  # both simulators at the truth, on the same draws of wealth and income
t1 = time.perf_counter(); ms_smooth = simmoms(theta0r, 50)
t2 = time.perf_counter(); se_m = np.sqrt(np.diag(Omegar) / Nr)   # standard errors of the data moments
print(f'largest gap between the simulators at the truth, S = 50: {np.max(np.abs(ms_step - ms_smooth) / se_m):.2f} s.e. of the data moments')
print(f'one evaluation with S = 50: stepped {t1 - t0:.2f} s, smooth {t2 - t1:.2f} s')
```

```{code-cell} python3
:tags: [hide-input]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
dg, bg, dz = np.linspace(0.4, 0.6, 151), np.linspace(0.94, 0.96, 151), np.linspace(0.4995, 0.5005, 71)
for crit, S_, col, lab in ((crits, 1, 'C2', 'smooth, S = 1'), (crits, 5, 'C1', 'smooth, S = 5')):
    ax[0].plot(dg, [crit([d, theta0r[1]], Wr, S_) for d in dg], color=col, lw=1, label=lab)
    ax[1].plot(bg, [crit([theta0r[0], b], Wr, S_) for b in bg], color=col, lw=1)
    ax[2].plot(dz, [crit([d, theta0r[1]], Wr, S_) for d in dz], '.-', color=col, ms=3, lw=0.7)
ax[0].set(xlabel='δ', yscale='log', title=f'Criterion along δ, β = {theta0r[1]}')
ax[1].set(xlabel='β', yscale='log', title=f'Criterion along β, δ = {theta0r[0]}')
ax[2].set(xlabel='δ', title='Zoom: the same window as above')
ax[0].legend()
for a in ax: a.axvline(theta0r[0] if a is not ax[1] else theta0r[1], color='k', lw=0.5, ls=':'); a.grid(color='0.9')
plt.show()
```

```{code-cell} python3
t0 = time.perf_counter()
ress = {name: msm.fit(crits, simmoms, (0.3, 0.9), W, Omegar, Nr) for name, W in WSr.items()}   # the three weighting matrices of the work shares
msm.table(ress, ('delta', 'beta'))
print(f'three runs in {time.perf_counter() - t0:.0f} s')
for name, sm, th in (('stepped', simmomr, th_r), ('smooth', simmoms, ress['optimal'][0])):   # does the derivative depend on the step?
    D2, D3 = msm.jacobian(sm, th, 5, h=1e-2), msm.jacobian(sm, th, 5, h=1e-3)
    print(f'{name:<8} largest change of the Jacobian from h = 1e-2 to h = 1e-3, relative: {np.max(np.abs(D2 - D3)) / np.max(np.abs(D2)):.3f}')
```

- The two simulators agree. With fifty copies of the data, where the taste-shock noise
  of the stepped one averages out, their moments at the truth differ by less than a tenth
  of a standard error of the data moments.
- The staircase is **gone**. In the zoom the smooth criterion is a straight line with
  $S = 1$ as with $S = 5$, where the [stepped one](#msm-practice-simulator) jumps. Its Jacobian changes by 0.1%
  from $h = 10^{-2}$ to $h = 10^{-3}$, that of the stepped simulator by 13%, so the
  standard errors of the smooth simulator are the ones to trust.
- The estimates are those of the stepped simulator within a fraction of a standard
  error, under all three weighting matrices. Nothing about the data changed, only the
  noise in the simulation: this is the remedy that applies to real data.

:::{div}
:class: discussion

- Where exactly along a wealth path does one draw crossing the threshold change the
  stepped moments, and does the effect stay in one period?
- The smooth simulator uses the logit formula for the probability to keep working. What
  would change with normally distributed taste shocks?
:::

(msm-practice-sigma)=
### The taste shock scale

The scale $\sigma$ of the taste shocks has been fixed at its true value so far. Estimated
together with $\delta$, with $\beta$ fixed at its true value, it shows what **poor
identification** looks like, and with two parameters the moment figures of
[Part 1](#msm-practice-deaton) apply as they are. The smooth simulator keeps the steps out
of the picture, so what remains is about the moments, not about the simulation.

Three sets of moments, as with the Deaton model: all eighteen, the work shares alone,
and the wealth profile alone.

```{code-cell} python3
theta0s = np.array([0.1, 0.5])                            # true sigma and delta; beta fixed at its true value
def retirement_simulate_sd(theta, S=5, seed=1):
    '''The smooth simulator with the taste shock scale and the disutility of work as the parameters'''
    sigma, delta = theta
    if not (0 < sigma < 1 and 0 < delta < 5): return None   # the bounds
    m = retirement.retirement(beta=theta0r[1], delta=delta, **{**FIXED_R, 'sigma': sigma})   # the estimated scale replaces the fixed one
    rng = np.random.default_rng(seed)
    return retirement.simulate(m, retirement.solve_dcegm(m), rng.uniform(2, 20, S * Nr), seed=rng, smooth=True)
grids_s, pnames_s = (np.linspace(0.02, 0.3, 29), np.linspace(0.4, 0.6, 21)), ('σ', 'δ')
SH, WL = Ar[:9], Ar[9:]                                   # the work shares, the wealth profile
mhats, Omegas, simmoms2, crits2 = msm.diagnostics(RET, Ar, (Md, Cd_r, Dd, VWd), retirement_simulate_sd, Nr, theta0s, grids_s, pnames_s)
```

```{code-cell} python3
_ = msm.diagnostics(RET, SH, (Md, Cd_r, Dd, VWd), retirement_simulate_sd, Nr, theta0s, grids_s, pnames_s)
```

```{code-cell} python3
_ = msm.diagnostics(RET, WL, (Md, Cd_r, Dd, VWd), retirement_simulate_sd, Nr, theta0s, grids_s, pnames_s)
```

- **All eighteen** have a trough instead of a basin. Along $\delta$ the moments sweep to
  five standard errors within $0.1$ of the truth; along $\sigma$ every one stays within
  two, from $\sigma = 0.02$, nearly no taste shocks, to three times the truth. The bottom
  of the trough is at $\sigma \approx 0.15$, not at the truth.
- **The work shares** carry what there is to know about $\sigma$. With small shocks fewer
  workers keep working in the first period, with large ones fewer in the last, each by
  about two standard errors at the ends of the range. Their trough is shorter, and its
  bottom is near the truth.
- **The wealth profile** sees neither parameter well. Only wealth in the last period
  responds to $\delta$, and the minimum of its map lies at $\sigma \approx 0.22$.

The scale $\sigma$ decides how sharply retirement follows wealth: with small shocks every
worker retires at the wealth where working stops paying, with large ones the dates spread
out. The work shares average over initial wealth uniform on $[2, 20]$, which spreads the
retirement dates already, so they hardly tell the two sources apart.

:::{div}
:class: discussion

- Which moment would measure how retirement varies *given* wealth, and why might it still
  be too noisy to help in a sample of a hundred workers?
- Why do the first and the last work shares respond to $\sigma$ in opposite directions?
:::

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
4. Estimate $\beta$ along with $\sigma$ and $\delta$. Start Nelder–Mead from
   $(0.2, 0.3, 0.9)$ and from $(0.05, 0.7, 0.98)$, with a restart each, and compare the
   two solutions and their standard errors with the Jacobian step $h = 10^{-3}$ and
   $h = 10^{-2}$.
5. Add moments that respond to $\sigma$: the mean and the standard deviation of the
   first period of retirement within each quartile of initial wealth, eight numbers,
   written as contributions whose mean over all individuals is the quartile statistic.
   Does the criterion along $\sigma$ become steeper, and does the estimate of $\sigma$
   settle?
6. [optional] Replace the simulated work shares by the choice probabilities, built with
   `m.choice_prob` from the values of working and of retiring, `m.value`, at the simulated
   wealth, keeping the data moments as they are. Do the steps in the zoomed criterion
   disappear, and does the estimate change?
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
- 📖 {cite:t}`bruins2018GeneralizedIndirectInference` "Generalized indirect inference for discrete choice models" — smoothing the simulated discrete choices so that the criterion becomes differentiable: the fix for the staircase when the choices cannot be integrated out
- 📖 {cite:t}`aupens` "Effects of taxes and safety net pensions on life-cycle labor supply, savings and human capital: The case of Australia" — a life-cycle model of labor supply, savings and retirement of Australians, structurally estimated with MSM methods using smoothed simulator
- 📖 {cite:t}`nelder1965SimplexMethodFunction` "A simplex method for function minimization" — the original paper of the Nelder–Mead method that `msm.estimate` uses
- 📄 `scipy.optimize.minimize`, method Nelder–Mead <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html>
````
