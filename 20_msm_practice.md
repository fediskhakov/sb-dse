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

<span class="tag wip">🚧 🚧 🚧 Not the final version 🚧 🚧 🚧</span>

Today we estimate the [retirement model of Class 18](18_dcegm.md#retirement-model) by the
[method of simulated moments of Class 19](19_msm.md): simulated data, moments, the
criterion, the disutility of work and the discount factor recovered, and the standard
errors checked against a Monte Carlo.

````{hint} Running the code for this lecture
:class: dropdown

The code for this class is in the folder `session20-nov3/` of the course **code
repository**. We write it together in class, starting from the pre-code notebook
`msm_practice_pre.ipynb`, which has the data, the headings and the stubs of the three
functions we write live. The complete notebook `msm_practice.ipynb` has all the code of
this page. Two modules in the same folder carry what we already have: `retirement.py`,
the DC-EGM solver and simulator of [Class 18](18_dcegm.md), and `msm.py`, the generic
part of the estimator of [Class 19](19_msm.md#msm-weighting).

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

## What we have and what we write

The course architecture again: model object, solver, simulator, estimator. The first
three exist, and today's work is the fourth:

| Piece | Where | Status |
| :-- | :-- | :-- |
| `retirement`, `solve_dcegm`, `worker` | `retirement.py`, [Class 18](18_dcegm.md#dcegm-full-algorithm) | given |
| `simulate(m, sol, M0, seed)` | `retirement.py`, [Class 18](18_dcegm.md) | given, with a `prob=True` switch that also returns the choice probabilities |
| `estimate`, `jacobian`, `vcov`, `jtest` | `msm.py`, [Class 19](19_msm.md#msm-weighting) | given, model-free |
| `moments(M, C, D)` | this notebook | **we write it** |
| `simmoments(theta, S, seed)` | this notebook | **we write it** |
| `criterion(theta, W, S)` | this notebook | **we write it** |

The estimator in `msm.py` takes the criterion and the simulated moments as function
arguments, so the three functions we write are the whole interface between the model
and the method.

```{code-cell} python3
:tags: [hide-input]

import sys, time, pathlib, tempfile, urllib.request
import numpy as np
import matplotlib.pyplot as plt
CODE = pathlib.Path('../gh-code/session20-nov3')          # the DC-EGM code of Class 18 and the estimator of Class 19
if not CODE.exists():                                     # otherwise fetch the modules from the code repository
    CODE = pathlib.Path(tempfile.mkdtemp())
    url = 'https://raw.githubusercontent.com/fediskhakov/sb-dse-code/main/session20-nov3/'
    for f in ('retirement.py', 'msm.py'):
        urllib.request.urlretrieve(url + f, CODE / f)
sys.path.insert(0, str(CODE))
from retirement import retirement, solve_dcegm, simulate
from msm import estimate, jacobian, vcov, jtest
```

The "data" are simulated as in [Class 18](18_dcegm.md#dcegm-simulation), with a seed of
their own: 1000 workers with
initial wealth uniform on $[2, 20]$, taste shocks $\sigma = 0.1$ and income shocks
$\sigma_y = 0.2$, over $T = 10$ periods. The distribution of initial wealth is treated
as part of the model, known to the estimator. The parameters we pretend not to know:

| | true value | estimated |
| :-- | :-- | :-- |
| disutility of work $\delta$ | $0.5$ | yes |
| discount factor $\beta$ | $0.95$ | yes |
| taste shock scale $\sigma$ | $0.1$ | fixed, see [what breaks](#msm-practice-sigma) |
| income shock $\sigma_y$, return $R$, wage $y$ | $0.2$, $1.05$, $1$ | fixed |

```{code-cell} python3
def model(theta):
    '''The retirement model at the parameters we estimate; everything else at the values of Class 18'''
    delta, beta = theta
    return retirement(beta=beta, delta=delta, sigma=0.1, sigma_y=0.2, nquad=10)

theta0 = np.array([0.5, 0.95])                            # true delta and beta
N = 1000
rng = np.random.default_rng(3)
M0 = rng.uniform(2, 20, N)                                # initial wealth of the sample
m0 = model(theta0)
Md, Cd, Dd = simulate(m0, solve_dcegm(m0), M0, seed=rng)   # the data: wealth, consumption, work status
print(f'share working by period  {np.round(Dd.mean(0), 2)}')
print(f'mean wealth by period    {np.round(Md.mean(0), 1)}')
```

Every column is a period of life: the share working falls from 83% to zero in the last
period, where no one works, and wealth is run down from 11 to below 2.

:::{div}
:class: discussion

- Which of the two profiles would move if $\delta$ doubled, and in which direction?
- Which would move if $\beta$ fell to $0.9$, and why is the first period of each profile
  the least informative?
:::

(msm-practice-moments)=
## Moments

The two profiles above *are* the moments: the share working in periods $0, \dots, T-2$
and mean wealth in periods $1, \dots, T-1$, $q = 18$ numbers. The last period of work
status is left out because it is zero whatever the parameters, and the first period of
wealth because it is the observed initial condition.

The function returns one row per individual, the **contributions** $h(z_i)$, so that the
data moments are a column mean and their covariance $\Omega$ is one call of `np.cov`:

```{code-cell} python3
def moments(M, C, D):
    '''Moment contributions h(z_i), one row per individual: work status in periods 0..T-2, wealth in periods 1..T-1'''
    return np.column_stack([D[:, :-1], M[:, 1:]])

h = moments(Md, Cd, Dd)                                   # N x q
mhat, Omega = h.mean(0), np.cov(h.T)                      # data moments and their covariance
q, K = mhat.size, theta0.size
Wdiag = np.diag(1 / np.diag(Omega))                       # first-step weighting: inverse variances
print(f'{q} moments, {K} parameters; standard errors of the data moments:')
print(f'  work status  {np.round(np.sqrt(np.diag(Omega)[:9] / N), 3)}')
print(f'  wealth       {np.round(np.sqrt(np.diag(Omega)[9:] / N), 2)}')
```

The two kinds of moments live on different scales: a residual in a work share is of
order $0.01$, a residual in mean wealth of order $0.1$, and the inverse variances in
`Wdiag` put them on the same footing. With the identity matrix the wealth residuals
would decide everything and the work shares nothing.

:::{div}
:class: discussion

- Consumption is in the data too. Why is it left out of the moments here, and what would
  it add?
- Retirement is absorbing, so the work status of one individual in periods 4 and 5 is
  strongly correlated. Where does this show up in $\Omega$, and does the diagonal $W$
  see it?
:::

(msm-practice-simulator)=
## The simulator with fixed draws

The simulated moments solve the model at $\theta$ and simulate $S$ copies of the data,
initial wealth included, with **the same seed at every $\theta$**, as in the
[criterion of Class 19](19_msm.md#msm-criterion). The simulated sample is independent
of the data, which the asymptotics require.

The parameter walls keep the optimizer where the solver works: $\beta < 1$ and
$\delta > 0$.

```{code-cell} python3
def simmoments(theta, S=5, seed=1):
    '''Simulated moments at theta: S copies of the data, initial wealth included, same draws at every theta'''
    delta, beta = theta
    if not (0 < delta < 5 and 0.5 < beta < 1): return None   # outside: no solution, or no retirement at all
    m = model(theta)
    sol = solve_dcegm(m)                                  # about 5 milliseconds
    rng = np.random.default_rng(seed)                     # fixed draws: the same generator state at every theta
    Ms, Cs, Ds = simulate(m, sol, rng.uniform(2, 20, S * N), seed=rng)
    return moments(Ms, Cs, Ds).mean(0)

def criterion(theta, W, S=5, seed=1):
    '''The MSM criterion g' W g'''
    ms = simmoments(theta, S, seed)
    if ms is None: return 1e10                            # a wall for the optimizer
    g = mhat - ms
    return g @ W @ g

t0 = time.perf_counter()
Q0 = criterion(theta0, Wdiag)
print(f'criterion at the truth {Q0:.4f}, one evaluation with S = 5 in {time.perf_counter() - t0:.3f} seconds')
```

The criterion at the true parameters is not zero: the data moments carry sampling noise
and the simulated moments carry simulation noise, and the criterion measures their
distance. The figure cuts the criterion along $\delta$ and along $\beta$ through the
truth, and zooms in on a window of $\delta$ one thousandth wide.

```{code-cell} python3
:tags: [hide-input]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
dg, bg, dz = np.linspace(0.3, 0.7, 41), np.linspace(0.9, 0.99, 37), np.linspace(0.4995, 0.5005, 51)
for S, col in ((1, 'C0'), (5, 'C1')):
    ax[0].plot(dg, [criterion([d, theta0[1]], Wdiag, S) for d in dg], color=col, lw=1, label=f'S = {S}')
    ax[1].plot(bg, [criterion([theta0[0], b], Wdiag, S) for b in bg], color=col, lw=1)
    ax[2].plot(dz, [criterion([d, theta0[1]], Wdiag, S) for d in dz], '.-', color=col, ms=3, lw=0.7)
ax[0].set(xlabel='δ', yscale='log', title=f'Criterion along δ, β = {theta0[1]}')
ax[1].set(xlabel='β', yscale='log', title=f'Criterion along β, δ = {theta0[0]}')
ax[2].set(xlabel='δ', title='Zoom: the steps of the indicator moments')
ax[0].legend()
for a in ax: a.axvline(theta0[0] if a is not ax[1] else theta0[1], color='k', lw=0.5, ls=':'); a.grid(color='0.9')
plt.show()
```

- Both cuts have a clear minimum at the truth, three orders of magnitude deep: $\delta$
  and $\beta$ are identified by these moments.
- The zoom shows **steps**: the work status is an indicator $\mathbf 1(\tilde u < P)$, so
  a simulated share only changes when a fixed draw crosses the moving threshold, by
  $1/(SN)$ at a time. The steps are smaller with $S = 5$ than with $S = 1$, and they
  never disappear.
- The steps are why `estimate` uses Nelder–Mead and why `jacobian` takes finite
  differences with a step $h = 10^{-2}$, wide enough to average over many of them: with
  $h = 10^{-3}$ the derivative of a work share is one step divided by $h$, a number
  that has nothing to do with the slope.

:::{div}
:class: discussion

- Where exactly along a wealth path does one draw crossing the threshold change the
  simulated moments, and does the effect stay in one period?
- The wealth moments are continuous in $\theta$ given the work status. Why do they still
  jump at the steps?
:::

(msm-practice-estimation)=
## Estimation and standard errors

The first step minimizes the criterion with the diagonal weighting matrix from a
deliberately poor start, and then **restarts Nelder–Mead from its own solution**: on a
criterion with steps the simplex can collapse early, and a restart from the solution
with a fresh simplex costs a second or two and catches it.

```{code-cell} python3
t0 = time.perf_counter()
th1, Q1, nfev = estimate(criterion, x0=(0.3, 0.9), args=(Wdiag,))      # first step, from a poor start
th1, Q1, nfev2 = estimate(criterion, x0=th1, args=(Wdiag,))           # restart from the solution
se1 = np.sqrt(np.diag(vcov(jacobian(simmoments, th1, S=5), Wdiag, Omega, N, S=5)))
print(f'first step   delta {th1[0]:.4f} ({se1[0]:.4f})  beta {th1[1]:.4f} ({se1[1]:.4f})  '
      f'{nfev + nfev2} evaluations in {time.perf_counter() - t0:.1f} s')
```

The first step lands within a tenth of a standard error of the truth in both parameters.
On other data sets drawn from the same model it can land two standard errors away, and
that is not a bug either: eighteen moments from an absorbing retirement process and a
persistent wealth path are strongly correlated, and a diagonal $W$ treats them as
independent, so the estimate follows whichever noise the sample has in common across
periods.

The second step uses the optimal weighting matrix $\hat\Omega^{-1}$, starts from the
first-step estimate, and delivers the $J$-test with $q - K = 16$ degrees of freedom.

```{code-cell} python3
t0 = time.perf_counter()
Wopt = np.linalg.inv(Omega)
th2, Q2, nfev = estimate(criterion, x0=th1, args=(Wopt,))
th2, Q2, nfev2 = estimate(criterion, x0=th2, args=(Wopt,))
se2 = np.sqrt(np.diag(vcov(jacobian(simmoments, th2, S=5), Wopt, Omega, N, S=5)))
J, p = jtest(Q2, N, S=5, df=q - K)
print(f'second step  delta {th2[0]:.4f} ({se2[0]:.4f})  beta {th2[1]:.4f} ({se2[1]:.4f})  '
      f'{nfev + nfev2} evaluations in {time.perf_counter() - t0:.1f} s')
print(f'J-test {J:.1f} on {q - K} degrees of freedom, p-value {p:.3f}')
```

The second step lands within half a standard error of $\delta = 0.5$ and $\beta = 0.95$,
with standard errors three times smaller than the first step for $\delta$ and five times
smaller for $\beta$. That is what the optimal $W$ buys when the moments are correlated:
it weights the *differences* between neighboring periods, which the diagonal $W$
cannot see.

The $J$-test does not reject the model that generated the data: $J = 16.9$ on 16
degrees of freedom, $p \approx 0.39$. It does not always come out this way. On nine data
sets drawn from the same model with different seeds the p-values ranged from $0.02$ to
$0.99$, with one rejection at the 5% level, and one rejection in twenty is what a 5% test
does. {cite:t}`altonji1996SmallSampleBias` is also in play: $\hat\Omega$ has a condition
number of $2 \cdot 10^4$ and is estimated from one thousand individuals, and tests based
on its inverse over-reject in samples of this size.

The table every MSM paper carries is the fit of the moments, here as a figure: data
moments with two standard error bands against the simulated moments at the estimate.

```{code-cell} python3
:tags: [hide-input]

ms = simmoments(th2)
se_m = np.sqrt(np.diag(Omega) / N)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, sl, off, lab in ((ax[0], slice(0, 9), 0, 'share working'), (ax[1], slice(9, 18), 1, 'mean wealth')):
    t = np.arange(9) + off
    a.errorbar(t, mhat[sl], yerr=2 * se_m[sl], fmt='o', ms=4, capsize=3, label='data, ± 2 s.e.')
    a.plot(t, ms[sl], 's--', ms=4, lw=0.8, color='C3', label='model at the estimate')
    a.set(xlabel='period', title=f'Fit of the moments: {lab}'); a.grid(color='0.9'); a.legend()
plt.show()
```

Both profiles are matched within the bands in every period, which is what the $J$-test
summarizes. When the test does reject, the figure usually shows why: the optimal $W$
does not close the gaps one by one, it minimizes a quadratic form in which a run of
positively correlated gaps on the same side of the data is cheap, and such a run is
what the $J$-test picks up.

:::{div}
:class: discussion

- The second-step standard errors are much smaller than the first-step ones. Which part
  of the sandwich changed, and why does the gain depend on the correlation of the moments?
- Eighteen moments and one thousand individuals: how many numbers does $\hat\Omega$
  estimate, and would you trust the $J$-test more with nine moments?
:::

(msm-practice-montecarlo)=
## Monte Carlo check of the standard errors

The standard errors come from a formula with a numerical Jacobian computed over steps.
The honest check is a **Monte Carlo**: draw ten data sets from the true model with
different seeds, initial wealth included, run the first-step estimator on each, and
compare the spread of the estimates to the formula.

Every replication keeps the same simulation seed in `simmoments`, so only the data
change, and the first step is enough for the check because its formula and the
second step's share every ingredient but $W$.

```{code-cell} python3
t0 = time.perf_counter()
sol0, ests = solve_dcegm(m0), []
for r in range(10):
    rng_r = np.random.default_rng(100 + r)
    Mr, Cr, Dr = simulate(m0, sol0, rng_r.uniform(2, 20, N), seed=rng_r)   # a new data set from the true model
    hr = moments(Mr, Cr, Dr)
    mh_r, W_r = hr.mean(0), np.diag(1 / np.var(hr, axis=0))
    def crit_r(theta, W):
        '''The criterion with this replication's data moments; the simulation seed stays the same'''
        ms = simmoments(theta)
        return 1e10 if ms is None else (mh_r - ms) @ W @ (mh_r - ms)
    th, _, _ = estimate(crit_r, x0=(0.3, 0.9), args=(W_r,))
    th, _, _ = estimate(crit_r, x0=th, args=(W_r,))
    ests.append(th)
ests = np.array(ests)
print(f'{ests.shape[0]} replications in {time.perf_counter() - t0:.0f} s')
print(f'mean of the estimates      delta {ests[:, 0].mean():.4f}  beta {ests[:, 1].mean():.4f}')
print(f'std of the estimates       delta {ests[:, 0].std():.4f}  beta {ests[:, 1].std():.4f}')
print(f'formula, first step above  delta {se1[0]:.4f}  beta {se1[1]:.4f}')
```

The estimator is centered on the truth, and the Monte Carlo spread matches the formula
within 10% for both parameters, with a Jacobian that averages over the steps.
Ten replications is a rough check, not a verdict: the standard error of a standard
deviation from ten draws is about a quarter of its value, so agreement within 30% is
all the check can promise.

:::{div}
:class: discussion

- The Monte Carlo keeps the simulation seed fixed across replications. Which source of
  noise in the $(1 + 1/S)$ formula does it then not measure?
- How many replications would it take to tell a formula standard error of $0.003$ from a
  true one of $0.004$?
:::

(msm-practice-sigma)=
## What breaks: the taste shock scale

The scale $\sigma$ of the taste shocks was fixed at its true value. Adding it to the
parameter vector is one line in `model` and one in the walls, and it is where the
method shows its limits. The criterion along $\sigma$ through the truth, computed with a
version of `model` that takes three parameters:

```{code-cell} python3
def model3(theta):
    '''The retirement model with the taste shock scale as a third parameter'''
    delta, beta, sigma = theta
    return retirement(beta=beta, delta=delta, sigma=sigma, sigma_y=0.2, nquad=10)

def criterion3(theta, W, S=5, seed=1):
    if not (0 < theta[0] < 5 and 0.5 < theta[1] < 1 and theta[2] > 0): return 1e10
    m = model3(theta)
    rng = np.random.default_rng(seed)
    Ms, Cs, Ds = simulate(m, solve_dcegm(m), rng.uniform(2, 20, S * N), seed=rng)
    g = mhat - moments(Ms, Cs, Ds).mean(0)
    return g @ W @ g

for s in (0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2):
    print(f'sigma {s:<5} criterion {criterion3([theta0[0], theta0[1], s], Wdiag):.4f}')
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
for instance. The practical task asks for it.

<!-- ````{warning} Practical task: MSM estimation of the retirement model

(task-msm-practice)=
Pull the class code and open the pre-code notebook `msm_practice_pre.ipynb` in
`session20-nov3/`:

```bash
cd sb-dse-code && git pull
# first time: git clone https://github.com/fediskhakov/sb-dse-code.git
```

1. Fill in `moments`, `simmoments` and `criterion` and reproduce the two-step estimates,
   the standard errors and the $J$-test of this page.
2. Run the first step with $S = 1$, $5$ and $20$. Report the estimates, the standard
   errors and the run time, and compare the change in the standard errors with the
   factor $\sqrt{1 + 1/S}$.
3. Run the first step with the identity weighting matrix instead of the inverse
   variances. Which moments drive the estimate now, and what happens to $\beta$?
4. Add $\sigma$ as a third parameter with `model3`. Start Nelder–Mead from
   $(0.3, 0.9, 0.2)$ and from $(0.7, 0.98, 0.05)$, with a restart each, and compare the
   two solutions and their standard errors with the Jacobian step $h = 10^{-3}$ and
   $h = 10^{-2}$.
5. Add moments that respond to $\sigma$: the mean and the standard deviation of the
   first period of retirement within each quartile of initial wealth, eight numbers,
   written as contributions whose mean over all individuals is the quartile statistic.
   Does the criterion along $\sigma$ become steeper, and does the estimate of $\sigma$
   settle?
6. [optional] Replace the simulated work shares by the choice probabilities that
   `simulate(..., prob=True)` returns, keeping the data moments as they are. Do the steps
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
- 📖 {cite:t}`mcfaddenMethodSimulatedMoments1989` "A Method of Simulated Moments for Estimation of Discrete Response Models Without Numerical Integration" — simulated indicators and why their steps do not hurt consistency
- 📖 {cite:t}`french2005EffectsHealthWealth` "The Effects of Health, Wealth, and Wages on Labour Supply and Retirement Behaviour" — MSM estimation of a retirement model on profiles of participation, hours and assets by age
- 📖 {cite:t}`gourinchas2002ConsumptionLifeCycle` "Consumption over the Life Cycle" — MSM on the age profile of consumption
- 📖 {cite:t}`altonji1996SmallSampleBias` "Small-Sample Bias in GMM Estimation of Covariance Structures" — why the optimal weighting matrix can hurt
- 📖 {cite:t}`corbae2024MethodMoments` "Method of Moments Estimators" — the problem set of the DSE 2024 lectures is the template of today's exercise
- 📄 `scipy.optimize.minimize`, method Nelder–Mead <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html>
````
