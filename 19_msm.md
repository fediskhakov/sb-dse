---
title: 📖 Method of simulated moments
short_title: 📖 MSM
subtitle: Class 19 — Thursday, October 29
downloads:
  - file: 19_msm.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

<span class="tag wip">🚧 🚧 🚧 Not the final version 🚧 🚧 🚧</span>

[EGM](17_egm.md) solves the consumption-savings model in milliseconds, but there is no
likelihood for consumption and wealth data. The **method of simulated moments (MSM)**
needs only the simulator: pick a few statistics of the data, and find the parameters
at which the model reproduces them. Theory today, [practice on Tuesday](20_msm_practice.md).

````{seealso} Key reading

{cite:t}`mcfaddenMethodSimulatedMoments1989` "A Method of Simulated Moments for
Estimation of Discrete Response Models Without Numerical Integration", *Econometrica*
57(5), 995–1026. The estimator, the role of fixed simulation draws, and the
$(1 + 1/S)$ variance correction of this class.

{cite:t}`pakes1989SimulationAsymptoticsOptimizationa` "Simulation and the Asymptotics
of Optimization Estimators", *Econometrica* 57(5), 1027–1057. The asymptotic theory
for criteria that are not smooth in the parameters, the case of every simulator with a
discrete choice in it.
````

````{hint} Running the code for this lecture
:class: dropdown

The code for this class is in the folder `session19-oct29/` of the course **code
repository**. The model object of Class 16 and the EGM solver of Class 17 are collected
in the module `deaton_egm.py` in that folder, and the notebook `msm.ipynb` has all the
code of this page.

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

## From solver and simulator to estimator

A solver and a simulator that are debugged let us study a theory: run the model for a
range of parameter values and see what the decision maker does. The **ultimate goal is
to match the model to the observed data**, in order to

- quantify the effects of the parts of the theoretical setup
- run **counterfactual experiments**: simulate behavior under a policy regime that has
  never been observed
- support or falsify the theory by its fit to the data

The workflow of a structural project is the arc of this course: model, implementation,
solver, *understanding how the model works*, estimation, validation, counterfactuals.
Step four is where the time goes: solve, simulate, ask whether the result makes
economic sense, repeat. It is also where the moments for step five are found.

### Estimation versus calibration

The terms are often used interchangeably, and calibration exercises also search for the
parameters that fit the data best. What estimation adds:

- **standard errors**, a measure of the variability of the estimates
- **a study of identification**, to make sure that no other parameter values explain
  the data equally well

Estimation often concerns only a subset of the parameters, with the rest fixed at values
from the literature, like a calibration. The [NFXP estimation of Class 10](10_nfxp.md)
fixed $\beta$, for instance.

### Why not maximum likelihood?

The [MLE of Class 10](10_nfxp.md#mle-asymptotics) is the most efficient estimator there
is, when the model gives the probability of the observed data. In the bus engine model it
does: the logit choice probabilities. In the consumption-savings model it does not, or
not easily:

- consumption is a **deterministic function of the state**, $c = c(M)$. Observed
  consumption that is off the policy function has likelihood zero, so a likelihood needs
  measurement error in consumption and an assumption about its distribution
- wealth is a sum of past shocks, so the likelihood of a wealth path is an integral over
  every unobserved income draw, and in a panel of $T$ periods that is a $T$-dimensional
  integral per individual
- the data are often **aggregated**: wealth by age from one survey, consumption from
  another, and no individual with both

Simulation is cheap where integration is not. The simulator of
[Class 16](16_cons_practice.md#simulation-and-keanes-test) produces a synthetic panel
for any $\theta$, and any statistic of the synthetic panel can be compared to the same
statistic of the data.

:::{div}
:class: discussion

- Which objects would a likelihood for consumption and wealth data have to integrate out,
  and which of them does the simulator draw for free?
- A calibration matches three moments with three parameters exactly. What question can it
  not answer?
:::

(msm-definition)=
## Method of moments, GMM and simulated moments

Matching statistics of the model to statistics of the data has a long history:

- **method of moments**: as many moments as parameters, a system of equations, solved
  exactly
- **generalized method of moments (GMM)** of {cite:t}`hansen1982LargeSampleProperties`:
  more moments than parameters, minimize a weighted distance between the data moments and
  the theoretical moments
- **method of simulated moments (MSM)** of {cite:t}`mcfaddenMethodSimulatedMoments1989`:
  GMM with the theoretical moments replaced by averages over simulated data

Let $z_i$, $i = 1, \dots, N$, be the observed data, one record per individual, and let
$h(z)$ be a vector of $q$ statistics of one record, the **moment functions**. Three
objects share the name "moments":

$$
\underbrace{\hat m_N = \frac{1}{N} \sum_{i=1}^N h(z_i)}_{\text{data moments}},
\qquad
\underbrace{m(\theta) = \mathbb{E}_\theta\, h(z)}_{\text{model moments}},
\qquad
\underbrace{\tilde m_S(\theta) = \frac{1}{SN} \sum_{s=1}^{SN} h\big(\tilde z_s(\theta)\big)}_{\text{simulated moments}}
$$

The model moments are the expectation of $h$ under the model with parameters $\theta$,
the object GMM works with. MSM replaces them by the average over a synthetic sample
$\tilde z_s(\theta)$ of $S N$ records, simulated from the model at $\theta$ with random
draws $\tilde u_s$. The vector

$$
g_N(\theta) = \hat m_N - \tilde m_S(\theta)
$$

of $q$ **moment conditions** is zero in expectation at the true $\theta_0$ and at no
other point, if the moments identify the parameters.

````{attention} Definition

The **method of simulated moments estimator** with weighting matrix $W$ is

$$
\hat\theta_{MSM}(W) = \arg\min_{\theta \in \Theta} \; Q_N(\theta),
\qquad
Q_N(\theta) = g_N(\theta)^T\, W\, g_N(\theta),
$$

where $g_N(\theta) = \hat m_N - \tilde m_S(\theta)$ is the $q \times 1$ vector of moment
conditions, $W$ is a $q \times q$ symmetric positive definite weighting matrix, $\theta$
is a $K \times 1$ parameter vector with $K \leqslant q$, and the simulated moments
$\tilde m_S(\theta)$ are computed from the **same random draws** $\tilde u_s$,
$s = 1, \dots, SN$, at every $\theta$.
````

The last clause is the whole trick. With the draws fixed, every evaluation of
$Q_N(\theta)$ solves and simulates the model on the same shocks, so the simulated
moments are a deterministic function of $\theta$ that an optimizer can minimize.

Redraw the shocks at every $\theta$ and the criterion is a random function: an
optimizer cannot tell a change in the parameters from a change in the draws.

Three conditions make the estimator work, in the words of
{cite:t}`corbae2024MethodMoments`:

- the simulation draws are **independent of the data**
- the model is **identified**: $m(\theta) = m(\theta_0)$ only at $\theta_0$
- the simulated moments are **continuous in the mean**: $\tilde m_S(\theta) \to m(\theta)$
  as the simulated sample grows

The count of moments against parameters is the **order condition**, necessary but not
sufficient:

- $q > K$: **overidentified**, the usual case, with a test of the model for free
- $q = K$: **just identified**, the criterion is zero at the optimum and the weighting
  matrix does not matter
- $q < K$: **underidentified**, no estimator

````{tip} Example: moments for the consumption-savings model

The data are a panel of wealth $M_{it}$ and consumption $c_{it}$ for $N$ individuals
over $T$ periods, from the [Deaton model of Class 16](16_cons_practice.md) with
$\beta = 0.92$, $\sigma = 0.5$, $R = 1.05$. Three moment functions, one record per
individual:

- $h_1$: mean wealth over the periods, $\frac{1}{T}\sum_t M_{it}$
- $h_2$: standard deviation of wealth over the periods, within individual
- $h_3$: standard deviation of consumption growth $\Delta \log c_{it}$

A more patient consumer holds more wealth, so $h_1$ moves with $\beta$. A riskier
income moves wealth and consumption around more, so $h_2$ and $h_3$ move with $\sigma$
— and $h_1$ moves with $\sigma$ too, through precautionary savings. With $q = 3$ moments
and $K = 2$ parameters the model is **overidentified**, and the one extra moment is a
test of the model.
````

:::{div}
:class: discussion

- The mean of a statistic over individuals is a moment. Is the median? Is a regression
  coefficient?
- Why must the number of moments be at least the number of parameters?
:::

(msm-criterion)=
## The simulated criterion

The data are simulated too, from the "true" parameters, so that we know the answer. The
module `deaton_egm.py` holds the [model object of Class 16](16_cons_practice.md#cons-model-object),
the [EGM solver of Class 17](17_egm.md), which returns the policy as points on the
endogenous grid, and the simulator.

```{code-cell} python3
:tags: [hide-input]

import sys, time, pathlib, tempfile, urllib.request
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2
CODE = pathlib.Path('../gh-code/session19-oct29')         # the model and EGM solver of Classes 16-17
if not CODE.exists():                                     # otherwise fetch it from the code repository
    CODE = pathlib.Path(tempfile.mkdtemp())
    urllib.request.urlretrieve('https://raw.githubusercontent.com/fediskhakov/sb-dse-code/main/session19-oct29/deaton_egm.py',
                               CODE / 'deaton_egm.py')
sys.path.insert(0, str(CODE))
from deaton_egm import deaton, solve_egm, simulate
```

```{code-cell} python3
theta0 = np.array([0.92, 0.5])                            # true beta and sigma, to be recovered
N, T, tail = 500, 40, 10                                  # individuals, simulated periods, observed periods
m0 = deaton(beta=theta0[0], sigma=theta0[1])              # R = 1.05, 100 grid points on [0, 20]
rng = np.random.default_rng(2026)
M0 = np.exp(rng.standard_normal(N))                       # initial wealth, log-normal
Wd, Cd = simulate(m0, solve_egm(m0), M0, T, seed=rng)     # the "data": N wealth and consumption paths

def moments(W, C):
    '''Moment functions h(z), one row per individual, from the last `tail` periods of the paths'''
    Wt, Ct = W[:, -tail:], C[:, -tail:]                   # the first 30 periods are burn-in, not observed
    return np.column_stack([Wt.mean(1), Wt.std(1), np.diff(np.log(Ct), axis=1).std(1)])

h = moments(Wd, Cd)                                       # N x q individual contributions
mhat = h.mean(0)                                          # data moments
Omega = np.cov(h.T)                                       # covariance of the contributions, q x q
for name, mh, s in zip(('mean wealth', 'std wealth', 'std consumption growth'), mhat, np.sqrt(np.diag(Omega) / N)):
    print(f'{name:<25} {mh:7.4f}  (s.e. {s:.4f})')
```

Only the last ten of the forty simulated periods are "observed": the first thirty wash
out the initial wealth, so the data come from the ergodic distribution of the model.
The simulation below runs the same forty periods.

The covariance matrix $\Omega$ of the individual contributions $h(z_i)$ is the one
object the asymptotics need from the data, and it comes for free once the moments are
means over individuals.

The simulated moments solve the model at $\theta$ and simulate $S$ copies of the data
with a **fixed seed**, so the same draws are used at every $\theta$:

```{code-cell} python3
def simmoments(theta, S=5, seed=1):
    '''Simulated moments at theta: S copies of the data, same draws at every theta'''
    beta, sigma = theta
    if not (0 < beta < 1 / 1.05 and sigma > 0): return None   # EGM does not converge for beta R >= 1
    m = deaton(beta=beta, sigma=sigma)
    pol = solve_egm(m)                                    # about 3 milliseconds
    Ws, Cs = simulate(m, pol, np.tile(M0, S), T, seed=np.random.default_rng(seed))   # fixed draws
    return moments(Ws, Cs).mean(0)

def criterion(theta, W, S=5, seed=1):
    '''The MSM criterion g' W g'''
    ms = simmoments(theta, S, seed)
    if ms is None: return 1e10                            # a wall for the optimizer
    g = mhat - ms
    return g @ W @ g
```

The figure plots the criterion in $\beta$, with $\sigma$ at its true value and a
diagonal weighting matrix, for three simulators: $S = 1$ with fixed draws, $S = 10$ with
fixed draws, and $S = 1$ with fresh draws at every $\beta$.

```{code-cell} python3
:tags: [hide-input]

Wdiag = np.diag(1 / np.diag(Omega))                       # moments scaled by their variances
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, (lo, hi) in zip(ax, ((0.85, 0.95), (0.912, 0.928))):
    bg = np.linspace(lo, hi, 81)
    a.plot(bg, [criterion([b, theta0[1]], Wdiag, S=1) for b in bg], lw=1, label='S = 1, fixed draws')
    a.plot(bg, [criterion([b, theta0[1]], Wdiag, S=10) for b in bg], lw=1, label='S = 10, fixed draws')
    a.plot(bg, [criterion([b, theta0[1]], Wdiag, S=1, seed=k) for k, b in enumerate(bg)], '.', ms=3, label='S = 1, fresh draws')
    a.axvline(theta0[0], color='k', lw=0.5, ls=':')
    a.set(xlabel='β', title=f'MSM criterion in β on [{lo}, {hi}], σ = {theta0[1]}'); a.grid(color='0.9')
ax[0].set(yscale='log'); ax[0].legend()
plt.show()
```

Three things to see:

- with fixed draws the criterion is a **smooth** function of $\beta$ and has one
  minimum, close to the true value
- with fresh draws it is a **cloud**: in the zoom the simulation noise is as large as the
  differences the optimizer needs to see, and the minimum is wherever the draws happened
  to fall
- a larger $S$ lowers the noise of the simulated moments, so the fixed-draw criterion with
  $S = 10$ is centered better than with $S = 1$ — but it is the *fixing*, not the number
  of draws, that makes the function minimizable

Where it breaks: a **discrete choice in the simulator**. The retirement decision of
[Class 18](18_dcegm.md#dcegm-simulation) is an indicator $\mathbf 1(\tilde u < P)$, so a
simulated retirement rate is a step function of $\theta$ even with fixed draws: it only
changes when a draw crosses the threshold.

This is the problem of {cite:t}`mcfaddenMethodSimulatedMoments1989`, and the reason
the asymptotics of {cite:t}`pakes1989SimulationAsymptoticsOptimizationa` do not
assume a smooth criterion. Gradient-based optimizers fail on the steps; derivative-free
ones, Nelder–Mead in our case, walk over them as long as the steps are small compared
to the simplex.

:::{div}
:class: discussion

- The criterion with fresh draws is a random variable at every $\beta$. What does its
  expectation look like, and why is the minimum of the expectation not the expectation of
  the minimum?
- The model is solved on a grid and the policy is interpolated. Is the simulated moment
  a smooth function of $\beta$ even with fixed draws?
:::

(msm-asymptotics)=
## Asymptotic properties

The MSM estimator is a GMM estimator with a noisy version of the model moments, and it
inherits the GMM theory of {cite:t}`hansen1982LargeSampleProperties` with one
correction. Let $D$ be the $q \times K$ **Jacobian of the model moments**,

$$
D = \frac{\partial m(\theta_0)}{\partial \theta^T},
$$

and $\Omega$ the $q \times q$ asymptotic covariance of the data moments,
$\sqrt N (\hat m_N - m(\theta_0)) \xrightarrow{d} N(0, \Omega)$.

````{attention} Theorem (asymptotics of the MSM estimator)

Under regularity conditions, with the simulation draws fixed across $\theta$ and
independent of the data, and with $S$ fixed as $N \to \infty$:

1. **Consistency**: $\hat\theta_{MSM}(W) \xrightarrow{p} \theta_0$ for any positive
   definite $W$, provided $\theta_0$ is the unique zero of
   $m(\theta_0) - m(\theta)$ on $\Theta$ (**identification**).
2. **Asymptotic normality**:

   $$
   \sqrt N\, \big( \hat\theta_{MSM}(W) - \theta_0 \big) \xrightarrow{d}
   N\Big( 0,\; \big(1 + \tfrac{1}{S}\big)\, (D^T W D)^{-1} D^T W \,\Omega\, W D\, (D^T W D)^{-1} \Big),
   $$

   provided $D$ has full column rank $K$ (**local identification**).
3. **Efficiency**: the asymptotic variance is smallest for $W^\star = \Omega^{-1}$, where it
   reduces to $(1 + \frac{1}{S})\, (D^T \Omega^{-1} D)^{-1}$.
````

Where does the factor $1 + 1/S$ come from? The moment conditions at $\theta_0$ carry two
independent sources of noise,

$$
g_N(\theta_0) = \underbrace{\hat m_N - m(\theta_0)}_{\text{data}} \;-\;
\underbrace{\big(\tilde m_S(\theta_0) - m(\theta_0)\big)}_{\text{simulation}},
$$

the data moments with variance $\Omega / N$ and the simulated moments with variance
$\Omega / (SN)$, because they average $SN$ records from the same distribution. Their
sum has variance $(1 + 1/S)\, \Omega / N$, and this is the only place where the
simulation enters the theory.

What the factor costs: with $S = 1$ the standard errors are $\sqrt 2$ times the GMM
ones, with $S = 10$ they are 5% larger, and simulating more than ten copies of the data
buys very little.

Three more things to read off the theorem:

- **Consistency needs no good $W$** and no large $S$. The estimator is consistent with the
  identity matrix and $S = 1$. What $W$ and $S$ buy is precision.
- **The sandwich** $(D^T W D)^{-1} D^T W \Omega W D (D^T W D)^{-1}$ is the
  [robust form of Class 10](10_nfxp.md#bhhh), with the Jacobian of the moments in place
  of the Hessian of the likelihood. It collapses to $(D^T \Omega^{-1} D)^{-1}$ only at the
  optimal weighting matrix.
- **Local identification** is a rank condition: $D$ has to have $K$ linearly independent
  columns, so every parameter must move at least one moment, and no two parameters may
  move the moments in the same direction. A flat direction of the criterion is a column
  of $D$ that is zero or a combination of the others.
- **Precision is governed by $D$.** A moment that barely responds to a parameter is a
  small entry of $D$ and a large entry of $(D^T W D)^{-1}$: this is how local
  identification is linked to the standard errors, and a weakly identified parameter
  announces itself through them.

Where it breaks: every formula above needs $D$, the derivative of a function we only
know through a simulator. The theorem does not require a smooth criterion,
{cite:t}`pakes1989SimulationAsymptoticsOptimizationa` cover step functions, but the
standard errors do need the slope.

In practice $D$ is computed by finite differences of $\tilde m_S(\theta)$ with fixed
draws, and the step size must be small enough to be a derivative but large enough to
jump over the steps and the interpolation kinks of the solver. Tuesday's class shows
how much rides on that choice.

:::{div}
:class: discussion

- The MLE of [Class 10](10_nfxp.md#mle-asymptotics) attains the Cramér–Rao bound. Why is
  the MSM estimator never as efficient, even with $S \to \infty$ and the optimal $W$?
- A parameter that moves two moments in exactly opposite directions to another parameter.
  Is it identified?
:::

(msm-weighting)=
## The weighting matrix and the two-step estimator

The optimal weighting matrix is the inverse of the covariance of the moment conditions,
which is unknown but estimable. For moments that are means over individuals,
$\hat\Omega$ is the sample covariance of the contributions $h(z_i)$, the one line
`np.cov(h.T)` above, computed from the data once.

For moments of other kinds, autocovariances of a time series for instance, a
Newey–West estimator or a bootstrap replaces it, and when the data moments come from
published tables rather than micro data, $\hat\Omega$ is itself simulated from the
model at a first-step estimate.

````{attention} Algorithm (two-step MSM)

**Input:** data moments $\hat m_N$ and their covariance $\hat\Omega$, a simulator of
$\tilde m_S(\theta)$ with fixed draws, a starting point $\theta^{(0)}$.

**Steps:**

1. **First step.** Minimize $Q_N(\theta)$ with a diagonal $W_1$, the identity or the inverse
   variances $\text{diag}(\hat\Omega)^{-1}$, to get the consistent $\hat\theta_1$.
2. **Weighting matrix.** $W_2 = \hat\Omega^{-1}$; or, when the moments are not means over
   individuals, the covariance of the moment conditions simulated at $\hat\theta_1$.
3. **Second step.** Minimize $Q_N(\theta)$ with $W_2$, from $\hat\theta_1$, to get
   $\hat\theta_2$.
4. **Standard errors.** Compute $\hat D$ by finite differences of $\tilde m_S(\theta)$ at
   $\hat\theta_2$ and the asymptotic variance of the theorem with $W_2$ and $\hat\Omega$.

**Output:** $\hat\theta_2$, its standard errors, and the criterion value $Q_N(\hat\theta_2)$
for the test of overidentifying restrictions.
````

Iterating steps 2 and 3 until $\theta$ stops changing is the **iterated MSM**; the
continuously updated version makes $W$ a function of $\theta$ inside the criterion. In
practice the weighting matrix is often simplified, and {cite:t}`altonji1996SmallSampleBias`
is the reason: in small samples the optimal $W$ is estimated with error that is
*correlated* with the moment conditions, and the second step can be more biased than
the first. The choices seen in applied work, from the simplest up:

- the **identity**, in the first step of a multi-step estimation
- a **diagonal** matrix of inverse variances, which brings the moments to a common scale
  and ignores their covariances
- **manually chosen weights**, to bring all moments to the same scale or to emphasize the
  moments the study is about
- the **optimal** $\hat\Omega^{-1}$, from the data or from a first-step simulation
- an **iteratively updated** $W$, a **Newey–West** estimate for serially correlated
  moments, and model-specific adjustments

Many ways to skin a cat, and the choice should be reported with the estimates.

(msm-jtest)=
## The test of overidentifying restrictions

With more moments than parameters the criterion is not zero at the optimum, and its
size says whether the extra moments are compatible with the model.

````{attention} Definition

The **test of overidentifying restrictions** (the $J$-test, {cite:t}`hansen1982LargeSampleProperties`)
rejects the model at level $\alpha$ if

$$
J_N = \frac{N}{1 + 1/S}\; Q_N\big(\hat\theta_{MSM}(\hat\Omega^{-1})\big) > \chi^2_{q - K, 1 - \alpha},
$$

where $Q_N$ is evaluated at the estimator with the optimal weighting matrix, and the
degrees of freedom are the number of moments in excess of the number of parameters.
````

The test is valid only with the optimal weighting matrix: with any other $W$ the
criterion at the optimum has a different distribution. And it tests the moments chosen,
not the model: a model that fits three moments can be rejected by a fourth.

```{code-cell} python3
def estimate(W, S=5, x0=(0.9, 0.4), seed=1):
    '''Minimize the MSM criterion by Nelder-Mead: derivative free, so steps in the simulator do not matter'''
    res = minimize(criterion, x0, args=(W, S, seed), method='Nelder-Mead',
                   options={'xatol': 1e-5, 'fatol': 1e-9})
    assert res.success, res.message
    return res.x, res.fun, res.nfev

def jacobian(theta, S=5, h=1e-4):
    '''Jacobian D of the model moments by central differences of the simulated moments, q x K'''
    theta, D = np.asarray(theta, float), []
    for e in np.eye(theta.size) * h:
        D.append((simmoments(theta + e, S) - simmoments(theta - e, S)) / (2 * h))   # same draws on both sides
    return np.array(D).T

def vcov(theta, W, S=5):
    '''Asymptotic variance of the MSM estimator, the sandwich with the (1 + 1/S) correction'''
    D = jacobian(theta, S)
    assert np.linalg.matrix_rank(D) == theta.size          # local identification: full column rank
    bread = np.linalg.inv(D.T @ W @ D)
    return (1 + 1 / S) * bread @ D.T @ W @ Omega @ W @ D @ bread / N
```

```{code-cell} python3
S = 5
for name, W in (('identity', np.eye(3)), ('diagonal', Wdiag), ('optimal', np.linalg.inv(Omega))):
    t0 = time.perf_counter()
    th, Q, nfev = estimate(W, S)
    se = np.sqrt(np.diag(vcov(th, W, S)))
    line = f'{name:<9} beta {th[0]:.4f} ({se[0]:.4f})  sigma {th[1]:.4f} ({se[1]:.4f})  {nfev} evaluations, {time.perf_counter() - t0:.2f} s'
    if name == 'optimal':                                 # the J-test needs the optimal weighting matrix
        J = N / (1 + 1 / S) * Q
        line += f'  J = {J:.2f}, p-value {chi2.sf(J, 3 - 2):.3f}'
    print(line)
```

All three weighting matrices recover $\beta = 0.92$ and $\sigma = 0.5$ within one
standard error, in under a second each, and the standard errors shrink as the weighting
improves. The $J$-test does not reject: the third moment is compatible with a model that
fits the first two, as it should be when the data come from the model itself.

The formula is not the only way to a standard error. A **bootstrap** resamples the
individuals, re-runs the whole two-step estimation with a fresh simulation seed, and
reports the spread of the estimates across replications.

It costs hundreds of estimations, minutes here and days in a large model, but it needs
no Jacobian and no asymptotic argument. The
[Monte Carlo of the next class](20_msm_practice.md#msm-practice-montecarlo) is the
same idea with the true parameters known.

:::{div}
:class: discussion

- The standard errors with the identity matrix are larger than with the optimal one, yet
  the point estimates are nearly the same. When would they differ a lot?
- The $J$-test uses $N / (1 + 1/S)$. What happens to it if $S$ is forgotten in the
  normalization?
:::

(msm-moments)=
## Choosing the moments

The choice of moments is the crucial part of an MSM estimation, and more art than
science. The principles:

- **the moments must be sensitive to the parameters**: understanding how the model works
  means understanding what variation the parameters induce in the simulated data, and the
  $q$ moments should represent that variation
- **every parameter needs its own moment**, in the sense of the rank condition: the
  columns of $D$ are the responses of the moments to the parameters, and they must be
  linearly independent
- **moments must be measurable precisely in the data**: a moment with a large variance in
  $\hat\Omega$ gets a small weight and contributes little
- **moments should be interpretable**, so that the fit can be shown in a table of data
  moments against simulated moments, which every MSM paper carries

The figure maps the criterion over $(\beta, \sigma)$ with the diagonal weighting matrix,
with all three moments on the left and with the two wealth moments only on the right.

```{code-cell} python3
:tags: [hide-input]

bg, sg = np.linspace(0.88, 0.95, 36), np.linspace(0.3, 0.7, 33)
B, Sg = np.meshgrid(bg, sg)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, keep, title in zip(ax, (slice(0, 3), slice(0, 2)), ('three moments', 'two wealth moments only')):
    Wk = np.zeros((3, 3)); Wk[keep, keep] = Wdiag[keep, keep]   # drop a moment by zeroing its weight
    Q = np.array([[criterion([b, s], Wk, S=5) for b in bg] for s in sg])
    cs = a.contourf(B, Sg, np.log10(Q), levels=20, cmap='viridis')
    a.plot(*theta0, 'r+', ms=12, mew=2)
    a.set(xlabel='β', ylabel='σ', title=f'log10 of the MSM criterion, {title}')
    fig.colorbar(cs, ax=a)
plt.show()
```

With three moments the criterion has one basin around the truth. With the two wealth
moments only, the basin stretches into a **valley**: a lower $\beta$ with a higher
$\sigma$ produces nearly the same mean and spread of wealth, impatience traded against
precautionary saving.

The rank condition still holds along the valley, $D$ has rank two, but its columns are
nearly collinear: the standard errors explode and an optimizer stops anywhere along the
floor. The consumption moment breaks the trade-off because $\sigma$ moves consumption
growth and $\beta$ does not.

Where it breaks, in applications:

- a parameter that moves no chosen moment: a flat criterion in that direction, an
  optimizer that stops at the starting value, and standard errors from a singular matrix
- two parameters with the same signature on the moments: a valley, as on the right
- moments that are **step functions** of $\theta$ in the simulator, which can hide a
  valley from a derivative-free optimizer and ruin the finite-difference Jacobian

{cite:t}`andrews2017MeasuringSensitivity` give a measure of what each moment contributes
to each estimate, $\Lambda = -(D^T W D)^{-1} D^T W$, the matrix that maps a perturbation
of the moments into a perturbation of the estimates. It is a by-product of the standard
errors and belongs in every MSM paper.

:::{div}
:class: discussion

- Which moments of the retirement data of [Class 18](18_dcegm.md#dcegm-simulation) would
  identify the disutility of work $\delta$, and which the taste shock scale $\sigma$?
- Wealth is measured with error in survey data. Which of the three moments above is hurt
  most?
:::

## MSM in practice

Why is the method so widely used in applied work?

- it is **not data hungry**: it matches aggregated moments, and can combine different
  sources of data, a wealth profile from one survey and a retirement hazard from another
- it does not rely on distributional assumptions as much as MLE, which needs the whole
  density of the data
- it only needs the simulator, which the structural project has anyway

The price is efficiency: the standard errors are larger than those of the MLE, the
weighting matrix is often simplified because of its small-sample bias, and the choice
of moments is a judgment call that has to be defended. Two relatives extend the idea:
**indirect inference** {cite:p}`gourieroux1993IndirectInference,smith1993EstimatingNonlinear`
matches the coefficients of an auxiliary model, a regression or a VAR, estimated on the
data and on the simulations, and simulated moments of time series
{cite:p}`lee1991SimulationEstimationTimeseriesa,duffie1993SimulatedMomentsEstimation`
match autocovariances of one long simulated path.

Two life-cycle applications of the kind we estimate next class:

- {cite:t}`gourinchas2002ConsumptionLifeCycle` estimate the discount factor and risk
  aversion of a consumption-savings model from the age profile of consumption, with the
  age profile of wealth as a check
- {cite:t}`french2005EffectsHealthWealth` estimates a labor supply and retirement model
  from profiles of hours, participation and assets by age, the model of
  [Class 20](20_msm_practice.md) with health and wages added

{cite:t}`aupens` estimate a dynamic model of labor supply, savings and human capital on
Australian survey data by MSM, and use it to evaluate a reform of the pension system: the
whole arc from model to counterfactual in one paper.

:::{div}
:class: discussion

- An MSM paper reports a table of data moments and simulated moments at the estimates.
  What should the reader look for in it beyond the fit?
- The counterfactual changes a policy parameter and simulates the model again. Which of
  the sources of noise in the estimates carry over to the counterfactual?
:::

````{note} References and additional resources

(19_msm_references)=
- 📖 {cite:t}`mcfaddenMethodSimulatedMoments1989` "A Method of Simulated Moments for Estimation of Discrete Response Models Without Numerical Integration"
- 📖 {cite:t}`pakes1989SimulationAsymptoticsOptimizationa` "Simulation and the Asymptotics of Optimization Estimators"
- 📖 {cite:t}`hansen1982LargeSampleProperties` "Large Sample Properties of Generalized Method of Moments Estimators"
- 📖 {cite:t}`lee1991SimulationEstimationTimeseriesa` "Simulation estimation of time-series models"
- 📖 {cite:t}`duffie1993SimulatedMomentsEstimation` "Simulated Moments Estimation of Markov Models of Asset Prices"
- 📖 {cite:t}`gourieroux1993IndirectInference` "Indirect Inference"
- 📖 {cite:t}`altonji1996SmallSampleBias` "Small-Sample Bias in GMM Estimation of Covariance Structures"
- 📖 {cite:t}`andrews2017MeasuringSensitivity` "Measuring the Sensitivity of Parameter Estimates to Estimation Moments"
- 📖 {cite:t}`gourinchas2002ConsumptionLifeCycle` "Consumption over the Life Cycle"
- 📖 {cite:t}`aupens` "Effects of taxes and safety net pensions on life-cycle labor supply, savings and human capital: The case of Australia"
- 📖 {cite:t}`corbae2024MethodMoments` "Method of Moments Estimators", Chapter 22 of *Quantitative Economics with Heterogeneity: A Guidebook* — the asymptotics, the two-step procedure and the discussion of the weighting matrix in this class follow these notes and the lectures below
- 📖 {cite:t}`adda2023DynamicEconomicsQuantitative` "Dynamic Economics", Chapter 4 on simulated methods
- 📺 Dean Corbae's lectures on the simulated method of moments at the DSE 2024 summer school [youtu.be/61KXTkxZb3o](https://youtu.be/61KXTkxZb3o)
- 📄 Richard W. Evans' notebook on MSM [notes.quantecon.org](https://notes.quantecon.org/submission/5b3db2ceb9eab00015b89f93)
````
