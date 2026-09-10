---
title: "🔬 Programming practice: nested fixed point estimation"
short_title: 🔬 NFXP
subtitle: Class 10 — Thursday, September 24
exports:
  - format: typst
    output: exports/10_nfxp.pdf
downloads:
  - file: 10_nfxp.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

Tuesday's solver becomes an estimator once a likelihood is wrapped around it: for
every parameter vector, solve the model and evaluate the probability of the data.
This class is maximum likelihood for dynamic models, and the nested fixed point (NFXP)
algorithm of {cite:t}`rustOptimalReplacementGMC1987` that makes it fast.

````{hint} Running the code for this lecture
:class: dropdown

The code for this class is in the folder `session10-sep24/` of the course **code
repository**. Today is a practical, so that folder holds the pre-code skeleton we fill
in together, and the completed version appears there afterwards.

You should have cloned that repository already — if not, the instructions are in the
[algorithms and complexity lecture](https://dse.iskh.me/algo#clone-code-repo).

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

## Maximum likelihood estimation

The maximum likelihood estimator is applicable when the model yields a probability
distribution for the observable data. Let $L(x,\theta)$ denote the distribution (pdf)
of the observables $x$ implied by the model with parameter vector $\theta$, and let
$Z_n = (z_1,\dots,z_n)$ denote the data, consisting of $n$ independent observations,
a random sample. MLE and MSM are the two main estimation methods for dynamic economic
models, and the method of simulated moments comes later in the course.

### Likelihood function

If $L(x,\theta)$ is a discrete distribution then $L(x,\theta)$ computed at the data
point gives the probability of observing that data point exactly. If $L(x,\theta)$ is
a continuous distribution then $L(x,\theta)$ computed at the data point is analogous
to the probability of observing it. Either way, independence across observations makes
the joint probability of the sample a product,

$$
\mathcal{L}_n(\theta) = L(Z_n,\theta) = \prod_{i=1}^n L(z_i,\theta)
$$

which is the **likelihood function**: the data are fixed, and it is a function of the
parameters.

````{attention} Definition

The **maximum likelihood estimator** is the parameter vector at which the observed
data are most probable,

$$
\hat{\theta}_{MLE} = \arg\max_{\theta \in \Theta} \, \prod_{i=1}^n \underbrace{L(z_i,\theta)}_{\mathcal{L}(z_i,\theta)} = \arg\max_{\theta \in \Theta} \mathcal{L}_n(\theta)
$$

or, equivalently, the maximizer of the log-likelihood

$$
\hat{\theta}_{MLE} = \arg\max_{\theta \in \Theta} \, \sum_{i=1}^n \underbrace{\log L(z_i,\theta)}_{\ell(z_i,\theta)} = \arg\max_{\theta \in \Theta} \ell_n(\theta)
$$

where $\theta \in \Theta$ is the parameter space and $Z_n = (z_1,\dots,z_n)$ the
observed data.
````

The log is there for the numerics as much as for the algebra: a product of a few
thousand probabilities underflows to zero long before the sample is exhausted, and a
sum of their logs does not. Never take the log of a probability, from Class 6, applies
to every term of $\ell_n(\theta)$.

### Asymptotic properties of MLE

Under regularity conditions the MLE has four properties that make it the default
estimator whenever a likelihood is available:

1. Consistency: $\hat{\theta}_{MLE} \xrightarrow{p} \theta_0$
2. Asymptotic normality:
   $\sqrt{n} ( \hat{\theta}_{MLE} - \theta_0 ) \xrightarrow{d} N(0,\mathcal{I}(\theta_0)^{-1})$
3. Asymptotic efficiency: MLE approaches the smallest possible variance (the
   Cramér–Rao bound) for an unbiased estimator when $n \rightarrow \infty$
4. Functional invariance: the MLE of $\gamma_0 = g(\theta_0)$ is given by
   $g(\hat{\theta}_{MLE})$ if $g(\cdot)$ is a continuously differentiable function

The price is the regularity conditions and the assumption that the model is correctly
specified: a misspecified likelihood still produces an estimate, but the properties
above no longer describe it.

### Asymptotic variance of the MLE

The asymptotic normality result can be read as a statement about the standardized
estimator,

$$
\sqrt{n} ( \hat{\theta}_{MLE} - \theta_0 ) \xrightarrow{d} N(0,\mathcal{I}(\theta_0)^{-1}) \Leftrightarrow
\sqrt{\mathcal{I}_n(\theta_0)} ( \hat{\theta}_{MLE} - \theta_0 ) \xrightarrow{d} N(0,1)
$$

where the variance is given by the inverse of the **Fisher information**, computed
from the pdf $L(x,\theta_0)$ as the expected curvature of the log-likelihood, or
equivalently as the expected squared score:

$$
\mathcal{I}(\theta_0) = - \mathbb{E}\left[ \frac{\partial^2}{\partial\theta \partial\theta} \ell(x,\theta_0) \right] =
\mathbb{E}\left[ \left( \frac{\partial}{\partial\theta} \ell(x,\theta_0) \right)^2 \right]
$$

Alternatively, the Fisher information matrix can be computed from the log-likelihood
function $\ell_n(\theta_0)$ of $n$ i.i.d. random variables in place of the data
$Z_n$, and it scales with the sample size:

$$
\mathcal{I}_n(\theta_0) = - \mathbb{E}\left[ \frac{\partial^2}{\partial\theta \partial\theta} \ell_n(\theta_0) \right] =
\mathbb{E}\left[ \left( \frac{\partial}{\partial\theta} \ell_n(\theta_0) \right)^2 \right] =
n \mathcal{I}(\theta_0)
$$

### Estimating the Fisher information

The Fisher information depends on the model pdf $L(x,\theta_0)$ and is hardly
computable in closed form. It can be consistently estimated thanks to the law of large
numbers,
$-\tfrac{1}{n} \sum_i^n \frac{\partial^2}{\partial\theta \partial\theta} \ell(z_i,\theta) \xrightarrow{p} \mathcal{I}(\theta_0)$,
which gives the *observed* Fisher information

$$
\hat{\mathcal{J}}_n(\theta_0) = - \sum_{i=1}^n \frac{\partial^2}{\partial\theta \partial\theta} \ell(z_i,\theta) =
- \frac{\partial^2}{\partial\theta \partial\theta} \ell_n(\theta_0)
\approx \mathcal{I}_n(\theta_0) = n \mathcal{I}(\theta_0)
$$

Plugging in the estimate $\hat{\theta} = \hat{\theta}_{MLE}$ we have

$$
\sqrt{\hat{\mathcal{J}}_n(\hat{\theta})} ( \hat{\theta} - \theta_0 ) \xrightarrow{d} N(0,1)  \Rightarrow
\hat{\theta} \approx N(\theta_0, \hat{\mathcal{J}}_n(\hat{\theta})^{-1})
$$

so the standard errors of the estimates are the square roots of the diagonal of the
inverse Hessian of the log-likelihood at the optimum. Every optimizer that uses a
Hessian, or an approximation to it, has the standard errors for free.

### Information equality

Similar to the two equivalent definitions for the *expected* Fisher information, for
the *observed* Fisher information it holds, assuming $\theta_0$ is a scalar, that

$$
\hat{\mathcal{J}}_n(\theta_0) = - \frac{\partial^2}{\partial\theta \partial\theta} \ell_n(\theta_0)
= \left( \frac{\partial}{\partial\theta} \ell_n(\theta_0) \right)^2
$$

The square on the right hand side originates in the calculation of the variance of the
score $\frac{\partial \ell_n(\theta_0)}{\partial\theta}$, whose expectation is zero at
the true parameters:

$$
Var\left( \frac{\partial \ell_n(\theta_0)}{\partial\theta} \right) =
\mathbb{E} \Big( \frac{\partial \ell_n(\theta_0)}{\partial\theta} \Big)^2 -
\Big( \underbrace{ \mathbb{E} \frac{\partial \ell_n(\theta_0)}{\partial\theta} }_{=0} \Big)^2
$$

When the parameter $\theta \in \mathbb{R}^K$ is a vector with $K$ elements, both the
second order derivative and the square of the first order derivative have to be
adjusted: the Hessian is a $K\times K$ matrix, and the square becomes an outer product.

### Outer product of gradients

Consider a vector random variable $\tilde{X} = (\tilde{X}_1,\dots,\tilde{X}_K)^{T}$, a
column vector $K \times 1$, with expectation given by the $K \times 1$ vector
$\mathbb{E}\tilde{X}$. The $K \times K$ variance-covariance matrix $\Sigma$ of
$\tilde{X}$ is given by the expectation of the **outer product** $\otimes$,

$$
\Sigma = \mathbb{E} \left\{ \big( \tilde{X}-\mathbb{E}\tilde{X} \big) \otimes \big( \tilde{X}-\mathbb{E}\tilde{X} \big)^{T} \right\}
$$

The variances of the elements of $\tilde{X}$ are on the main diagonal of $\Sigma$,
whereas the off-diagonal elements contain the covariances $cov(\tilde{X}_i,\tilde{X}_j)$
for all $i \ne j$.

### Information matrix equality

Let $H(\ell_n(\theta_0)) = \frac{\partial^2}{\partial\theta \partial\theta} \ell_n(\theta_0)$
denote the $K \times K$ Hessian matrix of $\ell_n(\theta_0)$, let
$\nabla f(\theta) = \frac{\partial}{\partial\theta} f(\theta)$ denote the gradient of
$f(\theta)$, a $K \times 1$ vector, and let
$\nabla\ell(Z_n, \theta_0) = \big( \frac{\partial}{\partial\theta} \ell(z_1,\theta_0),\dots,\frac{\partial}{\partial\theta} \ell(z_n,\theta_0) \big)$
denote the $K \times n$ matrix of gradients of $\ell(z_i,\theta_0)$ stacked for all
$i$. Then we have the **information matrix equality**

$$
\begin{aligned}
\hat{\mathcal{J}}_n(\theta_0) = - H(\ell_n(\theta_0))
&= \sum_{i=1}^n \nabla\ell(z_i, \theta_0) \otimes \nabla\ell(z_i, \theta_0)^{T} \\
&= \nabla\ell(Z_n, \theta_0) \otimes \nabla\ell(Z_n, \theta_0)^{T}
\end{aligned}
$$

The right hand side needs first derivatives only, one score vector per observation.
That is what makes it useful inside an optimizer.

### Berndt–Hall–Hall–Hausman (BHHH) algorithm

The information matrix equality gives an approximation of the Hessian of the
log-likelihood function that can be used in a quasi-Newton optimization method,

$$
H(\ell_n(\theta)) \approx - \nabla\ell(Z_n, \theta) \otimes \nabla\ell(Z_n, \theta)^{T}
$$

which is the **BHHH algorithm** of {cite:t}`berndtEstimationInferenceNonlinear1974`,
introduced with the other quasi-Newton methods in [Class 5](5_solvers.md). The outer
product of gradients is a positive semi-definite matrix for every $\theta$, so even if
the approximation is not accurate it never points the Newton iteration in the wrong
direction. A search for an appropriate step size in the direction found with the
approximated Hessian is part of the algorithm to ensure global convergence.

Where it breaks: the equality holds at $\theta_0$, in expectation, for a correctly
specified model. Far from the optimum, in small samples, or under misspecification the
outer product is a poor Hessian, and the standard errors it implies are wrong even
when the point estimate is fine. The **sandwich** estimator that combines both sides
of the equality is the robust alternative.

:::{div}
:class: discussion

- Why maximize the log-likelihood rather than the likelihood in code?
- Which of the four asymptotic properties survive when the model is misspecified?
- BHHH needs one score vector per observation. What is an observation in the bus
  data: a bus, a month, or a bus-month?
:::

## Estimating the bus engine model with NFXP

For every value of the structural parameters
$\theta = (RC,\theta_1,\theta_{20},\dots,\theta_{2J},\beta)$ we can solve the model
fast using NK iterations within the poly-algorithm, and based on the solution
$EV_\theta(x,d)$ form the choice probabilities $P(\text{keep}|x,\theta)$ and
$P(\text{replace}|x,\theta)$. Given data on mileage $x$ and choices $d$ we can then
form the likelihood function and proceed with maximum likelihood estimation.

### Data

Harold Zurcher kept maintenance records of 162 buses in 8 groups, with monthly
observations of the mileage on each bus (the odometer reading) and data on the
maintenance operations:

1. Routine periodic maintenance (i.e. brake adjustment)
2. Replacement or repair at the time of failure
3. Major engine overhaul and/or replacement — *the focus of the paper*

The data are $(x_{i,t},d_{i,t})$, where $x_{i,t}$ is the discretized mileage (bin
indexes) and $d_{i,t}$ the observed choice at this mileage for each bus $i$ in each
month $t$.

### Likelihood function

The model delivers two conditional probabilities per observation, the choice given
the state and the state transition given the previous state and choice, so

$$
\mathcal{L}_n(\theta, EV_\theta) = \prod_{i=1}^{162}\prod_{t=2}^{T_i} P(d_{i,t}|x_{i,t}) \pi(x_{i,t}|x_{i,t-1},d_{i,t-1})
$$

$$
\ell_n(\theta,EV_\theta) = \log \mathcal{L}_n(\theta,EV_\theta) = \sum_{i=1}^{162}\sum_{t=2}^{T_i} \big ( \log P(d_{i,t}|x_{i,t}) + \log \pi(x_{i,t}|x_{i,t-1},d_{i,t-1}) \big)
$$

The dependence on $EV_\theta$ is written explicitly because the choice probabilities
are a function of the solution of the model, not of $\theta$ directly. The transition
part depends on $\theta_2$ only and does not involve the solution at all.

### MLE estimator

$$
\hat{\theta} = \arg\max_\theta \ell_n(\theta, EV_{\theta})
$$

This is an unconstrained optimization, but it requires the computation of
$EV_{\theta}$ for each value of the parameter $\theta$ — a fixed point problem inside
every function evaluation.

### Nested loop

The **outer loop** is a hill-climbing algorithm. The log-likelihood function
$\ell_n(\theta,EV_{\theta})$ is maximized with respect to $\theta$ by a quasi-Newton
algorithm with the BHHH approximation of the Hessian, and each evaluation of
$\ell_n(\theta,EV_{\theta})$ requires the solution for the fixed point $EV_{\theta}$.

The **inner loop** is the fixed point algorithm: the solver for the fixed point of the
Bellman operator $EV_{\theta} = \Gamma(EV_{\theta})$, successive approximations (VFI)
plus Newton–Kantorovich iterations in the poly-algorithm of Tuesday.

```{image} _static/img/nfxp.gif
:width: 100%
:align: center
:alt: NFXP algorithm
```

### Important details

1. **Performance:** a gradient based Newton method to maximize the likelihood
2. **Analytical gradients:** using the implicit function theorem and the chain rule for
   the outer loop, and the Fréchet derivative for the inner loop
3. **Use BHHH:** the outer product of gradients approximation for the Hessian
4. **Numerical stability:** recenter the logsum and the choice probabilities, the
   de-maxing trick of Class 6
5. **Further info:** the NFXP manual, see the references below

Each of these is a difference between an estimator that runs in seconds and one that
runs overnight. The comparison in {cite:t}`ecma_comment` is exactly the difference
between an NFXP with items 1–3 and one without.

### Analytical gradient of the likelihood function

Differentiating the log-likelihood term by term,

$$
\frac{\partial}{\partial \theta} \ell_n(\theta,EV_\theta) = \sum_{i=1}^{162}\sum_{t=2}^{T_i} \big ( \frac{\partial}{\partial \theta} \log P(d_{i,t}|x_{i,t}) + \frac{\partial}{\partial \theta} \log \pi(x_{i,t}|x_{i,t-1},d_{i,t-1}) \big)
$$

is straightforward with multiple applications of the chain rule. The key point is
expressing the derivatives of the expected value function
$\frac{\partial EV_{\theta}}{\partial \theta}$, which is done with the Banach space
version of the *implicit function theorem* applied to the fixed point equation:

$$
\frac{\partial EV_{\theta}}{\partial \theta} = \frac{\partial \Gamma_{\theta}}{\partial \theta} + \frac{\partial \Gamma_{\theta}}{\partial EV_{\theta}} \frac{\partial EV_{\theta}}{\partial \theta}
\Rightarrow
\frac{\partial EV_{\theta}}{\partial \theta} = \Big( I - \frac{\partial \Gamma_{\theta}}{\partial EV_{\theta}} \Big)^{-1}  \frac{\partial \Gamma_{\theta}}{\partial \theta}
$$

Here $\Gamma_{\theta}(\cdot)$ is the finite approximation of the Bellman operator in
expected value function space, and
$\frac{\partial \Gamma_{\theta}}{\partial EV_{\theta}}$ is the finite approximation of
its Fréchet derivative — the same matrix the NK step already computes and factorizes.
The gradient of the likelihood is therefore nearly free once the model is solved by
NK: one more solve with the same matrix and a different right hand side.

📖 John Rust "NFXP Pocket Guide" {download}`Download pdf <_static/pdf/nfxp.pdf>`

:::{div}
:class: discussion

- $\theta_2$ enters the likelihood through $\pi$ only. Can it be estimated without
  solving the model at all, and what does that suggest for the estimation strategy?
- $\beta$ is listed among the structural parameters. Would you try to estimate it?
- The inner loop is warm-started from the previous $EV_\theta$. When is that a good
  idea, and when does it hurt?
:::

(task10.1)=
````{warning} Practical task 10.1: NFXP estimation of the bus engine model

We write this code together in class, starting from the pre-code skeleton
`session10-sep24/nfxp_pre.ipynb`, which carries the model class of Tuesday.
**Pull the code repository before the class:**

```bash
cd sb-dse-code && git pull
```

Nothing here is collected — finish what is left over at home, and keep the notebook,
because the estimator is reused for the comparison with MPEC and CCP methods later in
the course.

1. **Simulator.** Add `simulate(N, T, seed)` to the model class: draw the initial
   mileage, then for each bus and month draw the choice from $P(d|x)$ and the mileage
   increment from $\theta_2$. Return the panel $(x_{i,t}, d_{i,t})$.
2. **Transition parameters.** Estimate $\theta_2$ from the increment frequencies in
   the simulated data, in closed form, and check it against the values you simulated
   from.
3. **Log-likelihood.** Write `loglik(theta)` for $\theta = (RC, \theta_1)$ that solves
   the model with the poly-algorithm and sums the log choice probabilities. Use the
   de-maxed logit of Class 6, never `np.log` of a probability.
4. **Estimate.** Maximize with `scipy.optimize.minimize` and numerical gradients.
   Count the calls to the inner solver, and recover the true parameters.
5. **Analytical gradient.** Compute $\partial EV_\theta/\partial\theta$ from the
   implicit function theorem using the Fréchet derivative, then the score of every
   bus-month. Verify against finite differences, then pass the gradient to the
   optimizer and count the solver calls again.
6. **Standard errors.** Compute them from the outer product of the scores (BHHH) and
   from a finite difference Hessian, and compare.

Use of AI assistance is allowed and encouraged, subject to the
[course AI policy](https://dse.iskh.me/#ai-policy) — but you must be able to explain
every line, including where the implicit function theorem enters the code.
````

(10_nfxp_references)=
````{note} References and additional resources

- 📖 {cite:t}`rustOptimalReplacementGMC1987` "Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher"
- 📖 {cite:t}`rustNestedFixedPoint2000` "Nested Fixed Point Algorithm Documentation Manual", version 6 {download}`Download pdf <_static/pdf/nfxp_man_2000.pdf>`
- 📖 John Rust "NFXP Pocket Guide" {download}`Download pdf <_static/pdf/nfxp.pdf>`
- 📖 {cite:t}`ecma_comment` "Constrained optimization approaches to estimation of structural models: Comment"
- 📖 {cite:t}`berndtEstimationInferenceNonlinear1974` "Estimation and Inference in Nonlinear Structural Models" — the original BHHH paper
- 📖 {cite:t}`wooldridge2010EconometricAnalysisCross` "Econometric Analysis of Cross Section and Panel Data", chapters 12–13 on M-estimation and MLE
- 📺 Econometric Society Dynamic Structural Econometrics (DSE) lecture by Bertel Schjerning [YouTube video](https://youtu.be/houBb2vQFZE?si=VOIG544hnAiOx18x)
- Matlab implementation of the full solver and the NFXP estimator [DSE 2019 GitHub repo](https://github.com/dseconf/DSE2019/tree/master/02_DDC_SchjerningIskhakov)
- `ruspy` Python package implementing NFXP <https://github.com/OpenSourceEconomics/ruspy>

````
