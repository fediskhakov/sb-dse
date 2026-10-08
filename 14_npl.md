---
title: 📖 Nested pseudo-likelihood (NPL)
short_title: 📖 NPL estimator
subtitle: Class 14 — Thursday, October 8
downloads:
  - file: 14_npl.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

Compared to NRLS the [two-step estimator of Class 12](12_ccp.md#ccp-estimation) trades efficiency for speed.
{cite:t}`aguirregabiriaSwappingNestedFixed2002` provides the middle ground: the estimation algorithm that allows the researcher to decide how much of the model to solve, and how much to rely on the data. The main idea: use the Bellman equation as a fixed point in the space of choice probabilities, and iterate on it only as many times as the data requires.

````{seealso} Key reading

{cite:t}`aguirregabiriaSwappingNestedFixed2002` "Swapping the Nested Fixed Point
Algorithm: A Class of Estimators for Discrete Markov Decision Models", *Econometrica*
70(4), 1519–1543. The policy iteration operator in probability space, the
pseudo-likelihood estimator and the NPL iterations of this class all come from this
paper.
````

````{hint} Running the code for this lecture
:class: dropdown

The code for this class is in the folder `session14-npl/` of the course **code
repository**. The module `ccp.py` there collects the two-step CCP estimator of
[Class 13](13_ccp_practice.md), the first-stage CCPs and the second-stage logit, and it
reuses `zurcher.py`, `nfxp.py` and the bus data from `session10-11/`, so keep both
folders side by side.

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

````{danger} Homework theta_npl: iterating the two-step CCP estimator
:class: dropdown

(homework-theta_npl)=
Graded homework: task `theta_npl` in the class repository. Collect it and work on a
copy in your own repository:

```bash
git pull upstream main                     # collect the task
cp -r tasks/theta_npl solutions/theta_npl  # work on the copy
```

Take the two-step CCP estimator of [Class 13](13_ccp_practice.md) and turn it into an
iterative procedure similar to NPL.

The notebook `npl_iterated_ccp.ipynb` in the task folder fetches the code of the
[NFXP estimator](10_nfxp.md) and of the two-step CCP estimator, `ccp.py`, and carries
the details of the task.

Submit it as a pull request, following the [git workflow](2_workflow.md#submission). One of you presents a solution at the
start of the next Tuesday class.
````

## From value space to probability space

Like all CCP-based methods, pseudo-likelihood estimation starts with the consistent
estimation of the CCPs and transition probabilities from the data. Let's focus on the
specification of the statistical criterion for the second stage. Return to the
representation of the integrated value function through the correction terms,

$$
V^\sigma(x) =
\sum_{d \in D(x)} P(d|x)
\big(
v(x,d) + e(x,d)
\big)
$$

Using the definition of the choice-specific value functions

$$
v(x,d) = u(x,d) + \beta \int_X V^\sigma(x') \pi(x'|x,d) dx'
$$ 

we have

$$
V^\sigma(x) =
\sum_{d \in D(x)} P(d|x)
\left(
u(x,d) + \beta
\int_{X} V^\sigma(x')
\pi(x'|x,d) dx' + e(x,d)
\right)=
$$

$$
=
\sum_{d \in D(x)} P(d|x)
\left[
u(x,d) + e(x,d)
\right]
+ \beta
\sum_{d \in D(x)} P(d|x)
\int_{X} V^\sigma(x')
\pi(x'|x,d) dx'
$$

which is linear in $V^\sigma$ once the CCPs are held fixed. Again assuming a discrete
state space and stacking $V^\sigma(x)$, $P(d|x)$, $u(x,d)$ and $e(x,d)$ across the
state space to form column vectors, we can write the last equation for every point of
the state space in matrix form,

$$
V^\sigma =
\sum_{d \in D(x)} P(d) \ast
\left[
u(d) + e(d)
\right]
+ \beta
\sum_{d \in D(x)} P(d) \ast
\Pi(d) V^\sigma
$$

where $\ast$ is the element-wise (Hadamard) product of vectors. Let

$$
\Pi = \sum_{d \in D(x)} P(d) \ast \Pi(d)
$$

be the $|X|\times|X|$ *unconditional* transition matrix, the same Markov chain whose
stationary distribution was computed in homework [`zeta_zurcher`](9_zurcher.md#homework-zeta_zurcher), and denote by $I$
again the identity matrix of the same size. Then the linear system solves as

$$
V^\sigma = [I - \beta \Pi]^{-1}
\sum_{d \in D(x)} P(d) \ast
\left[
u(d) + e(d)
\right]
$$

This is the policy evaluation step of policy iterations, where the policy is given by the
CCPs. Recall that the correction term $e(d)$ is a function of the CCPs only, and therefore we can introduce an operator

$$
\varphi : [0,1]^{|X|\times|D|} \ni P \mapsto V^\sigma \in \mathbb{R}^{|X|}
$$

that maps the CCPs into the integrated value functions. Finally, denote by

$$
\Lambda: \mathbb{R}^{|X|} \ni V^\sigma \mapsto \{P(d|x)\} \in \mathbb{R}^{|X|\times|D|}
$$

the mapping from the integrated value functions to the CCPs given by the [choice probability formulas of Class 12](12_ccp.md#ccp-choice-probabilities), which in the simple EV1 case take the form of the
multinomial logit

$$
\Lambda(V^\sigma) =
\left\{\frac{\exp[u(x,d) + \beta \Pi(d) V^\sigma]}{\sum_{d'\in D(x)} \exp[u(x,d') + \beta \Pi(d') V^\sigma]}
\right\}_{d \in D(x), x \in X}
$$

````{attention} Definition

Let $\mathcal{P}$ be the set of CCP arrays $P = \{P(d|x)\}$ with $P(d|x) > 0$ and
$\sum_{d \in D(x)} P(d|x) = 1$ for every $x \in X$. Following
{cite:t}`aguirregabiriaSequentialEstimationDynamic2007` we define the
**policy iteration operator** as the composition

$$
\Psi = \Lambda \circ \varphi : \mathcal{P} \ni P \mapsto \Lambda(\varphi(P)) \in \mathcal{P}
$$

which maps the CCPs into the CCPs through the integrated value functions. For given
structural parameters $\theta$, the CCPs $P_\theta$ of the solution of the model are a
fixed point of $\Psi$,

$$
P_\theta = \Psi(P_\theta)
$$
````

We have encountered Policy Iteration (or Howard ) just before talking about the Newton–Kantorovich iterations in [Class 9](9_zurcher.md#nk-iterations).
Compare the definition of $\Psi$ operator and the operations required to evaluate it to the definition of the [Howard policy iteration](9_zurcher.md#howard-policy-iterations) to convince yourself that $\Psi$ is the Howard operator in the CCP space (and so the maximum is replaced by the logit formula).


Note that $\Psi$ depends on the structural parameters of the model $\theta$ through
$u(x,d)$ and $\beta$, so we write $\Psi(P,\theta)$ when the dependence matters. 

The original fixed point problem in *value space* is thereby reformulated as a fixed point
problem in *probability space*. One application of $\Psi$ is one policy evaluation, a
linear solve, followed by one policy improvement, a logit formula, so successive
approximations $P_{k+1} = \Psi(P_k,\theta)$ converge as fast as policy iterations do.

````{tip} Example: the policy iteration operator in the bus engine model

The ingredients of $\varphi$ for the [Zurcher model](9_zurcher.md#zurcher-transitions),
with $\theta = (RC, c)$ and two choices, keep and replace:

- $\Pi(\text{keep})$ is the mileage transition matrix, and $\Pi(\text{replace})$ has
  every row equal to the first row of $\Pi(\text{keep})$, since a replaced engine starts
  from zero mileage
- the flow utilities are both linear in $\theta = (RC, c)$
$$
u(x,\text{keep}) = -0.001\,c\,x = H(d)\,\theta \\
u(x,\text{replace}) = -RC = H(d)\,\theta
$$ 
  with the rows of $H(\text{keep})$ equal to $(0, -0.001x)$ and those of $H(\text{replace})$ to
  $(-1, 0)$
- with EV1 shocks the correction term is $e(x,d) = \gamma - \log P(d|x)$

Because $u(d)$ is linear in $\theta$ and $e(d)$ does not depend on it, the policy
valuation is linear in $\theta$ too:

$$
V^\sigma = \varphi(P) = A(P)\,\theta + b(P),
$$
$$
A(P) = [I - \beta \Pi]^{-1} \sum_{d} P(d) \ast H(d),
$$
$$
b(P) = [I - \beta \Pi]^{-1} \sum_{d} P(d) \ast e(d)
$$

Both are one linear solve with the same matrix, three right-hand sides in total in the code solved by `linalg.solve` at once.

Policy improvement then needs only the difference of the choice-specific values:

$$
\begin{aligned}
& v(x,\text{keep}) - v(x,\text{replace}) = \\
& \qquad\qquad u(x,\text{keep}) - u(x,\text{replace}) + \beta \big(\Pi(\text{keep}) - \Pi(\text{replace})\big) V^\sigma
\end{aligned}
$$
$$
v(x,\text{keep}) - v(x,\text{replace}) = \tilde h(x)'\theta + \tilde e(x), 
$$
$$
\tilde h(x)' = \big[H(\text{keep}) - H(\text{replace}) + \beta\,\big(\Pi(\text{keep}) - \Pi(\text{replace})\big) A(P)\big]_x,
$$
$$
\tilde e(x) = \beta\,\big[\big(\Pi(\text{keep}) - \Pi(\text{replace})\big) b(P)\big]_x
$$

where $[\cdot]_x$ is row $x$. The constant $\gamma$ adds $\gamma/(1-\beta)$ to every
element of $V^\sigma$, and it drops out of $\tilde e(x)$ because the rows of
$\Pi(\text{keep}) - \Pi(\text{replace})$ sum to zero. 

The policy iteration operator is the binary logit

$$
\Psi(P,\theta)(\text{keep}|x) =
\frac{\exp\big(\tilde h(x)'\theta + \tilde e(x)\big)}{1 + \exp\big(\tilde h(x)'\theta + \tilde e(x)\big)}
$$

- $\tilde h(x)$ and $\tilde e(x)$ depend on $P$ but not on $\theta$. Once they are
  computed, $\log\Psi(\hat P,\theta)$ is the log-likelihood of a static logit with
  regressors $\tilde h(x)$ and offset $\tilde e(x)$, the same logit as the second stage
  of the [two-step CCP estimator of Class 13](13_ccp_practice.md#zurcher-finite-dependence),
  with a different offset.
- The code is `npl_regressors()` and `psi()` in `npl_practice.ipynb` of the folder
  `session14-npl/`.
````

## Pseudo-maximum likelihood estimation

The operator $\Psi$ turns any first-stage CCP estimate into a set of model-implied
choice probabilities, and those can be put into a likelihood. 

The **pseudo-likelihood
function** is the log-likelihood of the observed choices evaluated at
$\Psi(\hat{P},\theta)$ rather than at the solution of the model,

$$
\ell_n(\theta) = \log \mathcal{L}_n(\theta) = \sum_{i=1}\sum_{t} \log \Psi(\hat{P},\theta)(d_{i,t}|x_{i,t})
$$

and the **pseudo-maximum likelihood estimator** is its maximizer,

$$
\hat{\theta}_{PML} = \arg\max_{\theta} \ell_n(\theta)
$$

The pseudo-likelihood is the likelihood of a model that has been solved for one policy
iteration only, starting from the data. The resulting algorithm is a two-step estimator in the sense of [Class 12](12_ccp.md#ccp-estimation), with a particular choice of criterion:

```
Input: panel data (x_it, d_it)
Algorithm:
  1. First stage: estimate P̂(d|x) and Π̂(d) for all x and d
  2. Specify the pseudo-likelihood ℓ_n(θ) = Σ log Ψ(P̂,θ)(d_it|x_it)
     using the estimated CCPs and transition probabilities
  3. Maximize ℓ_n(θ) over θ
Output: θ̂_PML
```

Every evaluation of $\ell_n(\theta)$ costs one application of $\Psi$, which is one
linear solve, instead of a full solution of the model. Because $\hat{P}$ is held fixed
inside the maximization, the gradient of $\ell_n(\theta)$ is the gradient of a static
logit with the value function differences $\beta\Pi(d)\varphi(\hat P)$ treated as a
fixed regressor, and [BHHH from Class 10](10_nfxp.md#bhhh) applies unchanged.

Where it breaks: in finite samples, exactly where the two-step estimator breaks. The
criterion is a likelihood only in name, since the probabilities are computed at
$\hat{P}$ and not at the model's own fixed point, and the first-stage noise passes
through $\varphi$ into every term.

Asymptotically nothing is lost, though, against the conventional wisdom of the time:
{cite:t}`aguirregabiriaSwappingNestedFixed2002` show that $\hat{\theta}_{PML}$ has
[the same limiting distribution as the MLE](#npl-asymptotics). The price is paid in
finite samples, where its error is [several times that of the MLE](#npl-monte-carlo).

## Swapping the nested fixed point

Aguirregabiria and Mira take this idea a step further and develop the **nested
pseudo-likelihood (NPL) estimator**, which is nothing else but the iteration of the
pseudo-maximum likelihood estimator, with the CCPs updated by $\Psi$ at the current
parameter estimate after every maximization:

$$
\hat{P}_0 \rightarrow
\ell_n(\theta) \rightarrow \hat{\theta}_1 \rightarrow
\hat{P}_1 = \Psi(\hat{P}_0,\hat{\theta}_1) \rightarrow
\ell_n(\theta) \rightarrow \hat{\theta}_2 \rightarrow
\hat{P}_2 = \Psi(\hat{P}_1,\hat{\theta}_2) \rightarrow
\dots
$$

The name says what happened to NFXP. There, the outer loop climbs the likelihood and
the inner loop solves the model to convergence at every step. Here the loops are
swapped: the outer loop takes one policy iteration step on the CCPs, and the inner loop
maximizes a pseudo-likelihood in which the CCPs are fixed.

```
Input: panel data (x_it, d_it), first-stage estimates P̂_0 and Π̂(d)
Algorithm:
  for K = 1, 2, ...
    1. θ̂_K = argmax_θ Σ log Ψ(P̂_{K-1},θ)(d_it|x_it)   # pseudo-likelihood step
    2. P̂_K = Ψ(P̂_{K-1}, θ̂_K)                            # policy iteration step
    3. stop when ‖P̂_K - P̂_{K-1}‖ < tol
Output: θ̂_K for the chosen K, and the NPL fixed point (θ̂, P̂) in the limit
```

This is an intuitive definition of the NPL operator, the fixed point of which provides
the NPL estimator. 

The pair $(\hat{\theta},\hat{P})$ at convergence satisfies two conditions:
$\hat{P} = \Psi(\hat{P},\hat{\theta})$, so $\hat{P} = P_{\hat\theta}$ is the solution of
the model at $\hat{\theta}$, and the first order conditions of the pseudo-likelihood at
$\hat P$, which at this fixed point coincide with the likelihood equations of NFXP
([shown below](#npl-zero-jacobian)).

### The sequence of estimators

Stopping the iteration after $K$ steps gives a whole class of estimators, which
{cite:t}`aguirregabiriaSwappingNestedFixed2002` call the **$K$-stage policy iteration
estimators**. They bridge the gap between the two-step CCP estimator and NFXP:

- $K=1$ is the pseudo-maximum likelihood estimator above. It is a
  [Hotz–Miller CCP estimator](12_ccp.md#ccp-estimation) whose moment conditions use the
  scores of the pseudo-likelihood, $\partial \log \Psi(\hat P_0,\theta)(d|x)/\partial\theta$,
  as instruments
- $K \rightarrow \infty$: if the iterations converge, the limit is a root of the
  likelihood equations, that is, the NFXP estimator when the likelihood has a single
  root
- every $K$ in between is $\sqrt{n}$-consistent, asymptotically normal and
  asymptotically equivalent to the MLE, as shown [below](#npl-asymptotics)
- each stage costs one policy iteration, a linear solve of the size of the state space,
  plus one pseudo-likelihood maximization, in which $\hat P_{K-1}$ is fixed and no
  matrix is inverted again

The last point is the computational case for NPL. With EV1 shocks and utility linear in
the parameters, $u(x,d) = h(x,d)'\theta$, the pseudo-likelihood is a static logit with
fixed regressors: globally concave in $\theta$, so Newton or BHHH converges from any
starting value.

(npl-zero-jacobian)=
## Why the first stage washes out

All the econometric results of {cite:t}`aguirregabiriaSwappingNestedFixed2002` rest on
one property of the policy iteration operator.

````{attention} Proposition (Aguirregabiria and Mira, 2002, Propositions 1–2)

Let the shocks be additively separable, with a density that is continuous and twice
differentiable and has full support on the real line, let conditional independence hold,
let the state space $X$ be finite and $\beta \in (0,1)$. Then for given $\theta$:

- $\Psi$ has a unique fixed point $P_\theta \in \mathcal{P}$, and the sequence
  $P_K = \Psi(P_{K-1})$, $K = 1, 2, \dots$ converges to $P_\theta$ from any
  $P_0 \in \mathcal{P}$
- for any $P_0 \in \mathcal{P}$, the values $V_K = \varphi(P_K)$ are the
  Newton–Kantorovich iterations on the Bellman equation for $V^\sigma$, started from
  $V_0 = \varphi(P_0)$
- the Jacobians of $\varphi$ and $\Psi$ with respect to the CCPs vanish at the fixed
  point:

$$
\left.\frac{\partial \varphi(P)}{\partial P'}\right|_{P = P_\theta} = 0,
\qquad
\left.\frac{\partial \Psi(P)}{\partial P'}\right|_{P = P_\theta} = 0
$$

where the derivatives are taken with respect to the free CCPs, $P(d|x)$ for all $x$ and
all $d$ except one reference alternative, since the CCPs in each state sum to one.
````

The intuition is an envelope argument. Treat the CCPs as the controls of a reformulated
problem, with $\varphi(P)$ the value of following them. At the optimal $P$ the value is
maximized, so a small change in the CCPs changes values, and hence $\Psi$, only to
second order. The second bullet answers the question put above about the cost of
$\Psi$: one application of $\Psi$ *is* one
[Newton–Kantorovich step](9_zurcher.md#nk-iterations).

What does the zero Jacobian buy? Differentiate the fixed point $P_\theta = \Psi(P_\theta,\theta)$
with respect to $\theta$ by the implicit function theorem, which applies because the
matrix in brackets is the identity at $P_\theta$:

$$
\frac{\partial P_\theta}{\partial \theta'}
= \left[ I - \frac{\partial \Psi(P_\theta,\theta)}{\partial P'} \right]^{-1}
\frac{\partial \Psi(P_\theta,\theta)}{\partial \theta'}
= \frac{\partial \Psi(P_\theta,\theta)}{\partial \theta'}
$$

so the pseudo-score evaluated at $P = P_\theta$ equals the score of the likelihood,
$\partial \log \Psi(P_\theta,\theta)(d|x)/\partial\theta = \partial \log P_\theta(d|x)/\partial\theta$.
Two results follow:

- **NPL is NFXP at convergence** (Proposition 3). Suppose the pseudo-likelihood
  maximization in step 1 has a unique interior solution for any sample and any
  $P \in \mathcal{P}$. If the NPL iterations converge to $(\hat\theta,\hat P)$, then
  $\hat P = P_{\hat\theta}$ and $\hat\theta$ is a root of the likelihood equations. This
  holds for the full likelihood as well as the partial one.
- **Convergence itself is not proved.** The paper only shows that NPL converges locally
  with probability approaching one as $n$ grows. Neither NFXP nor NPL guarantees the
  global maximum, so both should be started from several points.

(npl-asymptotics)=
## Asymptotics of the $K$-stage estimators

The paper follows the partial likelihood strategy of
[NFXP](10_nfxp.md#nfxp-likelihood): the transition probabilities $p$ are estimated first
from the observed transitions, $\beta$ is known, and $\theta$ is estimated given $\hat p$.

````{attention} Proposition (Aguirregabiria and Mira, 2002, Proposition 4)

Let the first-stage $\hat P_0$ and $\hat p$ be strongly consistent and jointly
$\sqrt{n}$-asymptotically normal. Let every state have positive probability,
$\Pr(x_i = x) > 0$, let $\Psi(P,\theta)(d|x) > 0$ everywhere, and let $\theta^*$ be
identified: for any $\theta \neq \theta^*$, $\Psi(P^*,\theta) \neq P^*$. Then for every
$K \geqslant 1$

$$
\sqrt{n}\,(\hat\theta_K - \theta^*) \rightarrow_d N(0, V^*)
$$

where $V^*$ is the asymptotic variance of the partial maximum likelihood estimator. With
$p$ known, $V^* = \Omega^{-1}$, the inverse of the information matrix
$\Omega = E\big[\partial \log P_{\theta^*}(d_i|x_i)/\partial\theta \;
\partial \log P_{\theta^*}(d_i|x_i)/\partial\theta'\big]$.
````

The first stage drops out because of the zero Jacobian. In a Taylor expansion of the
first order conditions of $\hat\theta_K$, the first-stage error enters as

$$
\left[\frac{1}{n}\sum_{i} \frac{\partial^2 \log \Psi(\hat P_{K-1},\hat\theta_K)(d_i|x_i)}{\partial\theta\,\partial P'}\right]
\sqrt{n}\,(\hat P_{K-1} - P^*)
$$

and the bracket converges to zero by the zero Jacobian and the information matrix
equality, while $\sqrt{n}(\hat P_{K-1} - P^*)$ stays bounded in probability. Therefore:

- **the precision of the first stage does not matter asymptotically**, so any
  $\sqrt{n}$-consistent first stage will do: the frequency estimator, with empty cells
  set to zero, nearest neighbors or a kernel
- **standard errors need no fixed point**: $V^*$ is estimated with
  $\partial\Psi(\hat P_{K-1},\hat\theta_K)/\partial\theta$ in place of
  $\partial P_{\theta}/\partial\theta$
- **a CCP estimator need not be inefficient.** What costs efficiency is the
  representation of the value function used in the second stage, not the two-step
  structure

The representation matters because Hotz and Miller also proposed a cheaper one for
models with a terminating action, which equals $\varphi(P)$ only at the fixed point and
has no zero Jacobian. CCP estimators built on it are not asymptotically equivalent to
the MLE. The [one-period finite dependence of Class 13](13_ccp_practice.md#zurcher-finite-dependence)
is of the same kind: it uses the CCPs one period ahead instead of the full $\varphi(\hat P)$,
which is consistent with the standard error of $c$ found there, 5.4 against 0.3 for NFXP.

(npl-monte-carlo)=
### Finite sample evidence

Asymptotic equivalence says nothing about finite samples, which is where one-stage
estimators earned their bad reputation. The Monte Carlo of the paper uses Rust's bus
model at the ML estimates, $\theta_0 = 10.47$, $\theta_1 = 0.58$, $\beta = 0.9999$, on a
201-point grid: 1000 samples of 1000 observations each, with a kernel first stage.

| | 1-PI | 2-PI | 3-PI | MLE |
| :-- | --: | --: | --: | --: |
| $\theta_0$ mean absolute error, vs MLE | +4.7% | +1.6% | +0.3% | 2.08 |
| $\theta_1$ mean absolute error, vs MLE | +18.7% | +1.5% | +0.2% | 0.17 |
| $\theta_1$ standard deviation, vs MLE | +11.0% | +1.3% | +0.2% | 0.16 |
| $\theta_0$ estimated s.e. / Monte Carlo s.d. | 0.80 | 1.01 | 1.03 | 1.02 |
| $\theta_1$ estimated s.e. / Monte Carlo s.d. | 0.67 | 1.04 | 1.08 | 1.07 |

The MLE column gives the levels; the other entries in the first three rows are
differences relative to the MLE. The table says:

- **precision improves monotonically with $K$**, and almost all of the gain comes from
  the second stage
- **the 1-stage standard errors are too small**, by 20% for $\theta_0$ and a third for
  $\theta_1$, so inference after one stage overstates precision

On speed, policy iterations take almost all of the estimation time once the grid exceeds
200 points, and their number does not depend on the grid size. 
- NFXP needed 5.5 times as many policy iterations as NPL with 2 parameters and 9 times with 4, starting both from
the Hotz–Miller estimate, and 10 and 15 times from arbitrary starting values. 
- Their NFXP solved the inner problem by policy iterations, which are
[Newton–Kantorovich steps](9_zurcher.md#nk-iterations), warm-started at the previous
CCPs and with an analytical gradient, so this is a comparison with a strong NFXP.

Where it breaks: the NPL iterations are successive approximations on
$(\theta,P) \mapsto (\hat\theta(P), \Psi(P,\hat\theta(P)))$, and **nothing guarantees
that this map is a contraction**. 
- In single-agent models it usually is, since $\Psi$
alone is a policy iteration step, but the parameter update can push it out of the
contraction region when the first stage is poor. 
- In dynamic games $\Psi$ is not a
contraction in general, the NPL fixed point need not be unique, and the iterations can
converge to a fixed point that is not the MLE, or fail to converge at all, see
{cite:t}`pesendorfer2010SequentialEstimationDynamic`.

:::{div}
:class: discussion

- NPL at convergence solves the same fixed point as NFXP. Where does the computational
  saving come from, then?
- Why does the asymptotic distribution not depend on $K$, while the finite sample
  distribution does?
:::

(npl-vs-nfxp)=
## NFXP versus $K$-stage NPL: what actually differs

Same first-order asymptotics under correct specification
([Proposition 4](#npl-asymptotics)). The differences:

- **Finite samples depend on the first stage**: large error at small $K$, close to NFXP
  after 2–3 stages {cite:p}`kasahara2008PseudolikelihoodEstimationBootstrap`
- **Standard errors at $K=1$ are too small**, accurate from $K=2$
- **$K$-stage NPL needs a $\sqrt n$-consistent first stage**: every state observed, no
  zero CCPs, no unobserved heterogeneity
- **Misspecification breaks the equivalence**: different $K$ have different probability
  limits, none equal to that of NFXP
- **The NPL limit is a root of the likelihood equations, not necessarily the maximum**;
  in games it can be inconsistent {cite:p}`pesendorfer2010SequentialEstimationDynamic`

NFXP *is* the MLE; $K$-stage NPL approximates it.

(npl-counterfactuals)=
### In the end the model is solved anyway

Estimation is not the goal. A structural model is estimated to run
[counterfactuals](1_intro.md), such as the
[demand for engine replacement as a function of its price](10_nfxp.md#nfxp-counterfactuals)
in Rust's paper, and a counterfactual needs the solution of the model under the
counterfactual scenario.

Why can the CCP shortcut not be used there too?

- **The first-stage CCPs describe the status quo only.** A counterfactual changes a
  primitive, $RC$ or $\beta$ or the transitions, and with it the behavior. Data on
  behavior under the new primitives do not exist, so $\hat P$ cannot stand in for it.
- **The counterfactual CCPs are a fixed point**, $\tilde P = \Psi(\tilde P, \tilde\theta)$
  at the counterfactual parameters $\tilde\theta$, which is the model solved in
  probability space. The demand curve of Class 10 is one such solution per point of
  the price grid.
- **The baseline must come from the model as well.** After a full NPL run $\hat P$ is
  already the model's fixed point at $\hat\theta$. After $K = 1$ or $2$ stages it is
  not, and comparing a model-generated counterfactual with a data-generated baseline
  mixes the effect of the policy with the misfit of the model.

What the CCP and NPL methods save, then, is the *repeated* solution of the model inside
the estimation loop, hundreds of fixed points in NFXP's outer iterations, not the
solution itself. The machinery of this class is reused for it: by
[Proposition 1](#npl-zero-jacobian) iterating $\Psi$ from $\hat P$ converges to the
counterfactual fixed point at the speed of Newton–Kantorovich.

The counterfactual also needs everything the CCP estimators sidestep. $\beta$ has to be
known, and the normalization of the flow utilities, harmless for estimation, is not
harmless for every counterfactual: see the
[identification discussion of Class 12](12_ccp.md#ccp-identification) and
{cite:t}`kalouptsidi2021IdentificationCounterfactualsDynamic`.

(npl-bus-example)=
### NFXP, two-step CCP and NPL on Zurcher's data

The three estimators on the bus data of Rust (groups 1–4, 175 mileage bins), with the
transition probabilities fixed at their frequency estimates. Both CCP-based estimators
start from the same first stage, a logit in Chebyshev polynomials of degree 3, and NPL
is shown at every step $K$ until the CCPs stop changing.

````{warning} Practical task: NFXP, two-step CCP and NPL in the code repository

(task-npl-practice)=
Run the comparison yourself: the notebook `npl_practice.ipynb` in the folder
`session14-npl/` of the code repository estimates the model by all three methods, on
the bus data and on panels simulated from the model. Update your copy first:

```bash
cd sb-dse-code && git pull
# first time: git clone https://github.com/fediskhakov/sb-dse-code.git
```

1. Run the notebook and reproduce the table below
2. Change the degree of the first-stage Chebyshev polynomial and see which estimates move
3. Compare the estimators on the simulated panels, where the true parameters are known
````

```{code-cell} python3
:tags: [hide-input]

import sys, pathlib, tempfile, urllib.request
import numpy as np

for folder, files in (('session10-11', ('zurcher.py', 'nfxp.py', 'busdata1234.csv')),
                      ('session14-npl', ('ccp.py',))):
    path = pathlib.Path('../gh-code') / folder        # the code of Classes 10-11 and of this class
    if not path.exists():                              # otherwise fetch it from the code repository
        path = pathlib.Path(tempfile.mkdtemp())
        url = f'https://raw.githubusercontent.com/fediskhakov/sb-dse-code/main/{folder}/'
        for f in files:
            urllib.request.urlretrieve(url + f, path / f)
    sys.path.insert(0, str(path))
import nfxp, ccp

def npl_regressors(model, pk):
    '''Policy valuation at pk = P(keep|x): v(x,keep) - v(x,replace) = h[x] @ (RC, c) + e[x]'''
    n, beta = model.n, model.beta
    Fk = model.trpr                                    # transitions after keep
    Fr = np.tile(model.trpr[0], (n, 1))                # after replace: row 0 for every x
    FU = Fr + pk[:, None] * (Fk - Fr)                  # unconditional transitions under pk
    Hk = np.column_stack([np.zeros(n), -0.001 * model.grid])   # u(x,keep)    = Hk @ (RC, c)
    Hr = np.column_stack([-np.ones(n), np.zeros(n)])           # u(x,replace) = Hr @ (RC, c)
    ent = -pk * np.log(pk) - (1 - pk) * np.log(1 - pk)  # sum_d P(d) e(d), gamma dropped
    rhs = np.column_stack([Hr + pk[:, None] * (Hk - Hr), ent])
    W = np.linalg.solve(np.eye(n) - beta * FU, rhs)    # policy valuation, linear in (RC, c)
    Z = np.column_stack([Hk - Hr, np.zeros(n)]) + beta * (Fk - Fr) @ W
    return Z[:, :2], Z[:, 2]

def npl_estimate(model, pk, tol=1e-8, maxiter=100, callback=None):
    '''NPL from the first-stage pk = P(keep|x); returns theta, se, loglik, pk at the fixed point'''
    keep = (model.d == 0).astype(float)
    for k in range(maxiter):
        h, e = npl_regressors(model, pk)
        theta, se, ll = ccp.logit_mle(h[model.x], keep, e[model.x])   # pseudo-likelihood step
        pk1 = ccp.logit_binary(h @ theta + e)                          # policy iteration step
        err = np.abs(pk1 - pk).max()
        if callback is not None:
            callback(K=k+1, theta=theta, se=se, loglik=ll, pk=pk1, err=err)
        if err < tol: break
        pk = pk1
    else:
        raise RuntimeError('Failed to converge in %d iterations' % maxiter)
    return theta, se, ll, pk1

data = nfxp.read_busdata()                             # bus groups 1-4, 175 bins
model = nfxp.estim_zurcher(data)                       # p from the transition frequencies
r = nfxp.estim_zurcher(data).estimate(full=False)      # NFXP partial likelihood, same p
ccp_theta, ccp_se, ccp_ll = ccp.ccp_second_stage(model, ccp.flexible_logit(model, 3))
rows = {'NFXP': (r.theta, r.se, r.stages[0]['loglik_choice'], None),
        'Two-step CCP, degree 3': (ccp_theta, ccp_se, ccp_ll, None)}
def add_kstage(K, theta, se, loglik, err, **kwargs):
    '''Callback: one row of the table per NPL step'''
    rows[f'NPL, K = {K}'] = (theta, se, loglik, err)
npl_estimate(model, ccp.flexible_logit(model, 3, keep=True), callback=add_kstage)

print(f'{"":<24}{"RC (s.e.)":>17}{"c (s.e.)":>17}{"loglik":>10}{"max dP":>10}')
for name, (t, s, ll, e) in rows.items():
    print(f'{name:<24}{f"{t[0]:.3f} ({s[0]:.3f})":>17}{f"{t[1]:.3f} ({s[1]:.3f})":>17}'
          f'{ll:10.2f}' + ('' if e is None else f'{e:10.1e}'))
```

The last column is the largest change in the CCPs between consecutive NPL steps.

- The NPL fixed point is the NFXP estimate, to the last digit of $RC$, $c$ and the
  log-likelihood, after 10 policy iterations, as
  [Proposition 3](#npl-zero-jacobian) says.
- One NPL step already lands close: $RC$ moves from 9.63 to 9.77 over the remaining
  steps, while the two-step CCP estimator of [Class 13](13_ccp_practice.md), started
  from the same first stage, puts $RC$ at 17.5. The difference is the representation of
  the future: $\varphi(\hat P)$ against the CCPs one period ahead.
- The standard error of $c$ is 5.4 for the two-step CCP estimator against 0.2–0.3
  for the others, the efficiency loss of a representation without the
  [zero Jacobian](#npl-asymptotics).
- The log-likelihood of the two-step CCP estimator is the highest, because nothing
  ties its first-stage CCPs to the estimated parameters, so it is not a better fit.
- NFXP and NPL report different standard errors for the same estimate: NFXP uses the
  outer product of the scores (BHHH), NPL the Hessian of the pseudo-likelihood, which
  at the fixed point is the information matrix. The Hessian-based standard errors of
  NFXP are close to those of NPL.

````{note} References and additional resources

(14_npl_references)=
- 📖 {cite:t}`aguirregabiriaSwappingNestedFixed2002` "Swapping the Nested Fixed Point Algorithm: A Class of Estimators for Discrete Markov Decision Models"
- 📖 {cite:t}`pesendorfer2010SequentialEstimationDynamic` "Sequential Estimation of Dynamic Discrete Games: A Comment"
- 📖 {cite:t}`kasahara2008PseudolikelihoodEstimationBootstrap` "Pseudo-likelihood estimation and bootstrap inference for structural discrete Markov decision models"
- 📖 {cite:t}`aguirregabiriaImposingEquilibriumRestrictions2021` "Imposing equilibrium restrictions in the estimation of dynamic discrete games"
- 📖 {cite:t}`aguirregabiriaDynamicDiscreteChoice2010` "Dynamic discrete choice structural models: A survey"
- 📺 Econometric Society Dynamic Structural Econometrics (DSE) lecture by Bertel Schjerning [YouTube video](https://youtu.be/houBb2vQFZE?si=VOIG544hnAiOx18x)
- Matlab implementation of the NPL estimator for the bus model [DSE 2019 GitHub repo](https://github.com/dseconf/DSE2019/tree/master/02_DDC_SchjerningIskhakov)

````
