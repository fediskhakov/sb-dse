---
title: 📖 Nested pseudo-likelihood (NPL)
short_title: 📖 NPL
subtitle: Class 13 — Tuesday, October 6
exports:
  - format: typst
    output: exports/13_npl.pdf
downloads:
  - file: 13_npl.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

The two-step estimator of Class 11 trades efficiency for never solving the model, and
NFXP trades the other way. {cite:t}`aguirregabiriaSwappingNestedFixed2002` put the two
on one line: rewrite the Bellman equation as a fixed point in the space of choice
probabilities, and iterate on it only as many times as the data deserve.

## From value space to probability space

Like all CCP-based methods, pseudo-likelihood estimation starts with the consistent
estimation of the CCPs and transition probabilities from the data. Let's focus on the
specification of the statistical criterion for the second stage. Return to the
representation of the integrated value function through the correction terms,

$$
V_\sigma(x) =
\sum_{d \in D(x)} P(d|x)
\big(
v(x,d) + e(x,d)
\big)
$$

Using the definition of the choice-specific value functions
$v(x,d) = u(x,d) + \beta \int_X V_\sigma(x') \pi(x'|x,d) dx'$ we have

$$
V_\sigma(x) =
\sum_{d \in D(x)} P(d|x)
\left(
u(x,d) + \beta
\int_{X} V_\sigma(x')
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
\int_{X} V_\sigma(x')
\pi(x'|x,d) dx'
$$

which is linear in $V_\sigma$ once the CCPs are held fixed. Again assuming a discrete
state space and stacking $V_\sigma(x)$, $P(d|x)$, $u(x,d)$ and $e(x,d)$ across the
state space to form column vectors, we can write the last equation for every point of
the state space in matrix form,

$$
V_\sigma =
\sum_{d \in D(x)} P(d) \ast
\left[
u(d) + e(d)
\right]
+ \beta
\sum_{d \in D(x)} P(d) \ast
\Pi(d) V_\sigma
$$

where $\ast$ is the element-wise (Hadamard) product of vectors. Let

$$
\Pi = \sum_{d \in D(x)} P(d) \ast \Pi(d)
$$

be the $|X|\times|X|$ *unconditional* transition matrix, the same Markov chain whose
stationary distribution was computed in the homework of Class 9, and denote by $I$
again the identity matrix of the same size. Then the linear system solves as

$$
V_\sigma = [I - \beta \Pi]^{-1}
\sum_{d \in D(x)} P(d) \ast
\left[
u(d) + e(d)
\right]
$$

This is policy evaluation from Class 8, with the policy given by the CCPs rather than
by a deterministic decision rule. Recall that the correction term $e(d)$ is a function
of the CCPs only, and therefore we can introduce an operator

$$
\varphi : [0,1]^{|X|\times|D|} \ni P \mapsto V_\sigma \in \mathbb{R}^{|X|}
$$

that maps the CCPs into the integrated value functions. Finally, denote by

$$
\Lambda: \mathbb{R}^{|X|} \ni V_\sigma \mapsto \{P(d|x)\} \in \mathbb{R}^{|X|\times|D|}
$$

the mapping from the integrated value functions to the CCPs given by the choice
probability formulas of Class 11, which in the simple EV1 case take the form of the
multinomial logit

$$
\Lambda(V_\sigma) =
\left\{\frac{\exp[u(x,d) + \beta \Pi(d) V_\sigma]}{\sum_{d'\in D(x)} \exp[u(x,d') + \beta \Pi(d') V_\sigma]}
\right\}_{d \in D(x), x \in X}
$$

````{attention} Definition

Following {cite:t}`aguirregabiriaSequentialEstimationDynamic2007` we define the
**policy iteration operator** as the composition

$$
\Psi = \Lambda \circ \varphi : [0,1]^{|X|\times|D|} \ni P \mapsto \Lambda(\varphi(P)) \in [0,1]^{|X|\times|D|}
$$

which maps the CCPs into the CCPs through the integrated value functions. The optimal
choice probabilities of the model are a fixed point of $\Psi$,

$$
P = \Psi(P)
$$
````

Note that $\Psi$ depends on the structural parameters of the model $\theta$ through
$u(x,d)$ and $\beta$, so we write $\Psi(P,\theta)$ when the dependence matters. The
original fixed point problem in *value space* is thereby reformulated as a fixed point
problem in *probability space*. One application of $\Psi$ is one policy evaluation, a
linear solve, followed by one policy improvement, a logit formula, so successive
approximations $P_{k+1} = \Psi(P_k,\theta)$ converge as fast as policy iterations do.

:::{div}
:class: discussion

- $\varphi$ needs $e(d)$, and $e(d)$ needs $\log P(d|x)$. What does $\Psi$ do with a
  CCP vector that has a zero in it?
- $\Psi$ inverts an $|X|\times|X|$ matrix. Compare its cost to one Newton–Kantorovich
  step of Class 9. Is that a coincidence?
:::

## Pseudo-maximum likelihood estimation

The operator $\Psi$ turns any first-stage CCP estimate into a set of model-implied
choice probabilities, and those can be put into a likelihood. The **pseudo-likelihood
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
iteration only, starting from the data. The resulting algorithm is a two-step
estimator in the sense of Class 11, with a particular choice of criterion:

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
fixed regressor, and BHHH from Class 10 applies unchanged.

Where it breaks: exactly where the two-step estimator breaks. The criterion is a
likelihood only in name, since the probabilities are computed at $\hat{P}$ and not at
the model's own fixed point, and the first-stage noise passes through $\varphi$ into
every term. {cite:t}`aguirregabiriaSwappingNestedFixed2002` show that
$\hat{\theta}_{PML}$ is $\sqrt{n}$-consistent and asymptotically normal, but it is not
asymptotically efficient, and in finite samples it can be far from the MLE.

## Swapping the nested fixed point

📖 {cite:t}`aguirregabiriaSwappingNestedFixed2002` "Swapping the Nested Fixed Point
Algorithm: A Class of Estimators for Discrete Markov Decision Models"

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
the NPL estimator. The pair $(\hat{\theta},\hat{P})$ at convergence satisfies both
$\hat{P} = \Psi(\hat{P},\hat{\theta})$, so $\hat{P}$ is the solution of the model at
$\hat{\theta}$, and the first order condition of the pseudo-likelihood, which at the
fixed point coincides with the likelihood of NFXP.

### The sequence of estimators

Stopping the iteration after $K$ steps gives a whole class of estimators, which
{cite:t}`aguirregabiriaSwappingNestedFixed2002` call the **$K$-stage policy iteration
estimators**. They bridge the gap between the two-step CCP estimator and NFXP:

- $K=1$ is the pseudo-maximum likelihood estimator above, and with a frequency
  first stage it coincides with the Hotz–Miller estimator of Class 11
- every $K$-stage estimator is $\sqrt{n}$-consistent and asymptotically normal, with a
  known variance-covariance matrix, so standard inference applies at any $K$
- for $K \geqslant 2$ the estimators are asymptotically equivalent to the MLE, and the
  sequence converges to the NFXP maximum likelihood estimator as $K \rightarrow \infty$
- each iteration is computationally cheap: one linear solve per likelihood evaluation,
  and the size of the state space does not affect the number of policy iterations

The trade-off is between finite sample precision and computational cost. In the Monte
Carlo of {cite:t}`aguirregabiriaSwappingNestedFixed2002` on Rust's bus model the
first stage is noisy, the second stage removes most of the finite sample bias, and the
full NPL fixed point is reached in a handful of iterations at 5 to 10 times the speed of
NFXP, though NFXP in that comparison ran without the poly-algorithm of Class 9.

Where it breaks: the NPL iterations are successive approximations on
$(\theta,P) \mapsto (\hat\theta(P), \Psi(P,\hat\theta(P)))$, and nothing guarantees
that this map is a contraction. In single-agent models it usually is, since $\Psi$
alone is a policy iteration step, but the parameter update can push it out of the
contraction region when the first stage is poor. In dynamic games $\Psi$ is not a
contraction in general, the NPL fixed point need not be unique, and the iterations can
converge to a fixed point that is not the MLE, or fail to converge at all, see
{cite:t}`pesendorfer2010SequentialEstimationDynamic`. This is the reason the games
part of the course returns to the question of which fixed point an estimator finds.

:::{div}
:class: discussion

- NPL at convergence solves the same fixed point as NFXP. Where does the computational
  saving come from, then?
- Why does the asymptotic distribution not depend on $K$ for $K \geqslant 2$, while the
  finite sample distribution does?
- The NPL fixed point is $\hat P = \Psi(\hat P,\hat\theta)$. If $\Psi$ has several fixed
  points at $\hat\theta$, which one does the data pick?
:::

(task13.1)=
````{danger} Homework: pseudo-likelihood and NPL on the bus engine model

This is a graded homework assignment.

The notebook `tasks/theta_npl/` in the class repository carries the model class, the
simulator and the NFXP estimator of Class 10 and the steps to take:

```bash
git pull upstream main                              # collect the task
cp -r tasks/theta_npl solutions/theta_npl           # work on the copy
```

1. Implement the operator $\Psi(P,\theta)$ for the Zurcher model: the correction terms
   $e(d)$, the policy evaluation $\varphi(P)$ as a linear solve, and the logit
   $\Lambda$. Verify that the model's own choice probabilities from the solver are a
   fixed point of $\Psi$ at the true parameters.
2. Write the pseudo-likelihood $\ell_n(\theta)$ for $\theta = (RC,\theta_1)$ given
   frequency-estimated CCPs on simulated data, and maximize it. Compare the estimate
   and its standard errors to NFXP on the same sample.
3. Implement the NPL iterations and record $\hat\theta_K$ and
   $\|\hat P_K - \hat P_{K-1}\|$ for $K = 1,2,\dots$ until convergence. Plot the path of
   the estimates against the NFXP estimate, and count solver-equivalent operations for
   both methods.
4. Repeat 2–3 on a panel ten times smaller. Report which $K$ removes most of the
   finite sample bias, and whether the iterations still converge.

Remember to follow the git workflow <https://dse.iskh.me/workflow/#submission> to submit
your solution.
````

(13_npl_references)=
````{note} References and additional resources

- 📖 {cite:t}`aguirregabiriaSwappingNestedFixed2002` "Swapping the Nested Fixed Point Algorithm: A Class of Estimators for Discrete Markov Decision Models"
- 📖 {cite:t}`aguirregabiriaSequentialEstimationDynamic2007` "Sequential Estimation of Dynamic Discrete Games"
- 📖 {cite:t}`hotz1993ConditionalChoiceProbabilitiesb` "Conditional Choice Probabilities and the Estimation of Dynamic Models"
- 📖 {cite:t}`pesendorfer2010SequentialEstimationDynamic` "Sequential Estimation of Dynamic Discrete Games: A Comment"
- 📖 {cite:t}`aguirregabiriaDynamicDiscreteChoice2010` "Dynamic discrete choice structural models: A survey"
- 📺 Econometric Society Dynamic Structural Econometrics (DSE) lecture by Bertel Schjerning [YouTube video](https://youtu.be/houBb2vQFZE?si=VOIG544hnAiOx18x)
- Matlab implementation of the NPL estimator for the bus model [DSE 2019 GitHub repo](https://github.com/dseconf/DSE2019/tree/master/02_DDC_SchjerningIskhakov)

````
