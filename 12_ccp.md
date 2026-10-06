---
title: 📖 Conditional choice probabilities, identification and two-step estimator
short_title: 📖 CCP estimation
subtitle: Class 12 — Thursday, October 1
downloads:
  - file: 12_ccp.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

NFXP solves the model inside every likelihood evaluation. This class turns the
argument around: the conditional choice probabilities (CCPs) observed in the data pin
down the value function differences, so the model can be estimated without ever
solving it. The same result of {cite:t}`hotz1993ConditionalChoiceProbabilitiesb` also
says exactly what the data can and cannot identify.


````{seealso} Key reading

{cite:t}`hotz1993ConditionalChoiceProbabilitiesb` "Conditional Choice Probabilities
and the Estimation of Dynamic Models", *Review of Economic Studies* 60(3), pp. 497–529
The inversion theorem that started it all, the early CCP estimator and application. Notation that been standardized later..

{cite:t}`arcidiaconoConditionalChoiceProbability2011` "Conditional Choice Probability Estimation of Dynamic Discrete Choice Models With Unobserved Heterogeneity", *Econometrica* 79(6), pp. 1823–1867.
Further CCP theory, modern notation, notion of finite dependence, CCP estimator for models with unobserved heterogeneity.

````

````{hint} Running the code for this lecture
:class: dropdown

Every code example below is also a runnable notebook in the course **code repository**,
in the folder `session12-oct1/`.

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

````{danger} Homework eta_ccp: recovering value differences from choice data
:class: dropdown

(homework-eta_ccp)=
Graded homework: task `eta_ccp` in the class repository. Collect it and work on a
copy in your own repository:

```bash
git pull upstream main                 # collect the task
cp -r tasks/eta_ccp solutions/eta_ccp  # work on the copy
```

Derive the Bellman equation of the Zurcher model in the space of integrated value
functions and implement it as a subclass `zurcher_ccp` of the model class, checking that
it is a contraction. Then verify numerically, at the NFXP estimates and a handful of
other parameter values, the Hotz–Miller inversion and the Arcidiacono–Miller
relationship between the integrated value function, the choice-specific values and the
choice probabilities.

The notebook `ccp_representations.ipynb` in the task folder loads the model class, the
NFXP estimator and Zurcher's data from `session10-11/` of the code repository, and has
the details of the task, step by step. The code of [Class 10](10_nfxp.md), `session10-11/` in the code
repository, is worth having beside you while you work.

Submit it as a pull request, following the [git workflow](2_workflow.md#submission). One of you presents a solution at the
start of the next Tuesday class.
````

## Integrated $\ne$ expected value function

Maintain the setup of the Zurcher engine replacement model of
{cite:t}`rustOptimalReplacementGMC1987` from [Class 9](9_zurcher.md), leaning to more generality. The
Bellman equation of the Zurcher problem is

$$V(x,\varepsilon) = \max_{d\in D(x)} \Big\{ \underbrace{u(x,d) + \beta
\int_{X} \Big( \int_{\Omega} V(x',\varepsilon') q(\varepsilon'|x') d\varepsilon'\Big)
\pi(x'|x,d) dx'}_{v(x,d)}
+ \varepsilon_d \Big\}$$

and it breaks into three pieces that are useful on their own: the choice-specific
value, the expected value, and the value function itself,

$$
V(x,\varepsilon) = \max_{d\in D(x)} \big\{ v(x,d) + \varepsilon_{d} \big\},
\qquad
v(x,d) = u(x,d) + \beta EV(x,d),
$$

$$
EV(x,d) = \int_{X} \log \big( \exp[v(x',0)] + \exp[v(x',1)] \big) \pi(x'|x,d) dx'
$$

The last equation uses the EV1 assumption on the distribution of $\varepsilon$, but in
more generality it can be written as

$$
EV(x,d) = \int_{X} \int_{\Omega}
V(x',\varepsilon')
q(\varepsilon'|x') \pi(x'|x,d) d\varepsilon' dx'
$$

Let's repeat the terminology:

- $v(x,d)$ is the **choice-specific value function**, or
conditional value function, and 
- $EV(x,d)$ is commonly referred to as the **expected
value function** and sometimes as the ex-post value function

Here is a new object and yet another representation. The **integrated value function**
$V^\sigma(x)$, also known as the ex-ante value function, is the value function with the
taste shock integrated out,

$$
V^\sigma(x) = \int_{\Omega} V(x,\varepsilon) q(\varepsilon|x) d\varepsilon,
\qquad
EV(x,d) = \int_{X} V^\sigma(x') \pi(x'|x,d) dx'
$$

The subscript $\sigma$ follows the notation of
{cite:t}`aguirregabiriaSequentialEstimationDynamic2007`, who define it as the value
function under a strategy profile $\sigma$.

```{figure} _static/img/bellman_circle.png
:width: 80%
:align: center

The Bellman circle of value functions: plain value function $V(x,\varepsilon)$, integrated value function $V^\sigma(x)$, expected value function $EV(x,d)$ and choice-specific value function $v(x,d)$
```

We could write the Bellman equation in the space of integrated value functions
$V^\sigma(x)$ by cutting the *circle of Bellman* at a different point:

$$
V(x,\varepsilon) \rightarrow V^\sigma(x) \rightarrow EV(x,d) \rightarrow v(x,d) \rightarrow V(x,\varepsilon) \rightarrow \dots
$$

The new representation of the Bellman equation is

$$
V^\sigma(x) = 
\int_{\Omega} \max_{d'\in D(x)} \Big\{ \underbrace{u(x,d') + \beta
\int_{X} V^\sigma(x')
\pi(x'|x,d') dx'}_{v(x,d')}
+ \varepsilon_{d'} \Big\} q(\varepsilon|x) d\varepsilon
=
$$
$$
=\int_{\Omega} \max_{d'\in D(x)} \left\{ v(x,d')
+ \varepsilon_{d'} \right\} q(\varepsilon|x) d\varepsilon
$$

This is the expectation of the maximum utility in a RUM with alternative utilities
given by $v(x,d') + \varepsilon_{d'}$, which {cite:t}`mcfadden1974ConditionalLogitAnalysisa`
called the **social surplus function** in [Class 6](6_logit.md#wdz-theorem). 

In the EV1 case max-stability gives
it in closed form, with $\gamma \approx 0.5772$ the Euler–Mascheroni constant,

$$
V^\sigma(x) = \log \big( \sum_{d' \in D(x)} \exp[v(x,d')] \big) + \gamma
$$

(ccp-choice-probabilities)=
### Choice probabilities

Recall that by the [Williams–Daly–Zachary theorem](6_logit.md#wdz-theorem) in the
general case the choice probabilities are the derivatives of the social surplus
function,

$$
P(d|x) = \frac{\partial}{\partial v(x,d)}
\mathbb{E}\left\{\max_{d' \in D(x)} \big[ v(d',x) +\varepsilon(d') \big]\Big|x\right\}
=
\frac{\partial V^\sigma(x)}{\partial v(x,d)}
$$

and under the EV1 assumption the derivative of the logsum is the multinomial logit,

$$
P(d|x) =
\frac{\partial \log \big( \sum_{d' \in D(x)} \exp[v(x,d')] \big)}{\partial v(x,d)}
=
\frac{\exp[v(x,d)]}{\sum_{d'\in D(x)} \exp[v(x,d')]}
$$

Exchanging the order of differentiation and integration in the general case, we can also 
express the choice probabilities as the integral with
respect to the distribution $q(\varepsilon|x)$ of the indicator that $d$ is the
maximizer

$$
P(d|x)=
\int_\Omega I\left\{ d = \arg\max_{d' \in D(x)} \{v(x,d')+\varepsilon_{d'}\}
\right\}q(\varepsilon|x) d\varepsilon
$$


## Inversion theorem

Let $d_0$ denote the reference alternative, so that the values of other alternatives
will be measured relative to it. 

For each $x$ consider the vector of value differences
$\Delta v(x) \in \mathbb{R}^{K-1}$, where $K=|D(x)|$ is the number of alternatives,
with elements

$$
\Delta v(x,d) = v(x,d) - v(x,d_0), \quad d \in D(x)\setminus\{d_0\}
$$

Rewriting the event $\left\{ d = \arg\max_{d' \in D(x)} \{v(x,d')+\varepsilon_{d'}\}
\right\}$ from above step by step, we have

$$
P(d|x)=
\int_\Omega I\left\{ d = \arg\max_{d' \in D(x)} \{v(x,d')+\varepsilon_{d'}\}
\right\}q(\varepsilon|x) d\varepsilon =
$$

$$
=
\int_\Omega I\left\{
\varepsilon_{d'} \leqslant v(x,d) - v(x,d') + \varepsilon_{d}, \forall d' \in D(x)\setminus\{d\}
\right\}q(\varepsilon|x) d\varepsilon
=
$$

$$
=
\int_\Omega I\left\{
\varepsilon_{d'} \leqslant \Delta v(x,d) - \Delta v(x,d') + \varepsilon_{d}, \forall d' \in D(x)\setminus\{d\}
\right\}q(\varepsilon|x) d\varepsilon
=
$$

$$
=
\int_{\{\varepsilon:\; \varepsilon_{d'} \leqslant \Delta v(x,d) - \Delta v(x,d') + \varepsilon_{d}, \forall d' \in D(x)\setminus\{d\}\}}
q(\varepsilon|x) d\varepsilon
=
Q_{d}(\Delta v(x),x)
$$

The steps are: 
- write the argmax as a system of inequalities, 
- subtract $v(x,d_0)$ from both sides of each inequality so that only differences appear, and 
- read the result as the probability mass of a region of $\Omega$ that depends on $\Delta v(x)$ only

Compare this derivation to the
[static multinomial logit model](6_logit.md#wdz-theorem) of Class 6.

In other words, if

$$
Q: \mathbb{R}^{K-1} \ni \delta \mapsto P \in [0,1]^{K}
$$

is a mapping from $K-1$ value differences to the $K$ vector of probabilities, then the
choice probabilities are given by $P(x) = Q(\Delta v(x),x)$.

````{attention} Definition

**Inversion theorem.** Under certain regularity conditions on $q(\varepsilon|x)$
the mapping $Q$ is invertible, i.e. there exists a mapping
$Q^{-1}: [0,1]^{K} \ni P \mapsto \delta \in \mathbb{R}^{K-1}$ such that

$$
\Delta v(x) = Q^{-1}(P(x),x)
$$
````

The theorem is the reason the reference alternative was needed: $K$ probabilities that
sum to one carry $K-1$ degrees of freedom, and so do the $K-1$ differences. Levels of
the choice-specific values are not recoverable from choices, only their differences.

````{tip} Example: inversion in the multinomial logit case

In the multinomial logit case for some $d_0 \in D(x)$

$$
P(d|x) = \frac{\exp[v(x,d)]}{\sum_{d'\in D(x)} \exp[v(x,d')]}
=
\frac{\exp[\Delta v(x,d)]}{1+\sum_{d'\in D(x)\setminus\{d_0\}} \exp[\Delta v(x,d')]},
$$

$$
P(d_0|x) = \frac{1}{1+\sum_{d'\in D(x)\setminus\{d_0\}} \exp[\Delta v(x,d')]}
$$

The inverse map is given by the log odds ratio

$$
\frac{P(d|x)}{P(d_0|x)} = \exp[\Delta v(x,d)] \implies
$$
$$
\Delta v(x,d) = \log P(d|x) - \log P(d_0|x), \quad d \in D(x)\setminus\{d_0\}
$$
````

What we have: the Hotz–Miller inversion gives a mapping from the choice probabilities
to the value differences for any choice model with random terms $\varepsilon$
satisfying the regularity conditions of the theorem, and in the EV1 and GEV cases we
have closed-form expressions for the inverse map.

:::{div}
:class: discussion

- Why is a reference alternative needed, and does the choice of $d_0$ matter?
- What happens to the inverse map when an estimated $P(d|x)$ is exactly zero at some
  $x$, and how likely is that with frequency estimates on the bus data?
:::

## Relationship between $V^\sigma(x)$, $v(x,d)$ and choice probabilities

Recall that the choice probability $P(d|x)$ in the general case is simply the
expectation of the indicator that a particular choice $d$ yields the maximum utility,

$$
P(d|x)= \hbox{Prob}\left\{d = \arg\max_{d' \in D(x)} \{v(x,d')+\varepsilon_{d'}\}|x\right\}
=
$$
$$
=
\mathbb{E}\left[\left. I\left\{ d = \arg\max_{d' \in D(x)} \{v(x,d')+\varepsilon_{d'}\} \right\} \right|x \right]
$$

The expression for $V^\sigma(x)$ as the expected maximum can then be expanded using
the *law of iterated expectations*, conditioning on which alternative is chosen:

$$
V^\sigma(x) 
= 
\mathbb{E}\left[
\max_{d'\in D(x)} \left\{ v(x,d')
+ \varepsilon_{d'} \right\}
\right] 
=
$$
$$
=
\sum_{d \in D(x)} P(d|x)
\mathbb{E}\left[\left.
\max_{d'\in D(x)} \left\{ v(x,d')
+ \varepsilon_{d'} \right\}
\right|
d = \arg\max_{d''\in D(x)} \{v(x,d'')+\varepsilon_{d''}\}
\right]
=
$$

$$
=
\sum_{d \in D(x)} P(d|x)
\mathbb{E}\left\{
v(x,d) + \varepsilon_{d}
\left|
d = \arg\max_{d''\in D(x)} \{v(x,d'')+\varepsilon_{d''}\}
\right.\right\}
=
$$

$$
=
\sum_{d \in D(x)} P(d|x)\left[
v(x,d) +
\mathbb{E}\left\{
\varepsilon_{d}
\left|
d = \arg\max_{d''\in D(x)} \{v(x,d'')+\varepsilon_{d''}\}
\right.\right\}\right]
$$

Conditional on $d$ being the maximizer, the maximum *is* $v(x,d)+\varepsilon_d$, and
$v(x,d)$ is not random, which is all that happens between the second and the last line.
The term that remains,

$$
e(x,d) = \mathbb{E}\left\{
\varepsilon_{d}
\left|
d = \arg\max_{d''\in D(x)} \{v(x,d'')+\varepsilon_{d''}\}
\right.\right\}
$$

has the special name of **correction term**, see
{cite:t}`aguirregabiriaSwappingNestedFixed2002`.
It is the expectation of the
random component conditional on the event that the alternative it is associated with
has the highest value. We end up with

$$
V^\sigma(x) =
\sum_{d \in D(x)} P(d|x)
\big(
v(x,d) + e(x,d)
\big)
$$

We can make one more step, using the Hotz–Miller inversion with a reference alternative
$d_0$, to arrive at

$$
V^\sigma(x) - v(x,d_0) =
\sum_{d \in D(x)} P(d|x)
\big(
\Delta v(x,d) + e(x,d)
\big) =
$$

$$
=
\sum_{d \in D(x)\setminus \{d_0\}} P(d|x)
Q^{-1}_d(P(x),x) +
\sum_{d \in D(x)} P(d|x)
e(x,d)
=
\psi(x,d_0)
$$

where $\psi(x,d_0)$ *is a function of the choice probabilities and not of the value
functions*, see Lemma 1 in {cite:t}`arcidiaconoConditionalChoiceProbability2011`. 
This result holds can be written for any reference alternative $d_0$, and therefore holds for every $d$.

In other words, the difference between the integrated value function and any choice-specific
reference value is a function of the choice probabilities only. This is the result
that every CCP-based method builds on.

### Correction terms for EV1 and GEV

{cite:t}`hotz1993ConditionalChoiceProbabilitiesb` and 
{cite:t}`arcidiaconoConditionalChoiceProbability2011` show that the correction term
has a closed form in the two workhorse cases,

$$
\text{EV1} \;\implies\;e(x,d) = \gamma-\log P(d|x)
$$

$$
\text{GEV} \;\implies\;e(x,d) = \gamma - \sigma\log P(d| x) -(1-\sigma)\log\sum_{d' \in N(d)} P(d'|x)
$$

where $N(d)$ is the set of alternatives in the same *nest* as $d$ and
$\sigma\leqslant 1$ is the scale parameter within the nest.


(finite-dependence)=
## Finite dependence

Hotz-Miller inversion establishes the relationship between the choice probabilities and the value differences, but values are still defined recursively.

Main idea of finite dependence is to note that it may be possible to *cancel out the future* after several periods

- recursion reduces to several periods (one repeated cycle)
- then no fixed point to solve for
- estimator can be applied to short panels

The idea goes back to
{cite:t}`altug1998EffectWorkExperience`, and we follow the general version in Section 3
of {cite:t}`arcidiaconoConditionalChoiceProbability2011`.

The starting point is the expression derived above

$$
V^\sigma(x) = v(x,d) + \psi(x,d), \qquad d \in D(x)
$$

Substituting it into the definition of the choice-specific value gives

$$
v(x,d) = u(x,d) + \beta \sum_{x' \in X} \big[ v(x',k) + \psi(x',k) \big] \pi(x'|x,d),
\qquad k \in D(x')
$$

where $k \in D(x')$ is just some alternative chosen in the next period.

- It does not have to be the same in every $x'$. 

- It does not even have to be a single alternative either, and can be represented by some 
*choice weights* $\{\omega_k\}$, such that $\sum_{k\in D(x')} \omega_k = 1$:

$$
V^\sigma(x') = 
\sum_{k\in D(x)} \omega_k V^\sigma(x') =
\sum_{k\in D(x)} \omega_k \big[ v(x',k) + \psi(x',k) \big]
$$

Let's now repeat this step again in the next period, and so on. This is referred to as *telescoping* the integrated value function.

1. Fix the current period $t$, the state $x_t = x$ and the initial choice
$d$. 

2. A **choice sequence** assigns to every later period $\tau > t$ and every state
$x_\tau$ a vector of **decision weights** $\omega_\tau(k|x_\tau,d) \geqslant 0$ with
$\sum_{k \in D(x_\tau)} \omega_\tau(k|x_\tau,d) = 1$. 
- The weights may depend on the state reached and may mix over alternatives
- {cite:t}`arcidiaconoConditionalChoiceProbability2011` write them as $d^*_{k\tau}(z_\tau,j)$

Along the sequence, the state evolves according to the distribution

$\kappa_\tau(\cdot|x,d)$ of $x_{\tau+1}$, defined recursively by

$$
\kappa_\tau(x'|x,d) =
\begin{cases}
\pi(x'|x,d), & \tau = t, \\
\sum_{x_\tau \in X} \sum_{k \in D(x_\tau)} \omega_\tau(k|x_\tau,d)\, \pi(x'|x_\tau,k)\,
\kappa_{\tau-1}(x_\tau|x,d), & \tau > t.
\end{cases}
$$

In other words, we look at the distribution of states resulting from the chosen sequence of choice weights

$$
\begin{matrix}
0 & \omega_{t+1}(1|x_{t+1},d) & \omega_{t+2}(1|x_{t+2},d) & \omega_{t+3}(1|x_{t+3},d) & \cdots \\
1 & \omega_{t+1}(2|x_{t+1},d) & \omega_{t+2}(2|x_{t+2},d) & \omega_{t+3}(2|x_{t+3},d) & \cdots \\
\vdots & \vdots & \vdots & \\
0 & \omega_{t+1}(K|x_{t+1},d) & \omega_{t+2}(K|x_{t+2},d) & \omega_{t+3}(K|x_{t+3},d) & \cdots
\end{matrix}
$$

or in the simple case of a deterministic sequence of choices $d_\tau$

$$
\begin{matrix}
d & k_{t+1}(x_{t+1},d) & k_{t+2}(x_{t+2},d) & k_{t+3}(x_{t+3},d) & \cdots
\end{matrix}
$$

In both cases the induced distribution of states is

$$
\begin{matrix}
\pi(1|x,d) & \kappa_{t+1}(1|x,d) & \kappa_{t+2}(1|x,d) & \kappa_{t+3}(1|x,d) & \cdots \\
\pi(2|x,d) & \kappa_{t+1}(2|x,d) & \kappa_{t+2}(2|x,d) & \kappa_{t+3}(2|x,d) & \cdots \\
\vdots & \vdots & \vdots & \\
\pi(N|x,d) & \kappa_{t+1}(N|x,d) & \kappa_{t+2}(N|x,d) & \kappa_{t+3}(N|x,d) & \cdots
\end{matrix}
$$

Now we can compute the choice specific value *along some choice sequence*!

Theorem 1 of {cite:t}`arcidiaconoConditionalChoiceProbability2011` shows that, for any
state $x$, any initial choice $d$ and *any* choice sequence,

$$
v(x,d) = u(x,d) + \sum_{\tau=t+1}^{\infty} \beta^{\tau-t}
\sum_{x_\tau \in X} \sum_{k \in D(x_\tau)}
\big[ u(x_\tau,k) + \psi(x_\tau,k) \big]\,
\omega_\tau(k|x_\tau,d)\, \kappa_{\tau-1}(x_\tau|x,d)
$$


The term $\psi(x_\tau,k) = V^\sigma(x_\tau) - v(x_\tau,k)$ compensates at each step for the
sequence not being the optimal policy. 


````{attention} Definition

**$\rho$-period finite dependence.** A pair of alternatives $d, d' \in D(x)$ exhibits
**$\rho$-period finite dependence** at state $x$ if there exist a choice sequence
starting with $d$ and a choice sequence starting with $d'$ such that

$$
\kappa_{t+\rho}(x''|x,d) = \kappa_{t+\rho}(x''|x,d') \quad \text{for all } x'' \in X,
$$

that is, the two sequences lead to the same distribution of the state in period
$t+\rho+1$.
````

Why is this useful?

From period $t+\rho+1$ on, set the weights of the two sequences equal.

Every term of Theorem 1 after $t+\rho$ is then identical for $d$ and $d'$ and cancels in the
difference, leaving

$$
\begin{aligned}
v(x,d) - v(x,d') = & u(x,d) - u(x,d') \\
& + \sum_{\tau=t+1}^{t+\rho} \beta^{\tau-t} \sum_{x_\tau \in X} \sum_{k \in D(x_\tau)}
\big[ u(x_\tau,k) + \psi(x_\tau,k) \big]
\Delta_\tau(k,x,x_\tau,d,d')
\end{aligned}
$$
$$
\Delta_\tau(k,x,x_\tau,d,d') = \omega_\tau(k|x_\tau,d)\, \kappa_{\tau-1}(x_\tau|x,d)
- \omega_\tau(k|x_\tau,d')\, \kappa_{\tau-1}(x_\tau|x,d')
$$

````{hint}
What is observed and not observed in this expression?

- $v(x,d) - v(x,d')$ can be recovered from the CCPs by Hotz-Miller inversion
- $\psi(x_\tau,k)$ only depend on CCPs
- $\omega_\tau(k|x_\tau,d)$ and $\omega_\tau(k|x_\tau,d')$ are chosen by the researcher
- $\kappa_{\tau-1}(x_\tau|x,d)$ and $\kappa_{\tau-1}(x_\tau|x,d')$ are computed recursively, assuming $\pi(x'|x,d)$ are known/estimated
- $u(x,d) - u(x,d')$, $u(x_\tau,k)$ contain the structural parameters of interest, 
- $\beta$ is another structural parameter

We have enough information to form GMM-type moment conditions entangling the structural parameters and data.

The data required are states and choices for $\rho+1$ periods, including short panels if the model admits low period finite dependence.

````

Helpful properties of finite dependence:

- the $\psi$ mappings, the first-stage CCPs and the transition probabilities, the
value difference is a linear function of flow utilities over $\rho+1$ periods.
- only the $\psi$ functions of alternatives that carry positive weight in one of the two sequences
are needed

(renewal-terminal)=
````{tip} Example: renewal actions and terminal choices

Let a **renewal action** $r \in D(x)$ be an alternative that, taken at $t+1$, makes the distribution of the
state at $t+2$ independent of the choice at $t$:

$$
\sum_{x' \in X} \pi(x''|x',r)\, \pi(x'|x,d)
=
\sum_{x' \in X} \pi(x''|x',r)\, \pi(x'|x,d')
\quad \text{for all } x'', d, d'
$$

The state at $t+2$ may still depend on the state at $t+1$, but only through variables
that the choice at $t$ did not affect. Put weight one on $r$ at $t+1$ in both sequences.
In terms of the general formula:

- **Sequences.** $(d, r, \dots)$ and $(d', r, \dots)$, that is
  $\omega_{t+1}(k|x',d) = \omega_{t+1}(k|x',d') = I\{k=r\}$ for every $x' \in X$.
- **State distributions.** $\kappa_t(x'|x,d) = \pi(x'|x,d)$ at $t+1$, and
  $\kappa_{t+1}(x''|x,d) = \sum_{x' \in X} \pi(x''|x',r)\, \pi(x'|x,d)$ at $t+2$. The
  renewal condition above says exactly that $\kappa_{t+1}(\cdot|x,d) = \kappa_{t+1}(\cdot|x,d')$.
- **Dependence length.** $\rho = 1$, so the sum over $\tau$ keeps only $\tau = t+1$.
- **Weights from $t+2$ on.** Any common choice, for example the CCPs themselves. They
  never have to be specified, because their terms cancel.
- **Weight differences.** $\Delta_{t+1}(k,x,x',d,d') = I\{k=r\}
  \big[ \pi(x'|x,d) - \pi(x'|x,d') \big]$, so the sum over $k$ keeps only $k = r$.

Substituting, the difference collapses to

$$
\begin{aligned}
v(x,d) - v(x,d') = & u(x,d) - u(x,d') +\\ &
+ \beta \sum_{x' \in X} \big[ u(x',r) + \psi(x',r) \big]
\big[ \pi(x'|x,d) - \pi(x'|x,d') \big]
\end{aligned}
$$

Only the mapping $\psi(\cdot,r)$ of the renewal action is needed. A **terminal
choice**, such as exit or retirement, works the same way
{cite:p}`hotz1993ConditionalChoiceProbabilitiesb`. No choices follow it, so its
continuation value folds into its flow payoff, and the same expression holds for any
two non-terminal alternatives $d, d'$ with $r$ the terminal choice. In the general
formula, after $r$ the state moves to an absorbing state with zero payoffs, so
$\kappa_{t+1}(\cdot|x,d) = \kappa_{t+1}(\cdot|x,d')$ is a point mass on it, $\rho = 1$,
and $u(x',r)$ is the lifetime value of exiting at $x'$.
````

The Zurcher model is the leading case: engine replacement is a renewal action, so the
model has one-period finite dependence. The
[practical of Class 13](13_ccp_practice.md#zurcher-finite-dependence) derives the
resulting value difference and codes the estimator built on it.

Finite dependence can take more than one period and require sequences that react to the
state. In the stylized labor-supply example of Section 3.2.2 of {cite:t}`arcidiaconoConditionalChoiceProbability2011`, working today
raises human capital $z$ by 1 or 2 units with equal probability. Working in any later
period raises it by exactly 1, and staying home leaves it unchanged. Starting with
*home*, the sequence *work, work* reaches $z+2$ at $t+3$ for sure. Starting with
*work*, stay home at $t+1$ if the gain was 2 and work if it was 1, then stay home at
$t+2$: again $z+2$ at $t+3$. The pair has two-period finite dependence, and the second
sequence is state-contingent.

### What if no finite dependence?

Finite dependence is a property of the transition probabilities, and
nothing guarantees it. If no later choice can undo or replicate the effect of today's
choice on the state, the terms never cancel. 
Then the infinite sum, or the matrix
inversion of the [identification section](#ccp-identification), is back. 

Finding the sequences by hand is application-specific;
{cite:t}`arcidiacono2019NonstationaryDynamicModels` search for the decision weights
numerically. 

See also the JMP by Jaepil {cite:t}`lee_structural_2025` for an application of numerically searching for finite dependence sequences in estimation.

The CCPs that enter are those at the states reachable within $\rho$
periods, including rarely visited ones, where $\log \hat P$ is noisy. The pay-off is
largest in nonstationary models with short panels. There, the CCPs in the last observed
period carry all the information about the unobserved future, and flow utilities are
estimable up to $\rho$ periods before the sample ends without modeling the horizon.

:::{div}
:class: discussion
- For bus engines does replacement constitute a renewal action, and what does $\psi$ depend on?
- Which CCPs does the Zurcher model require, and at which mileages are
  they expected to be poorly estimated in the bus data?
- Think of a model in your own field. Does it have a renewal or terminal action, and if
  not, what would a finite-dependence sequence look like?
:::



(ccp-identification)=
## Identification

The previous results have immediate implications for **identification** of the model
primitives of dynamic discrete choice models in general.

- The model is characterized by the primitives $\{u, \pi, q, \beta\}$ (and the horizon,
  infinite here)
- The data are a panel of states and choices $(x,d)$

The question is: which primitives the data pin down??

**Assumption:** 
the discount factor $\beta$ and the distribution of the taste shocks $q$ are taken as **known**

Further, consistent estimate of the transition probabilities $\pi$ are available (see below).

So, main focus is *non-parametric* identification of flow utilities $u$!

Assume a discrete state space, so that the integral over $\pi(x'|x,d)$ is a sum and can
be represented by a matrix multiplication. As before, $\Pi(d)$ the transition probability
matrix for the state space $X$ under action $d$.

First, we can consistently estimate the choice probabilities $P(d|x)$, the CCPs, and
the transition probabilities $\Pi(d)$ from the data on $(x,d)$ in the first stage,
giving common name *CCP* or *two-stage* estimators to the whole family of CCP-based estimation methods.

````{hint}
What does the first stage need, and what does it deliver?

- every state is visited in the data, so that $P(d|x)$ and $\Pi(d)$ can be estimated
  everywhere
- every choice probability is positive, because theory requires full support of error terms, implying choice probabilities strictly interior in the corresponding simplex; in the EV1 case we need to compute $\log P(d|x)$!
- $q$ known $\implies$ $\psi(x,d)$ is a known function of the identified CCPs
````

Now fix the reference alternative $d_0$ and let its flow utility $u(x,d_0)$ be given for
every $x$. Stack all entities into vectors over the state space $X$.

The definition of the choice-specific value and the relationship
$V^\sigma - v(d_0) = \psi(d_0)$ give

$$
V^\sigma - \psi(d_0) = u(d_0) + \beta \Pi(d_0) V^\sigma
\implies
V^\sigma = [I - \beta \Pi(d_0)]^{-1} \big(\psi(d_0) + u(d_0)\big)
$$

This is the integrated value function in terms of first-stage objects and the flow
utility of the reference alternative $u(d_0)$.

To recover the utility of the other actions, follow the same route from
$v(d) = u(d) + \beta \Pi(d) V^\sigma$ and $V^\sigma - v(d) = \psi(d)$:

$$
V^\sigma - \psi(d) = u(d) + \beta \Pi(d) V^\sigma
\implies
u(d) = -\psi(d) + [I - \beta \Pi(d)] V^\sigma
$$

and substituting $V^\sigma$,

$$
u(d) = -\psi(d) + [I - \beta \Pi(d)][I - \beta \Pi(d_0)]^{-1} \big(\psi(d_0)+u(d_0)\big)
$$

The same result can be written without matrix inversion, as a discounted sum along a
choice sequence, using
Theorem 1 [above](#finite-dependence):

- take the sequence that chooses $d_0$ in every period after $t$,
  $\omega_\tau(d_0|x_\tau,\cdot) = 1$ for $\tau > t$
- let $\kappa_\tau(\cdot|x,d)$ be the distributions of the state it generates after the
  initial choice $d$

````{attention} Theorem: Identification

Let $\beta$, $q$
and $\pi$ be known, and let $\kappa_\tau(\cdot|x,d)$ be generated by the choice sequence
that takes $d_0$ in every period after $t$. Then for all $x \in X$ and $d \in D(x)$

$$
\begin{aligned}
u(x,d) = {} & u(x,d_0) + \psi(x,d_0) - \psi(x,d) \\
& + \sum_{\tau=t+1}^{\infty} \beta^{\tau-t} \sum_{x_\tau \in X}
\big[ u(x_\tau,d_0) + \psi(x_\tau,d_0) \big]
\big[ \kappa_{\tau-1}(x_\tau|x,d_0) - \kappa_{\tau-1}(x_\tau|x,d) \big]
\end{aligned}
$$

so if the flow utility of the reference alternative $u(x,d_0)$ is known in every state,
$u$ is identified.

{cite:p}`arcidiaconoIdentifyingDynamicDiscrete2020`
````

How does this relate to what we already have?

- It is the matrix formula above written out term by term
- It is Theorem 1 of the finite dependence section applied to $d$ and $d_0$ with the
  same continuation, and differenced

````{note} From the matrix formula to the sum
:class: dropdown

Split $I - \beta\Pi(d) = [I - \beta\Pi(d_0)] + \beta[\Pi(d_0) - \Pi(d)]$ and expand
$[I - \beta\Pi(d_0)]^{-1} = \sum_{s \geqslant 0} \beta^s \Pi(d_0)^s$. The row $x$ of
$\Pi(d)\Pi(d_0)^{\tau-t-1}$ is $\kappa_{\tau-1}(\cdot|x,d)$, which gives exactly the
sum in the theorem.
````

Why is this useful?

When the two sequences reach the same distribution of the state after $\rho$ periods,
the sum stops at $t+\rho$.

- $u$ is then identified from CCPs at most $\rho$ periods ahead
- this is what makes identification off short panels possible in nonstationary models

The converse matters as much. {cite:t}`arcidiaconoIdentifyingDynamicDiscrete2020` also
show that *any* bounded choice of $u(x,d_0)$, state by state, together with the $u(x,d)$
implied by the formula above, generates exactly the same CCPs and transitions as the
true model: the two are **observationally equivalent**.

In the matrix form, replacing $u(d_0)$ by an arbitrary vector $c$ changes the other
utilities to

$$
u^*(d) = u(d) + [I - \beta \Pi(d)][I - \beta \Pi(d_0)]^{-1} \big(c - u(d_0)\big)
$$

without changing anything in the data. Therefore:

- flow utilities are identified only relative to one alternative per state
- the normalization does not have to set it to zero, nor use the same alternative in
  every state
- a normalization is needed unless some data inform the level of utility, for example
  data on costs or prices
- or unless the parameterization of $u$ restricts how utilities vary across states and
  alternatives tightly enough to rule out the transformation above

A parametric form alone is not enough: if it is flexible enough to absorb
$u^*(d) - u(d)$, the normalization is still implicit in it.

````{note} Normalizations are not innocuous

Normalizing the flow utility of an alternative, to zero or to anything else, is common
but not innocuous. Observationally equivalent models can predict different outcomes of a
counterfactual: some counterfactuals are invariant to the normalization and others are
not, see {cite:t}`kalouptsidi2021IdentificationCounterfactualsDynamic`. The discount
factor $\beta$ and the distribution $q$, taken as known above, are likewise not
identified without further restrictions; for $\beta$ see
{cite:t}`abbringIdentifyingDiscountFactor2020`.
````

:::{div}
:class: discussion

- In the Zurcher model, $u(x,\text{keep}) = -c(x,\theta_1)$ and
  $u(x,\text{replace}) = -RC - c(0,\theta_1)$. Which normalization does this
  correspond to?
- The derivation takes $\beta$ as given. Where exactly does it enter, and why can it not
  be recovered from the same equations?
- Which counterfactuals survive a change of the normalization, and which do not?
:::



(ccp-estimation)=
## CCP-based estimation

There are many estimation approaches based on the CCP representation of the dynamic
discrete choice models:

- Minimum distance {cite:p}`altug1998EffectWorkExperience`
- Simulated moments {cite:p}`hotz1994SimulationEstimatorDynamicc`
- Asymptotic least squares {cite:p}`pesendorfer2008AsymptoticLeastSquares`
- Quasi-maximum likelihood {cite:p}`hotz1993ConditionalChoiceProbabilitiesb`
- Pseudo-maximum likelihood {cite:p}`aguirregabiriaSwappingNestedFixed2002`
- Nested pseudo-likelihood {cite:p}`aguirregabiriaSequentialEstimationDynamic2007`
- Many-many more

All of them share the same two-step structure, which is why they are collectively
called **two-step CCP estimators**:

```
Input: panel data (x_it, d_it)
Algorithm:
  1. Estimate the CCPs P(d|x) and transition probabilities Π(d) directly from the data
     - non-parametrically, like frequency counts
     - parametrically, like multinomial logit or nested logit for P(d|x)
       and a discrete choice model for Π(d)
     - semi-parametrically, like flexible or local logit
  2. Use the estimated CCPs and transition probabilities to construct
     a criterion function that depends on the structural parameters θ
  3. Optimize the criterion function to obtain the estimates of θ
     by one of the approaches above
Output: estimate of θ, obtained without solving the model
```

The second step is where the methods differ: the value function differences
$\Delta v(x)$ implied by $\theta$ and the first-stage estimates are matched to the
first-stage CCPs by a distance, a set of moments, or a likelihood. In the Zurcher model
with the logit inverse map, step 1 estimates the replacement probability in every mileage
bin, a frequency table in principle and a smoothed one in practice, and step 2 is a static logit estimation with the future differenced out by finite
dependence. This is what we code in the
[practical of Class 13](13_ccp_practice.md), and the pseudo-likelihood versions of
step 2 are the subject of [Class 14](14_npl.md).

### Issues with CCP-based estimation

The first-stage estimates enter the criterion function directly, so
their sampling error is inherited by $\hat\theta$. 

With many states and a modest panel, frequency estimates of $P(d|x)$ are noisy or exactly zero in rarely visited
states, and $\log P$ in the inverse map amplifies the noise. 

Two-step estimators are therefore less efficient than NFXP in finite samples, and in the tails of the state
space they can be badly biased. 

Smoothing the first stage helps, at the price of a
choice of smoother that the theory does not make for you.

:::{div}
:class: discussion

- What are the strengths of CCP-based estimation, and how does it compare to NFXP?
- What are the potential weaknesses and difficulties of this approach?
:::

## Further topics

The CCP estimation approach gave rise to a number of further avenues of methodological
research and applied work:

- Unobserved heterogeneity among decision-makers,
  {cite:t}`arcidiaconoConditionalChoiceProbability2011`, who developed the theory of
  representation with correction terms and an expectation-maximization algorithm to
  estimate finite mixtures of types
- Computational finite dependence: find and exploit the finite dependence structure in
  applications by comparing the distributions of future states at different points of
  the state space, {cite:t}`arcidiacono2019NonstationaryDynamicModels`
- Identification literature: the discount factor by
  {cite:t}`abbringIdentifyingDiscountFactor2020`, and further special cases and
  circumstances

````{note} References and additional resources

(12_ccp_references)=
- 📖 {cite:t}`hotz1993ConditionalChoiceProbabilitiesb` "Conditional Choice Probabilities and the Estimation of Dynamic Models"
- 📖 {cite:t}`arcidiaconoConditionalChoiceProbability2011` "Conditional Choice Probability Estimation of Dynamic Discrete Choice Models With Unobserved Heterogeneity"
- 📖 {cite:t}`arcidiacono2019NonstationaryDynamicModels` "Nonstationary dynamic models with finite dependence"
- 📖 {cite:t}`arcidiaconoIdentifyingDynamicDiscrete2020` "Identifying dynamic discrete choice models off short panels"
- 📖 {cite:t}`kalouptsidi2021IdentificationCounterfactualsDynamic` "Identification of counterfactuals in dynamic discrete choice models"
- 📖 {cite:t}`abbringIdentifyingDiscountFactor2020` "Identifying the discount factor in dynamic discrete choice models"
- 📖 {cite:t}`aguirregabiriaDynamicDiscreteChoice2010` "Dynamic discrete choice structural models: A survey"
- 📺 Econometric Society Dynamic Structural Econometrics (DSE) lecture by Robert Miller [YouTube video](https://youtu.be/CjAHEy3RZOs?si=ZIjDujz8KstwNOhD)

````
