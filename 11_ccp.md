---
title: 📖 Conditional choice probabilities, identification and two-step estimator
short_title: 📖 CCP estimation
subtitle: Class 11 — Tuesday, September 29
exports:
  - format: typst
    output: exports/11_ccp.pdf
downloads:
  - file: 11_ccp.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

:::{div}
:class: homework-link
[Homework: recovering value differences from choice data](#task11.1)
:::

NFXP solves the model inside every likelihood evaluation. This class turns the
argument around: the conditional choice probabilities (CCPs) observed in the data pin
down the value function differences, so the model can be estimated without ever
solving it. The same result of {cite:t}`hotz1993ConditionalChoiceProbabilitiesb` also
says exactly what the data can and cannot identify.

## Integrated $\ne$ expected value function

Maintain the setup of the Zurcher engine replacement model of
{cite:t}`rustOptimalReplacementGMC1987` from Class 9, leaning to more generality. The
Bellman equation of the Zurcher problem is

$$V(x,\varepsilon) = \max_{d\in D(x)} \Big\{ \underbrace{u(x,d) + \beta
\int_{X} \Big( \int_{\Omega} V(x',\varepsilon') q(\varepsilon'|x') d\varepsilon'\Big)
\pi(x'|x,d) dx'}_{v(x,d)}
+ \varepsilon_d \Big\}$$

and it breaks into three pieces that are useful on their own: the choice-specific
value, the expected value, and the value itself,

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

Let's repeat the terminology. $v(x,d)$ is the **choice-specific value function**, or
conditional value function, and $EV(x,d)$ is commonly referred to as the **expected
value function** and sometimes as the ex-post value function.

Here is a new object and yet another representation. The **integrated value function**
$V_\sigma(x)$, also known as the ex-ante value function, is the value function with the
taste shock integrated out,

$$
V_\sigma(x) = \int_{\Omega} V(x,\varepsilon) q(\varepsilon|x) d\varepsilon,
\qquad
EV(x,d) = \int_{X} V_\sigma(x') \pi(x'|x,d) dx'
$$

The subscript $\sigma$ follows the notation of
{cite:t}`aguirregabiriaSequentialEstimationDynamic2007`, who define it as the value
function under a strategy profile $\sigma$.

```{figure} _static/img/bellman_circle.jpg
:width: 80%
:align: center

The Bellman circle of value functions: plain value function $V(x,\varepsilon)$, integrated value function $V_\sigma(x)$, expected value function $EV(x,d)$ and choice-specific value function $v(x,d)$
```

We could write the Bellman equation in the space of integrated value functions
$V_\sigma(x)$ by cutting the *circle of Bellman* at a different point:
$V(x,\varepsilon) \rightarrow V_\sigma(x) \rightarrow EV(x,d) \rightarrow v(x,d) \rightarrow V(x,\varepsilon) \rightarrow \dots$
The new representation of the Bellman equation is

$$
V_\sigma(x) = \int_{\Omega} \max_{d'\in D(x)} \Big\{ \underbrace{u(x,d') + \beta
\int_{X} V_\sigma(x')
\pi(x'|x,d') dx'}_{v(x,d')}
+ \varepsilon_{d'} \Big\} q(\varepsilon|x) d\varepsilon
=
\int_{\Omega} \max_{d'\in D(x)} \left\{ v(x,d')
+ \varepsilon_{d'} \right\} q(\varepsilon|x) d\varepsilon
$$

This is the expectation of the maximum utility in a RUM with alternative utilities
given by $v(x,d') + \varepsilon_{d'}$, which {cite:t}`mcfadden1974ConditionalLogitAnalysisa`
called the **social surplus function** in Class 6. In the EV1 case max-stability gives
it in closed form, with $\gamma \approx 0.5772$ the Euler–Mascheroni constant,

$$
V_\sigma(x) = \log \big( \sum_{d' \in D(x)} \exp[v(x,d')] \big) + \gamma
$$

### Choice probabilities

Recall that by the [Williams–Daly–Zachary theorem](6_logit.md#wdz-theorem) in the
general case the choice probabilities are the derivatives of the social surplus
function,

$$
P(d|x) = \frac{\partial}{\partial v(x,d)}
\mathbb{E}\left\{\max_{d' \in D(x)} \big[ v(d',x) +\varepsilon(d') \big]\Big|x\right\}
=
\frac{\partial V_\sigma(x)}{\partial v(x,d)}
$$

and under the EV1 assumption the derivative of the logsum is the multinomial logit,

$$
P(d|x) =
\frac{\partial \log \big( \sum_{d' \in D(x)} \exp[v(x,d')] \big)}{\partial v(x,d)}
=
\frac{\exp[v(x,d)]}{\sum_{d'\in D(x)} \exp[v(x,d')]}
$$

## Inversion theorem

📖 {cite:t}`hotz1993ConditionalChoiceProbabilitiesb` "Conditional Choice Probabilities
and the Estimation of Dynamic Models"

Let $d_0$ denote the reference alternative, so that the values of other alternatives
will be measured relative to it. For each $x$ consider the vector of value differences
$\Delta v(x) \in \mathbb{R}^{K-1}$, where $K=|D(x)|$ is the number of alternatives,
with elements

$$
\Delta v(x,d) = v(x,d) - v(x,d_0), \quad d \in D(x)\setminus\{d_0\}
$$

Another way to express the choice probabilities $P(d|x)$ is through the integral with
respect to the distribution $q(\varepsilon|x)$ of the indicator that $d$ is the
maximizer. Rewriting the event step by step,

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

The steps are: write the argmax as a system of inequalities, subtract $v(x,d_0)$ from
both sides of each inequality so that only differences appear, and read the result as
the probability mass of a region of $\Omega$ that depends on $\Delta v(x)$ only.
Compare this derivation to the
[static multinomial logit model](6_logit.md#wdz-theorem) of Class 6, where the same
region was integrated in closed form.

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

## Relationship between $V_\sigma(x)$, $v(x,d)$ and choice probabilities

Recall that the choice probability $P(d|x)$ in the general case is simply the
expectation of the indicator that a particular choice $d$ yields the maximum utility,

$$
P(d|x)= \hbox{Prob}\left\{d = \arg\max_{d' \in D(x)} \{v(x,d')+\varepsilon_{d'}\}|x\right\}
=
\mathbb{E}\left[\left. I\left\{ d = \arg\max_{d' \in D(x)} \{v(x,d')+\varepsilon_{d'}\} \right\} \right|x \right]
$$

The expression for $V_\sigma(x)$ as the expected maximum can then be expanded using
the *law of iterated expectations*, conditioning on which alternative is chosen:

$$
V_\sigma(x) = \int_{\Omega} \max_{d'\in D(x)} \left\{ v(x,d')
+ \varepsilon_{d'} \right\} q(\varepsilon|x) d\varepsilon
=
\mathbb{E}\left[
\max_{d'\in D(x)} \left\{ v(x,d')
+ \varepsilon_{d'} \right\}
\right] =
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
{cite:t}`arcidiaconoConditionalChoiceProbability2011`. It is the expectation of the
random component conditional on the event that the alternative it is associated with
has the highest value. We end up with

$$
V_\sigma(x) =
\sum_{d \in D(x)} P(d|x)
\big(
v(x,d) + e(x,d)
\big)
$$

We can make one more step, using the Hotz–Miller inversion with a reference alternative
$d_0$, to arrive at

$$
V_\sigma(x) - v(x,d_0) =
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

where $\psi(x,d_0)$ is a function of the choice probabilities and not of the value
functions, as shown by {cite:t}`arcidiaconoConditionalChoiceProbability2011`. In other
words, the difference between the integrated value function and any choice-specific
reference value is a function of the choice probabilities only. This is the result
that every CCP-based method builds on.

### Special case of EV1 and GEV

{cite:t}`arcidiaconoConditionalChoiceProbability2011` show that the correction term
has a closed form in the two workhorse cases,

$$
\text{EV1} \;\implies\;e(x,d) = \gamma-\log P(d|x)
$$

$$
\text{GEV} \;\implies\;e(x,d) = \gamma - \sigma\log P(d| x) -(1-\sigma)\log\sum_{d' \in N(d)} P(d'|x)
$$

where $N(d)$ is the set of alternatives in the same *nest* as $d$ and
$\sigma\leqslant 1$ is the scale parameter within the nest. In the EV1 case
$\psi(x,d_0) = \gamma - \log P(d_0|x)$: one line of algebra from the logsum, and the
whole relationship between $V_\sigma$ and the CCPs is the logit formula read backwards.

## Identification

The previous results have immediate implications for **identification** of the model
primitives of dynamic discrete choice models in general. We are interested in
non-parametric identification of the model primitives $\{u, \pi, q, \beta\}$ given the
data on $(x,d)$. Here is a brief sketch.

Assume a discrete state space, so that the integral over $\pi(x'|x,d)$ is a sum and can
be represented by a matrix multiplication. Denote $\Pi(d)$ the transition probability
matrix for the state space $X$ under action $d$.

First, we can consistently estimate the choice probabilities $P(d|x)$, the CCPs, and
the transition probabilities $\Pi(d)$ from the data on $(x,d)$ in the first stage,
giving the name to the corresponding CCP-based estimation methods. Then fix the
reference alternative $d_0$ and set $u(x,d_0) = 0$ for all $x$. Stacking all entities
into vectors over the state space $X$ we obtain

$$
v(d_0) =
\underbrace{u(d_0)}_{=0} + \beta \Pi(d_0) V_\sigma
$$

From the relationship between $V_\sigma(x)$ and $v(x,d)$ we have
$V_\sigma - v(d_0) = \psi(d_0)$, and substituting,

$$
V_\sigma - \psi(d_0) = \beta \Pi(d_0) V_\sigma
\implies
V_\sigma = [I - \beta \Pi(d_0)]^{-1} \psi(d_0)
$$

This is an expression for the integrated value function that only depends on objects we
can estimate in a first stage. To non-parametrically recover the utility function for
the other actions, follow the same route from $v(d) = u(d) + \beta \Pi(d) V_\sigma$
and $V_\sigma - v(d) = \psi(d)$,

$$
V_\sigma - \psi(d) = u(d) + \beta \Pi(d) V_\sigma
\implies
u(d) = -\psi(d) + [I - \beta \Pi(d)] V_\sigma
$$

$$
u(d) = -\psi(d) + [I - \beta \Pi(d)][I - \beta \Pi(d_0)]^{-1} \psi(d_0)
$$

Given $\beta$ and $q$, the utility function $u(d)$ seems to be non-parametrically
identified for all $d \in D(x)$. However, if we don't assume $u(d_0)=0$ and run through
the same derivation again, we will end up with

$$
u(d) = -\psi(d) + [I - \beta \Pi(d)][I - \beta \Pi(d_0)]^{-1} \big(\psi(d_0)+u(d_0)\big)
$$

and the obtained values of $u(d)$ will be perfectly consistent with the observed data
informing $P(d|x)$ and $\Pi(d)$, whatever $u(d_0)$ is. A normalization is needed unless
some data informs the level of utility, for example data on costs or prices, or we
resort to a parameterized utility function.

````{note} Normalizations are not innocuous

Normalizing the value of an alternative to zero is common but not innocuous: it
changes the counterfactuals the model predicts, see
{cite:t}`kalouptsidi2021IdentificationCounterfactualsDynamic`. The discount factor
$\beta$ is likewise not identified without further restrictions,
{cite:t}`abbringIdentifyingDiscountFactor2020`.
````

:::{div}
:class: discussion

- In the Zurcher model, which normalization does $u(x,\text{keep}) = -c(x,\theta_1)$
  and $u(x,\text{replace}) = -RC$ correspond to, and is $RC$ a level or a difference?
- The derivation takes $\beta$ as given. Where exactly does it enter, and why can it not
  be recovered from the same equations?
- Which counterfactuals survive a change of the normalization, and which do not?
:::

### Finite dependence

Finite dependence is a powerful idea that helps identification and estimation. In many
applications there may be different paths from a point in the state space $x_1$ at time
$t_1$ to the point $x_2$ at time $t_2$. Two different paths require different
sequences of choices, yet at $x_2$ and time $t_2$ the future should look exactly the
same regardless of the path taken to get there.

Therefore the expected values at $t_2$ can be differenced out, resulting in a finite
structure of dependence with no need for matrix inversions, similar to finite horizon
problems.

````{tip} Example: one-period finite dependence in the Zurcher model

The Zurcher model has one-period finite dependence. Indeed, all regenerative models
have this property: renewing today leads to exactly the same future outlook as renewing
one period later, and here exact time subscripts do not matter due to stationarity.
Compare the value of replacing at $x$ today to the value of keeping today and replacing
tomorrow: after tomorrow both paths sit at mileage zero with the same continuation
value, so it drops out of the difference.
````

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
with the logit inverse map, step 1 is a frequency table of replacements by mileage bin,
and step 2 is a static logit estimation with the future differenced out by finite
dependence. This is what we code on Thursday, and the pseudo-likelihood versions of
step 2 are the subject of Class 13.

Where it breaks: the first-stage estimates enter the criterion function directly, so
their sampling error is inherited by $\hat\theta$. With many states and a modest
panel, frequency estimates of $P(d|x)$ are noisy or exactly zero in rarely visited
states, and $\log P$ in the inverse map amplifies the noise. Two-step estimators are
therefore less efficient than NFXP in finite samples, and in the tails of the state
space they can be badly biased. Smoothing the first stage helps, at the price of a
choice of smoother that the theory does not make for you.

:::{div}
:class: discussion

- What are the strengths of CCP-based estimation, and how does it compare to NFXP?
- What are the potential weaknesses and difficulties of this approach?
- Which states of the Zurcher model are visited rarely in the bus data, and what does
  that do to the first stage?
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

(task11.1)=
````{danger} Homework: recovering value differences from choice data

This is a graded homework assignment.

The notebook `tasks/eta_ccp/` in the class repository carries the model class and the
simulator of Class 10 and the steps to take:

```bash
git pull upstream main                              # collect the task
cp -r tasks/eta_ccp solutions/eta_ccp               # work on the copy
```

1. Simulate a panel from the Zurcher model at known parameters, and estimate the CCPs
   $P(\text{replace}|x)$ by frequency counts on the mileage grid. Plot them against the
   model's own $P(\text{replace}|x)$ from the solver, and mark the grid points that are
   never visited.
2. Apply the logit inverse map to recover $\Delta v(x)$ from the estimated CCPs, and
   compare to the value differences from the solver. Report where the two disagree and
   explain why in terms of the first-stage sample sizes per state.
3. Compute $\psi(x,d_0)$ and the integrated value function
   $V_\sigma = [I - \beta \Pi(d_0)]^{-1}\psi(d_0)$ from the first-stage estimates alone,
   and compare to the solver's $V_\sigma(x)$.
4. Repeat 1–3 with a panel ten times smaller and ten times larger, and report how the
   discrepancies scale.

Remember to follow the git workflow <https://dse.iskh.me/workflow/#submission> to submit
your solution.
````

(11_ccp_references)=
````{note} References and additional resources

- 📖 {cite:t}`hotz1993ConditionalChoiceProbabilitiesb` "Conditional Choice Probabilities and the Estimation of Dynamic Models"
- 📖 {cite:t}`arcidiaconoConditionalChoiceProbability2011` "Conditional Choice Probability Estimation of Dynamic Discrete Choice Models With Unobserved Heterogeneity"
- 📖 {cite:t}`arcidiacono2019NonstationaryDynamicModels` "Nonstationary dynamic models with finite dependence"
- 📖 {cite:t}`kalouptsidi2021IdentificationCounterfactualsDynamic` "Identification of counterfactuals in dynamic discrete choice models"
- 📖 {cite:t}`abbringIdentifyingDiscountFactor2020` "Identifying the discount factor in dynamic discrete choice models"
- 📖 {cite:t}`aguirregabiriaDynamicDiscreteChoice2010` "Dynamic discrete choice structural models: A survey"
- 📺 Econometric Society Dynamic Structural Econometrics (DSE) lecture by Robert Miller [YouTube video](https://youtu.be/CjAHEy3RZOs?si=ZIjDujz8KstwNOhD)

````
