---
title: 📖 Rust bus engine replacement model
short_title: 📖 Bus engine model
subtitle: Class 9 — Tuesday, September 22
exports:
  - format: typst
    output: exports/9_zurcher.pdf
downloads:
  - file: 9_zurcher.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

:::{div}
:class: homework-link
[Homework: the bus engine model under the hood](#task9.1)
:::

The bus engine replacement model of {cite:t}`rustOptimalReplacementGMC1987` is the
simplest dynamic discrete choice model taken to real data, and the template for every
model we estimate later. This class builds it from the Bellman equation of Class 7 to a
solver that converges in a handful of Newton steps.

````{seealso} Key reading

{cite:t}`rustOptimalReplacementGMC1987` "Optimal Replacement of GMC Bus Engines: An
Empirical Model of Harold Zurcher", *Econometrica* 55(5), 999–1033. The model, the
solver and the estimator of this class and the next all come from this one paper.
````

````{hint} Running the code for this lecture
:class: dropdown

Every code example below is also a runnable notebook in the course **code repository**,
in the folder `session09-sep22/`.

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

## Rust (1987) and Harold Zurcher

📖 {cite:t}`rustOptimalReplacementGMC1987`, *Econometrica* "Optimal Replacement of GMC
Bus Engines: An Empirical Model of Harold Zurcher" is the foundational paper of dynamic
structural econometrics. It develops the framework that is used across many fields
today, and it does so with five ingredients:

1. A simple dynamic model: a binary discrete state regenerative optimal stopping
   problem
2. A smart solution method: a polyalgorithm with fast Newton–Kantorovich iterations
3. A coherent econometric specification: unobserved states ensuring a non-degenerate
   likelihood
4. A new maximum likelihood estimator: the nested fixed point (NFXP) estimator
5. A very real application with actual data collected by a real person named Harold
   Zurcher

The citation count of the paper is still growing, forty years on.

```{image} _static/img/Rust1987CitationCount_may25.png
:width: 80%
:align: center
```

````{note} Who cares about Harold Zurcher?
:class: dropdown

The same model, with different names for the state and the choice, sits underneath

- Occupational choice: {cite:t}`keaneCareerDecisionsYoung1997`, *JPE*
- Retirement: {cite:t}`rustHowSocialSecurity1997`, *Econometrica*
- Brand choice and advertising: {cite:t}`erdemDecisionMakingUnder1996`, *Marketing
  Science*
- Choice of college major: {cite:t}`arcidiaconoAbilitySortingReturns2004`, *Journal of
  Econometrics*
- Individual migration decisions: {cite:t}`kennanEffectExpectedIncome2011`,
  *Econometrica*
- High school attendance and work decisions: {cite:t}`ecksteinWhyYouthsDrop1999`,
  *Econometrica*
- Sales and dynamics of consumer inventory behavior:
  {cite:t}`hendelMeasuringImplicationsSales2006`, *Econometrica*
- Advertising, learning, and consumer choice in experience good markets:
  {cite:t}`ackerbergAdvertisingLearningConsumer2003`, *IER*
- Route choice models: {cite:t}`fosgerauLinkBasedNetwork2013`, *Transportation
  Research B*
- Fertility and labor supply decisions: {cite:t}`francesconiJointDynamicModel2002`,
  *JoLE*
- Residential and work-location choice: {cite:t}`buchinskyResidentialLocationWork2014`,
  *Econometrica*
- Equilibrium allocations under alternative waitlist designs, deceased donor kidneys:
  {cite:t}`agarwalEquilibriumAllocationsUnder2021`, *Econometrica*
- Equilibrium trade in automobiles: {cite:t}`iruc2`, *JPE*
- ...and many more
````

To this day NFXP is a **very powerful** method for both solving and estimating dynamic
discrete choice models. In {cite:t}`ecma_comment` we show that with a proper
implementation Rust's method is as powerful as the modern general-purpose solvers
unleashed at the same problem today.

The components of a dynamic model — state $x_t$, decision $d_t$, flow utility
$u(x_t,d_t)$, motion rules, value function and policy function — are those of
[Class 7](7_dp.md), and we now fill each of them with content.

## Harold Zurcher's engine replacement problem

```{figure} _static/img/Zuercher_0.jpg
:width: 200px
:align: center

Harold Alois Zuercher, superintendent of the Madison, Wisconsin bus company, June 16,
1926 – June 21, 2020
```

### States and choices

Each bus comes in for repair once a month and Zurcher chooses between ordinary
maintenance $(d_{t}=0)$ and overhaul/engine replacement $(d_{t}=1)$, so the choice set is
$\{ \text{keep} , \text{replace} \} = \{0,1\}$.

The state is the mileage $x_t$ since the last engine overhaul. Harold observes many
other attributes of the buses which come into the shop, but we focus on the main one
for now.

### Zurcher's preferences

Instantaneous payoffs are given by the cost function that depends on the choice:

$$
u(x_{t},d_t,\theta_1)=\left \{
\begin{array}{ll}
-RC-c(0,\theta_1) & \text{if }d_{t}=\text{replace}=1 \\
-c(x_{t},\theta_1) & \text{if }d_{t}=\text{keep}=0
\end{array} \right.
$$

where $RC$ is the replacement cost and $c(x,\theta_1)$ the cost of maintenance with
preference parameters $\theta_1$. Replacing resets the mileage to zero, so the
maintenance cost after replacement is $c(0,\theta_1)$.

### Motion rules / transition probabilities

Mileage is continuous, and the first modeling decision is how to deal with a
*continuous* state space. Rust discretized the range of travelled miles into $n=175$
bins, indexed with $i$, $\hat{X} = \{\hat{x}_1,...,\hat{x}_n\}$ with $\hat{x}_1=0$. The
mileage transition probability is then, for $j = 0,...,J$,

$$
p(x'|\hat{x}_k, d,\theta_2)=
\begin{cases}
Pr\{x'=\hat{x}_{k+j}|\theta_2\}= \theta_{2j} \text{ if }  d=0  \\
Pr\{x'=\hat{x}_{1+j}|\theta_2\}= \theta_{2j}  \text{ if }  d=1
\end{cases}
$$

Mileage in the next period $x'$ can move up at most $J$ grid points, with $J$
determined from the observed distribution of mileage increments. The probabilities of
an increase from any existing mileage are the same, so $\theta_2$ is a short vector.

If **not replacing** ($d=0$) the transition matrix is banded:

$$
\Pi(d=0)_{n \times n} =
\begin{pmatrix}
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
0 & \theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & 0 \\
0 & 0 &\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & 0 \\
\cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
0 & \cdot & \cdot & 0 & \theta_{20} & \theta_{21} & \theta_{22} & 0 \\
0 & \cdot & \cdot & \cdot & 0 & \theta_{20} & \theta_{21} & \theta_{22} \\
0 & \cdot & \cdot & \cdot & \cdot  & 0 & \theta_{20} & 1-\theta_{20} \\
0 & \cdot & \cdot & \cdot & \cdot & \cdot  & 0 & 1
\end{pmatrix}
$$

The last rows are adjusted so that every row sums to one: the top bin absorbs whatever
mileage would fall beyond the grid. If **replacing** ($d=1$) every row is the first row
of $\Pi(d=0)$, because the mileage restarts from zero:

$$
\Pi(d=1)_{n \times n} =
\begin{pmatrix}
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\end{pmatrix}
$$

### Dynamic optimization and the Bellman equation

To minimize the discounted expected value of the costs, Zurcher should find the policy
function $\delta(x_t):X\rightarrow \{\text{keep},\text{replace}\}$ such that
$d_t=\delta(x_t)$ maximizes

$$
\mathbb{E}\sum_{t=0}^{\infty} \beta^t u(x_t,d_t) \longrightarrow \max
$$

The function $V(x_t)$ denotes the maximum attainable value at period $t$,

$$
V(x_t) = \max_{\delta \in \Delta} \mathbb{E} \sum_{j=t}^{\infty} \beta^{j-t} u(x_j,d_j)
$$

where $\Delta$ is the space of policy functions
$\delta(x_t):X\rightarrow \{\text{keep},\text{replace}\}$, and $d_j = \delta(x_j)$.
Using Bellman's principle of optimality, the value function $V(x_t)$ is the solution of
the functional equation

$$
V(x) = \max_{d\in \{\text{keep},\text{replace}\}} \big\{ u(x,d) + \beta \mathbb{E}\big[ V(x')\big|x,d\big] \big\}
$$

where the expectation is taken over the next period values of the state $x'$ given the
motion rule of the problem.

### Bellman operator

The Bellman equation can be written as a fixed point equation of the **Bellman
operator** in the functional space,

$$
T(V)(x) \equiv \max_{d \in \{\text{keep},\text{replace}\}} \big\{ u(x,d) + \beta \mathbb{E}\big[ V(x') \big|x,d\big] \big\}
$$

The Bellman equation is then $V(x) = T(V)(x)$, with the solution given by the fixed
point $T(V) = V$. Everything from [Class 8](8_dp_infinite.md) on contraction mappings
applies.

:::{div}
:class: discussion

Classification of the Zurcher model:

- Finite or infinite horizon?
- Discrete or continuous choice?
- Discrete or continuous states?
- Which solution methods are suitable?
:::

## Making the model suitable for empirical work

Zurcher observes many different attributes of the buses that come into the shop, but
we as econometricians do not. Yet these are likely to be the reason for observing
different behavior in the same states, that is for the same observed mileage. Without
them the model predicts the same choice every time a bus shows up with a given
mileage, and one observation to the contrary makes the likelihood zero.

We therefore need **error terms** in the model, denoted $\varepsilon$, and we are back to
the random utility framework $u(x,d)+\varepsilon(d)$ of Class 6, now in a dynamic
setting.

:::{div}
:class: discussion

- Should the error term be part of the state space of the problem?
- What would the econometrician have to observe for the model without error terms to
  fit the data?
:::

### Updating the Bellman equation

$\varepsilon$ is a new (vector) state variable, and the Bellman equation becomes

$$
V(x,\varepsilon) = \max_{d\in \{0,1\}} \big\{ u(x,\varepsilon_d,d) + \beta \mathbb{E}\big[ V(x',\varepsilon')\big|x,\varepsilon,d\big] \big\}
$$

Writing the expectation out as an integral over both future states,

$$
V(x,\varepsilon) = \max_{d\in \{0,1\}} \big\{ u(x,\varepsilon_d,d) + \beta
\int_{X} \int_{\Omega} V(x',\varepsilon') p(x',\varepsilon'|x,\varepsilon,d) dx'd\varepsilon' \big\}
$$

where $\varepsilon_d$ is the component of the vector $\varepsilon \in \mathbb{R}^2$
which corresponds to $d$. As it stands this is a fixed point in a function of a
continuous two-dimensional argument, with a double integral inside — not something we
want to evaluate thousands of times inside an estimator.

### Rust assumptions

Three assumptions bring the problem back to a fixed point on the mileage grid alone.

````{attention} Definition

**(AS)** Additive separability in preferences:

$$
u(x,\varepsilon_d,d) = u(x,d) + \varepsilon_d
$$

**(CI)** Conditional independence:

$$
p(x',\varepsilon'|x,\varepsilon,d) = q(\varepsilon'|x')\cdot \pi(x'|x,d)
$$

**(EV)** Extreme value type I (EV1) distribution of $\varepsilon$.
````

(AS) is the same as in the static random utility models. (CI) adds to the standard
independence assumptions of static RUM — error terms independent across observations
by random sampling, and independent across choices within the vector — one assumption
specific to dynamic models: conditional on $x$, there is no serial correlation in the
error terms **across time**. (EV) makes things simpler by providing closed form
expressions for the choice probabilities. It can easily be relaxed to GEV, and with the
trouble of multidimensional integration to any other continuous distribution with full
support.

### What the Rust assumptions allow

Under (AS) and (CI) the Bellman equation reads

$$
V(x,\varepsilon) = \max_{d\in \{0,1\}} \big\{ u(x,d) + \varepsilon_d + \beta
\int_{X} \int_{\Omega} V(x',\varepsilon') \pi(x'|x,d) q(\varepsilon'|x') dx' d\varepsilon' \big\}
$$

and the three assumptions are used in turn:

1. Separate out the deterministic part of the **choice specific value function**
   $v(x,d)$ (AS)
2. Compute the expectation by parts (CI)
3. Use max-stability of EV1 to compute the expectation w.r.t. $\varepsilon'$ (EV)

The first two steps give

$$
V(x,\varepsilon) = \max_{d\in \{0,1\}} \big\{ \underbrace{u(x,d) + \beta
\int_{X} \Big( \int_{\Omega} V(x',\varepsilon') q(\varepsilon'|x') d\varepsilon'\Big)
\pi(x'|x,d) dx'}_{v(x,d)}
+ \varepsilon_d \big\}
$$

so that $V(x',\varepsilon') = \max_{d\in \{0,1\}} \big\{ v(x',d) + \varepsilon'_d \big\}$
is the maximum of choice specific values plus EV1 shocks. The third step is the logsum
formula of Class 6 for the expectation of that maximum:

$$
\mathbb{E}\big[ V(x',\varepsilon')\big|x,d\big] =
\int_{X} \log \big( \exp[v(x',0)] + \exp[v(x',1)] \big) \pi(x'|x,d) dx'
$$

The $\varepsilon$ has been integrated out analytically, and what is left is a function
of the mileage grid only.

### Bellman equation in expected value function space

Let $\mathbb{E}\big[ V(x',\varepsilon')\big|x,d\big] = EV(x,d)$. Then the model is
the pair of equations

$$
\begin{aligned}
EV(x,d) &= \int_{X} \log \big( \exp[v(x',0)] + \exp[v(x',1)] \big) \pi(x'|x,d) dx' \\
v(x,d) &= u(x,d) + \beta EV(x,d)
\end{aligned}
$$

This is the Bellman equation *in expected value function space*. When the state space
is discrete the integral is, of course, a simple sum over future values, and
substituting $v$ into the first line gives

$$
EV(x,d) = \sum_{X} \log \big( \exp[u(x',0) + \beta EV(x',0)] + \exp[u(x',1) + \beta EV(x',1)] \big) \pi(x'|x,d)
$$

which defines the operator $\Gamma$ in the expected value function space,

$$
\Gamma(EV)(x,d) \equiv \sum_{X} \log \big( \exp[u(x',0) + \beta EV(x',0)] + \exp[u(x',1) + \beta EV(x',1)] \big) \pi(x'|x,d)
$$

The solution $EV(x,d)$ of the Bellman functional equation is a fixed point of $\Gamma$,
$\Gamma(EV)(x,d)=EV(x,d)$, and this is the fixed point we compute. It is the better one for
four reasons:

- $\Gamma$ is **also a contraction mapping** in the space of expected value functions, as
  shown by {cite:t}`maDynamicProgrammingDeconstructed2021`, so VFI is guaranteed to find
  the unique solution
- the dimensionality of this fixed point problem is smaller than the one in value
  function terms, because $\varepsilon$ is not included
- it is numerically easier to work with the smooth expected values $EV(x,d)$ than with
  $V(x,\varepsilon)$
- it opens very nice numerical optimization possibilities, which is the rest of this
  class

### Choice probabilities

Once the fixed point is found, the *optimal* choice probability $P(d|x)$ is given by
the logit structure (assumption EV):

$$
P(d|x) = \frac{\exp[v(x,d)]}{\sum_{d'\in \{0,1\}} \exp[v(x,d')]}
$$

The choice probabilities are the basis for forming the likelihood function, and we
continue with them on Thursday when talking about structural estimation.

### Solution approaches

The Zurcher model has the following features: infinite horizon, discretized mileage
which is the only state in the $EV$ formulation, so a finite state space, discrete
choice, and idiosyncratic random components. Therefore the suitable solution methods
are

1. value function iterations (VFI)
2. policy iterations
    - *much faster than VFI*
3. Newton–Kantorovich (NK) iterations = Newton-Raphson for Bellman fixed point
    - the reason behind the fast convergence of policy iterations
    - mathematically equivalent to NK iterations


## Policy iterations aka Howard policy improvement algorithm

Break the search of the fixed point of Bellman operator into two steps:

1. Policy evaluation: compute value function for fixed policy function  
1. Policy improvement : find best policy for given value function  

Iterative (back and forth) approach instead of jointly finding the value and policy functions in the fixed point problem.

$$
V(\text{state}) = \max_{\text{decisions}} \big[ U(\text{state},\text{decision}) + \beta \mathbb{E}\big\{ V(\text{next state})  \big| \text{state},\text{decision} \big\} \big]
$$

### Step 1: Policy evaluation

For fixed policy function $\delta$ (decisions) solve for function $ V^\delta(\text{state}) $:

$$
V^\delta(\text{state}) = U(\text{state},\text{decision}) + \beta \mathbb{E}\big\{ V^\delta(\text{next state})  \big| \text{state},\text{decision} \big\}
$$

- functional equation, although simpler than Bellman equation  
- becomes non-linear system of equations with discrete or discretized states  
- becomes *linear* system of equations in many applications where $ \mathbb{E}\{\cdot\} $ can be expressed as matrix multiplication
  - Markov discrete choice problems (all randomness in transition probabilities, no interpolation)
  - when quadrature integration and linear interpolation are used in discretized state spaces  

### Step 2: Policy improvement

For fixed value function $ V^{\delta_0}(\text{state}) $ find the improved optimal policy $\delta_1$ while computing:

$$
V^{\delta_1}(\text{state}) = \max_{\text{decisions}} \big[ U(\text{state},\text{decision}) + \beta \mathbb{E}\big\{ V^{\delta_0}(\text{next state})  \big| \text{state},\text{decision} \big\} \big]
$$

- standard evaluation of the Bellman operator  
- any approaches for implementation of the Bellman operator are applicable (continuous, discrete or discretized choice spaces)

**Policy iterations algorithm:**

```
1. Initialize policy function
2. Compute the value of the current policy 
    - by solving the Bellman equation without max operation
    - assuming that current policy will be applied forever
3. Re-compute the policy function 
    - by applying the Bellman operator to the found value function  
4. Repeat until convergence in policy and/or value function space
```

Why policy iterations? *Rate of convergence!*

Policy iterations are equivalent to applying Newton-Raphson method to solve for the fixed point of the Bellman operator.


## Newton–Kantorovich iterations

Main idea: apply Newton-Raphson method to the fixed point equation $EV = \Gamma(EV)$

- Kantorovich showed how to do this in functional spaces.

Write the fixed point equation as a root finding problem for the operator
$\Gamma$,

$$
EV(x,d) = \Gamma(EV)(x,d) \quad\Leftrightarrow\quad (I - \Gamma)(EV)(x,d)=\mathbf{0}
$$

- $I$ is the identity operator
- so $I-\Gamma$ maps a function to a difference between that function and its image under $\Gamma$
- $\mathbf{0}$ is the zero function

The **NK iteration** is Newton's step on that equation,

$$
EV_{k+1} = EV_{k} - (I-\Gamma')^{-1} (I-\Gamma)(EV_k)
$$

- $I-\Gamma'$ is the Fréchet derivative of the operator $I-\Gamma$
- $\Gamma'$ is the Fréchet derivative of the Bellman operator

We work with finite approximations on the discrete state space, so everything is a
vector or a matrix. 

Let $n$ denote the number of state points (in mileage). 
- $EV(x,d)$ is a vector of length $n$
    - with the first element reused to describe the expected value of replacing
- $\Gamma(EV)(x,d)$ is a non-linear $n$-valued multivariate function of $EV$
- the Fréchet derivative $I-\Gamma'$ is an $n \times n$ Jacobian matrix

NK iterations on finite approximations are therefore solving a
system of $n$ equations with $n$ unknowns with Newton's method — see Class 5 — in $n$ dimensions

### Matrix expression for the finite approximation of the Bellman operator

On the grid, the sum over future states is a matrix product:

$$
EV = \Pi \cdot L \big( U(\text{keep}) + \beta EV, U(\text{replace}) + \beta EV[0] \big)
$$

- $EV$ is an $n \times 1$ column vector
- $\Pi$ is the $n \times n$ matrix of mileage transition probabilities
- $U(\cdot)$ is a column vector of costs for all points in the state space, conditional
  on decision
- $L(\cdot,\cdot)$ is the logsum function returning an $n \times 1$ vector
- the notation $\bullet[i]$ denotes the $i$-th element of a vector

Only $\Pi(d=0)$ appears: replacing sends every bus to the first row, which is why the
replacement value uses the single element $EV[0]$.

### Implementation of the Fréchet derivative

With the finite approximation of the Bellman operator

$$
\Gamma(EV) = \Pi \cdot L \big( U(\text{keep}) + \beta EV, U(\text{replace}) + \beta EV[0] \big)
$$

the Fréchet derivative w.r.t. $EV$ is the $n \times n$ matrix

$$
\frac{\partial \Gamma}{\partial EV} = \Pi \cdot \frac{\partial L\big( U(\text{keep}) + \beta EV, U(\text{replace}) + \beta EV[0] \big)}{\partial EV}
$$

so all that is needed is the derivative of the logsum function. Recall
$L(w_1,w_2) = \log\big[ \exp(w_1) + \exp(w_2)  \big]$ is the expectation of the maximum
of $w_i + \varepsilon_i$ with independent EV1 shocks, and let
$p_i = {\exp(w_i) \over \exp(w_1) + \exp(w_2)}$ denote the corresponding *choice
probabilities*. Differentiating with respect to any scalar $x$,

$$
\frac{\partial L(w_1,w_2)}{\partial x} = p_1 \frac{\partial w_1}{\partial x} + p_2 \frac{\partial w_2}{\partial x}
$$

The derivative of the logsum is the probability-weighted derivative of its arguments.
For $EV[i]$ with $i>0$ only the keep argument depends on it, in one row, so

$$
\frac{\partial L}{\partial EV} = \beta
\begin{pmatrix}
\bullet & 0 & 0 & 0 & \cdot & 0 \\
\bullet & P[1] & 0 & 0 & \cdot & 0 \\
\bullet & 0 & P[2] & 0 & \cdot & 0 \\
\bullet & 0 & 0 & P[3] & \cdot & 0 \\
\cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
\bullet & 0 & 0 & 0 & \cdot & P[n-1]
\end{pmatrix}
$$

where $P[i]$ is shorthand for the *probability of keeping* $P(0|x[i])$ at state point
$i$. The first column, $EV[0]$, is special because the replacement value in every row
depends on it:

$$
\frac{\partial L}{\partial EV[0]} = \beta
\begin{pmatrix}
P[0] + \bar{P}[0] \\
\bar{P}[1] \\
\bar{P}[2] \\
\cdot \\
\bar{P}[n-1]
\end{pmatrix}
$$

where $\bar{P}[i]$ is shorthand for the *probability of replacing* $P(1|x[i])$ at state
point $i$. Putting the two together, the Fréchet derivative in matrix notation is

$$
\frac{\partial \Gamma}{\partial EV} =
\beta \Pi
\begin{pmatrix}
P[0] & 0 & \cdot & 0 \\
0 & P[1] & \cdot & 0 \\
\cdot & \cdot & \cdot & \cdot \\
0 & 0 & \cdot & P[n-1]
\end{pmatrix}
+
\beta \Pi
\begin{pmatrix}
\bar{P}[0] & 0 & \cdot & 0 \\
\bar{P}[1] & 0 & \cdot & 0 \\
\cdot & \cdot & \cdot & \cdot \\
\bar{P}[n-1] & 0 & \cdot & 0
\end{pmatrix}
$$

The first term is $\Pi$ with its columns scaled by the keep probabilities, the second is
non-zero in the first column only, so the whole matrix is

$$
\frac{\partial \Gamma}{\partial EV} = \beta
\begin{pmatrix}
\theta_{20} P[0] & \theta_{21} P[1] & \theta_{22} P[2] & 0 & \cdot & \cdot & \cdot & 0 \\
0 & \theta_{20}P[1] & \theta_{21}P[2] & \theta_{22}P[3] & 0 & \cdot & \cdot & 0 \\
0 & 0 &\theta_{20}P[2] & \theta_{21}P[3] & \theta_{22}P[4] & 0 & \cdot & 0 \\
\cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
\cdot & \cdot & \cdot & \cdot & \cdot & \cdot  & 0 & P[n-1]
\end{pmatrix}
+ \beta
\begin{pmatrix}
\Pi \bar{P}, 0, \dots, 0
\end{pmatrix}
$$

In code this is two lines: an element-wise product of $\Pi$ with the row vector of
keep probabilities, and a correction to the first column.

### NK iterations algorithm

```
Input: model, starting point EV_0, tolerance, maxiter
Algorithm:
  1. Initialize the expected value function at EV_0 (starting values matter!)
  2. Apply the Bellman operator Γ(EV_k), computing the choice probabilities
     and the Fréchet derivative Γ' along the way
  3. Newton–Kantorovich step
        EV_{k+1} = EV_k - (I - Γ')^{-1} (I - Γ)(EV_k)
  4. Repeat from 2 until ||EV_{k+1} - EV_k|| < tolerance
```

The model class below follows the course architecture: the model object holds the
parameters and builds the transition matrix, `bellman()` is the operator with an
optional Fréchet derivative, and three solvers share the `(maxiter, tol, callback)`
signature. Parameter values are the estimates from Rust (1987), with mileage measured
in thousands and the maintenance cost linear in it.

```{code-cell} python3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

class zurcher():
    '''Harold Zurcher bus engine replacement model class, VFI version'''

    def __init__(self,
                 n = 175,           # number of state points
                 RC = 11.7257,      # replacement cost
                 c = 2.45569,       # parameter of maintance cost (theta_1)
                 p = [0.0937,0.4475,0.4459,0.0127],  # probabilities of transitions (theta_2)
                 beta = 0.9999):    # discount factor
        '''Init for the Zurcher model object'''
        assert sum(p)<=1.0, 'Transition probability parameters must sum up to <1'
        self.RC, self.c, self.p, self.beta, self.n= RC, c, p, beta, n

    @property
    def n(self):
        '''Attribute getter for n'''
        return self.__n

    @n.setter
    def n(self, value):
        '''Attribute n setter'''
        self.__n = value
        self.grid = np.arange(self.__n)
        self.trpr = self.__transition_probs()

    def __repr__(self):
        '''String representation of the Zurcher model'''
        return 'Rust model of bus engine replacement (id={})'.format(id(self))

    def __transition_probs(self):
        '''Computing the transision probability matrix'''
        trpr = np.zeros((self.__n,self.__n))  # init
        probs = self.p + [1-sum(self.p)]  # ensure sum up to 1
        for i,p in enumerate(probs):
            trpr += np.diag([p]*(self.__n-i),k=i)
        trpr[:,-1] = 1.-np.sum(trpr[:,:-1],axis=1)  # last column absorbs the mass beyond the grid
        return trpr

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
        # Fréchet derivative
        dev1 = self.beta * self.trpr * pk[np.newaxis,:] # element-wise, pk in rows
        dev1[:,0] += self.beta * self.trpr @ (1-pk)     # w.r.t. EV[0] special case
        return ev1, pk, dev1

    def solve_vfi(self, maxiter=100, tol=1e-6, callback=None):
        '''Solves the Rust model using value function iterations
        '''
        ev0 = np.zeros(self.n) # initial point for VFI
        err0 = 1.0 # initial lagged error
        for iter in range(maxiter):  # main loop
            ev1, pk = self.bellman(ev0)  # update approximation
            err = np.amax(np.abs(ev0-ev1))
            if callback:
                callback(iter=iter,model=self,ev1=ev1,ev0=ev0,err=err,err_prev=err0,pk=pk,method='vfi',itertype='sa')
            if err<tol:
                break  # break out if converged
            ev0 = ev1  # get ready to the next iteration
            err0 = err
        else:
            raise RuntimeError('Failed to converge in %d iterations'%maxiter)
        return ev1, pk

    def solve_nk(self, maxiter=100, tol=1e-6, callback=None):
        '''Solves the model using the Newton-Kantorovich iterations
        '''
        ev0 = np.zeros(self.n) # initial point
        err0 = 1.0 # initial lagged error
        for iter in range(maxiter):
            ev1,pk,dev = self.bellman(ev0,deriv=True) # compute with Fréchet derivative
            ev1 = ev0 - np.linalg.solve(np.eye(self.n)-dev,ev0 - ev1)  # NK step
            err = np.max(np.abs(ev1-ev0))
            if callback:
                callback(iter=iter,model=self,ev1=ev1,ev0=ev0,err=err,err_prev=err0,pk=pk,method='nk',itertype='nk')
            if err < tol:
                break  # break out if converged
            ev0 = ev1  # get ready to the next iteration
            err0 = err
        else:
            raise RuntimeError('Failed to converge in %d iterations'%maxiter)
        ev1,pk = self.bellman(ev1) # compute choice probabilities after convergence
        return ev1,pk

    def solve_poly(self,
                   maxiter=100,
                   tol=1e-10,
                   sa_min=5,         # minimum number of contraction steps
                   sa_max=25,        # maximum number of contraction steps
                   switch_tol=0.025, # tolerance of the switching rule
                   callback=None):
        '''Solves the model using the poly-algorithm'''
        ev0 = np.zeros(self.n) # initial point
        err0 = 1.0 # initial lagged error
        nk = False # start with successive approximations
        for iter in range(maxiter):
            ev1,pk,dev = self.bellman(ev0,deriv=True) # update EV for both types of a step
            err = np.max(np.abs(ev1-ev0))
            nk = True if iter>= sa_max else nk  # have to switch to NK after sa_max
            nk = nk or (iter>=sa_min and abs(err/err0 - self.beta)<switch_tol)  # check if need to switch to NK
            if nk:
                ev1 = ev0 - np.linalg.solve(np.eye(self.n)-dev,ev0 - ev1)  # NK step
                err = np.max(np.abs(ev1-ev0))
            if callback:
                itertype = 'nk' if nk else 'sa'  # label for the iteration type
                callback(iter=iter,model=self,ev1=ev1,ev0=ev0,err=err,err_prev=err0,pk=pk,method='poly',itertype=itertype)
            if err < tol:
                break  # break out if converged
            ev0 = ev1  # get ready to the next iteration
            err0 = err
        else:
            raise RuntimeError('No convergence: maximum number of iterations achieved! Increase maxiter')
        ev1,pk = self.bellman(ev1) # compute choice probabilities after convergence
        return ev1,pk

    def solve_show(self,solver='vfi',verbosity=0,plot=True,**kvargs):
        '''Illustrate solution for given solver = {vfi,nk,poly} and
           print errors/relative errors from iterations (when verbose=True)
           All other arguments are passed to the solver
        '''
        if solver=='vfi':
            chosen_solver = self.solve_vfi
        elif solver=='nk':
            chosen_solver = self.solve_nk
        elif solver=='poly':
            chosen_solver = self.solve_poly
        else:
            raise RuntimeError('Unknown solver in solve_show()')
        if plot:
            fig1, (ax1,ax2) = plt.subplots(1,2,figsize=(14,8))
            ax1.grid(visible=True, which='both', color='0.65', linestyle='-')
            ax2.grid(visible=True, which='both', color='0.65', linestyle='-')
            ax1.set_xlabel('Mileage grid')
            ax2.set_xlabel('Mileage grid')
            ax1.set_title(f'Value function ({solver})')
            ax2.set_title(f'Probability of replacing the engine ({solver})')
        def callback(**argvars):
            iter,itertype,err,derr = argvars['iter'],argvars['itertype'],argvars['err'],argvars['err_prev']
            mod, ev, pk = argvars['model'],argvars['ev1'],argvars['pk']
            if verbosity>1:
                if iter==0:
                    print('Solver = %s'%solver)
                    print('-'*42)
                    print('%7s %16s %16s'%('iter','err','err(i)/err(i-1)'))
                    print('-'*42)
                print('%4d %2s %16.4e %16.12f'%(iter,itertype[:2],err,err/derr))
            elif verbosity>0:
                if iter==0:
                    print('Solver = %s'%solver)
                    print('-'*22)
                    print('%4s %16s'%('iter','err'))
                    print('-'*22)
                print('%4d %16.4e'%(iter,err))
            if plot:
                ax1.plot(mod.grid,ev,color='k',alpha=0.25)
                ax2.plot(mod.grid,pk,color='k',alpha=0.25)
            callback.nriter = iter  # save iter in function object attribute
        # run the chosen solver
        ev,pk = chosen_solver(callback=callback,**kvargs)
        if plot:
            # add solutions
            ax1.plot(self.grid,ev,color='r',linewidth=2.5)
            ax2.plot(self.grid,pk,color='r',linewidth=2.5)
            plt.show()
        print('{} solved with {} in {} iterations'.format(self,solver,callback.nriter))
        return ev,pk
```

The two solvers find the same fixed point, and the grey lines show how differently they
get there:

```{code-cell} python3
# compare SA, NK
model = zurcher(beta=0.9)  # try different value of beta
ev1,pk1 = model.solve_show(maxiter=1500)
ev2,pk2 = model.solve_show(solver='nk')
print()
print('Max diff between value functions is ' ,np.amax(np.abs(ev1-ev2)))
print('Max diff between policy functions is',np.amax(np.abs(pk1-pk2)))
```

:::{div}
:class: discussion

- Does the VFI algorithm always converge?
- What determines the speed of convergence of the VFI algorithm?
- Does the NK algorithm always converge?
:::

### Properties of VFI vs Newton–Kantorovich

VFI is **globally convergent**, because the Bellman operator is a contraction mapping
with a single fixed point, but its convergence rate is $\beta$: **very slow** in
approaching the fixed point when $\beta <1$ is close to one. Newton–Kantorovich has
**quadratic convergence** but is **sensitive to the starting point**, like every
Newton method. Raising $\beta$ from 0.9 to 0.975 makes the difference visible in the
iteration counts:

```{code-cell} python3
# compare SA, NK
model = zurcher(beta=0.975)
ev1,pk1 = model.solve_show(maxiter=1500,verbosity=1,plot=False)
ev2,pk2 = model.solve_show(solver='nk',verbosity=1,plot=False)
print()
print('Max diff between value functions is ' ,np.amax(np.abs(ev1-ev2)))
print('Max diff between policy functions is',np.amax(np.abs(pk1-pk2)))
```

### Poly-algorithm

The NK method may not be convergent at the initial point, whereas successive
approximation (SA) iterations are always convergent. The **poly-algorithm** is the
combination of the two: start with SA iterations, and at approximately the optimal
time switch to NK iterations. The question is what "optimal time" means.

### When to switch to NK iterations?

Suppose the current approximation is a constant away from the fixed point,
$EV_{k-1} = {EV}^\star + C$. Then two consecutive SA errors are

$$
err_{k} = ||EV_{k-1}-EV_{k}|| = ||{EV}^\star+C - \Gamma({EV}^\star+C)|| = ||{EV}^\star + C - {EV}^\star - \beta C|| = C (1-\beta)
$$

$$
err_{k+1} = ||EV_{k}-EV_{k+1}|| = ||\Gamma({EV}^\star+C) - \Gamma(\Gamma({EV}^\star+C))|| = ||{EV}^\star + \beta C - {EV}^\star - \beta^2 C|| = \beta C (1-\beta)
$$

and the ratio of the two errors is $\frac{err_{k+1}}{err_{k}} = \beta$ exactly when the
current approximation is a constant away from the fixed point. An NK iteration
"strips away" such a constant in one step. **Thus, switch to NK iterations when**
$\frac{err_{k+1}}{err_{k}}$ **is close to** $\beta$. The ratio is printed in the third
column below:

```{code-cell} python3
# when to switch from SA to NK
model = zurcher(beta=0.975)
model.solve_show(maxiter=1500,verbosity=2,plot=False);
```

With a tight switching tolerance the poly-algorithm spends a few SA steps, then a few
NK steps, and beats both pure solvers:

```{code-cell} python3
# compare SA, NK and polyalgorithm
m = zurcher(beta=0.975)
ev,pk = m.solve_show(tol=1e-10,maxiter=1500)
ev,pk = m.solve_show(tol=1e-10,solver='nk',plot=False)
polyset = {'sa_min':10,
           'sa_max':100,
           'switch_tol':0.000215,
          }
ev,pk = m.solve_show(tol=1e-10,verbosity=2,solver='poly',**polyset)
```

At Rust's original $\beta = 0.9999$ pure VFI would need tens of thousands of iterations,
and the poly-algorithm solves the model in milliseconds — which is what makes it
possible to put the solver inside a likelihood on Thursday:

```{code-cell} python3
# original parameters from Rust 1987
m = zurcher()
polyset = {'sa_min':10,
           'sa_max':100,
           'switch_tol':0.0005,
          }
ev,pk = m.solve_show(tol=1e-10,solver='poly',**polyset)
# time the solver
%timeit -n 5 -r 10 m.solve_poly(tol=1e-10,**polyset)
```

:::{div}
:class: discussion

- The switching rule compares the error ratio to $\beta$. What happens if the tolerance
  `switch_tol` is too loose, and what if it is too tight?
- NK is equivalent to policy iterations. Where in `solve_nk` is the policy
  evaluation step, and where is the policy improvement?
- What in the model changes if the maintenance cost is made non-linear in mileage, and
  what in the code?
:::

(task9.1)=
````{danger} Homework: the bus engine model under the hood

This is a graded homework assignment.

The notebook `tasks/zeta_zurcher/` in the class repository carries the model class of
this class and the steps to take:

```bash
git pull upstream main                              # collect the task
cp -r tasks/zeta_zurcher solutions/zeta_zurcher     # work on the copy
```

1. Verify the Fréchet derivative: compare `dev1` returned by `bellman()` to a finite
   difference approximation of $\partial\Gamma/\partial EV$ at a random point, and report
   the maximum absolute discrepancy.
2. Break NK: at $\beta = 0.9999$ start `solve_nk` from $EV_0 = 0$ and from a few other
   starting points, such as a large constant or random noise. Record where it converges,
   where it diverges or stalls, and how many SA steps the poly-algorithm needs before its
   NK steps succeed from the same points.
3. Ergodic distribution: under the optimal policy the mileage follows a Markov chain
   with transition matrix $\bar{\Pi} = \text{diag}(P)\,\Pi(d=0) + \text{diag}(\bar P)\,\Pi(d=1)$.
   Compute its stationary distribution and plot it against the replacement
   probability on the same mileage grid. This is the distribution of the data the model
   generates, and we simulate from it on Thursday.

The code shown in this class is `session09-sep22/zurcher.ipynb` in the code
repository, which is worth having beside you while you work:

```bash
cd sb-dse-code && git pull
```

Remember to follow the git workflow <https://dse.iskh.me/workflow/#submission> to submit
your solution.
````

(9_zurcher_references)=
````{note} References and additional resources

- 📖 {cite:t}`rustOptimalReplacementGMC1987` "Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher"
- 📖 {cite:t}`rustNestedFixedPoint2000` "Nested Fixed Point Algorithm Documentation Manual", version 6 {download}`Download pdf <_static/pdf/nfxp_man_2000.pdf>`
- 📖 John Rust "NFXP Pocket Guide" {download}`Download pdf <_static/pdf/nfxp.pdf>`
- 📖 {cite:t}`ecma_comment` "Constrained optimization approaches to estimation of structural models: Comment"
- 📖 {cite:t}`maDynamicProgrammingDeconstructed2021` "Dynamic programming deconstructed" — why $\Gamma$ is a contraction
- 📖 {cite:t}`adda2023DynamicEconomicsQuantitative` "Dynamic Economics", pp. 83–85
- 📺 Econometric Society Dynamic Structural Econometrics (DSE) lecture by Bertel Schjerning [YouTube video](https://youtu.be/houBb2vQFZE?si=VOIG544hnAiOx18x)
- John Rust [wiki](https://en.wikipedia.org/wiki/John_Rust), and Google Scholar [papers citing Rust (1987)](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=16527795233338248687)
- Matlab implementation of the full solver and the NFXP estimator [DSE 2019 GitHub repo](https://github.com/dseconf/DSE2019/tree/master/02_DDC_SchjerningIskhakov)
- `ruspy` Python package implementing NFXP <https://github.com/OpenSourceEconomics/ruspy>
- Computing the ergodic distribution of a Markov chain [CompEcon notebook](https://github.com/fediskhakov/CompEcon/blob/main/20_markov.ipynb)

````
