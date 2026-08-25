---
title: 📖 Introduction to structural estimation
short_title: 📖 Introduction
subtitle: Class 1 — Tuesday, August 25
---

````{admonition} Definition
:class: attention

Structural estimation refers to econometric methods that estimate parameters of
economic models derived from economic theory, often involving optimization behavior
by agents.
````

## Historical overview

### Stage 1: early foundations (1930s–1950s)

**Leonid Hurwicz, Jacob Marschak, Trygve Haavelmo, Tjalling Koopmans, and the Cowles
Commission**

:::{div}
:class: portrait-row

```{figure} _static/img/portraits/hurwicz.jpg
:alt: Leonid Hurwicz

Leonid Hurwicz
```

```{figure} _static/img/portraits/marschak.jpg
:alt: Jacob Marschak

Jacob Marschak
```

```{figure} _static/img/portraits/haavelmo.jpg
:alt: Trygve Haavelmo

Trygve Haavelmo
```

```{figure} _static/img/portraits/koopmans.jpg
:alt: Tjalling Koopmans

Tjalling Koopmans
```
:::

Key contributions:

- Introduced the **structural vs reduced form** distinction.
- Formalized **simultaneous equations**, endogeneity, identification.
- Haavelmo (1944): probability approach to econometrics → modern estimation foundations.
- Marschak: structure needed for policy analysis.
- Hurwicz: identification theory (order/rank conditions).

**Legacy:** SEM becomes the dominant approach to causal inference in economics.

### Stage 2: the Lucas critique and microfoundations (1970s)

**Robert E. Lucas Jr. and the rational expectations revolution**

:::{div}
:class: portrait-row

```{figure} _static/img/portraits/lucas.jpg
:alt: Robert E. Lucas Jr.

Robert E. Lucas Jr.
```
:::

Key points:

- Structural parameters must be **policy invariant**.
- Ad hoc SEM (with "behavioral equations") fail under policy changes.
- Led to **microfounded models** derived from optimization and equilibrium conditions.
- Birth of **DSGE models** as dynamic, expectations-driven SEM descendants.

**Legacy:** SEM concepts survive but become embedded inside microfounded dynamic systems.

### Stage 3: individual-level structural modelling (1980s–1990s)

**John Rust (1987)**: dynamic discrete choice
**V. Joseph Hotz and Robert A. Miller (1993)**: CCP inversion, reduced-form based identification
**Steven Berry, James Levinsohn and Ariel Pakes (1995)**: random-coefficients demand estimation

:::{div}
:class: portrait-row

```{figure} _static/img/portraits/rust.jpg
:alt: John Rust

John Rust
```

```{figure} _static/img/portraits/hotz.jpg
:alt: V. Joseph Hotz

V. Joseph Hotz
```

```{figure} _static/img/portraits/miller.jpg
:alt: Robert A. Miller

Robert A. Miller
```

```{figure} _static/img/portraits/berry.jpg
:alt: Steven Berry

Steven Berry
```

```{figure} _static/img/portraits/levinsohn.jpg
:alt: James Levinsohn

James Levinsohn
```
:::

Key innovations:

- Microfoundations at the *agent level* (Bellman equations).
- Structural estimation using NFXP and GMM.
- Discrete choice demand becomes the IO workhorse.

**Legacy:** structural microeconometrics becomes a major field.

### Stage 4: modern structural IO (1990s–2020s)

**Richard Ericson and Ariel Pakes (1995)** $\rightarrow$ dynamic games
**Victor Aguirregabiria and Pedro Mira, Ariel Pakes et al., and more recent
computational IO**

:::{div}
:class: portrait-row

```{figure} _static/img/portraits/ericson.jpg
:alt: Richard Ericson

Richard Ericson
```
```{figure} _static/img/portraits/pakes.jpg
:alt: Ariel Pakes

Ariel Pakes
```

```{figure} _static/img/portraits/aguirregabiria.jpg
:alt: Victor Aguirregabiria

Victor Aguirregabiria
```

```{figure} _static/img/portraits/mira.jpg
:alt: Pedro Mira

Pedro Mira
```
:::

Advances:

- Multi-agent dynamic games, heterogeneous firms, entry/exit, investment.
- GMM, simulation, and high-dimensional methods.
- Close integration with industrial organization and antitrust practice.

**Legacy:** modern IO is a fully microfounded, dynamic descendant of SEM.

---

```{figure} _static/img/sem_to_structural.png
:alt: Flow diagram: classical SEM leads to DSGE models, then to dynamic discrete choice models, then to structural IO
:width: 68%
:align: center
:class: img-thin-border
```

Modern structural econometric models — DSGE, dynamic discrete choice, and structural
IO — are **dynamic, microfounded generalizations** of classical **simultaneous
equations models (SEM)**.


## Structural and reduced form econometrics

| Aspect | Structural econometrics | Reduced form econometrics |
|--------|------------------------|---------------------------|
| Essence | Estimation of parameters of economic models derived from theory | Estimation of relationships directly from the data |
| Purpose | Policy analysis, counterfactuals, understanding theoretical mechanisms | Prediction, local causal inference |
| Model | Based on economic theory, optimization behavior | Statistical relationships without explicit economic model |
| Assumptions | About details of economic behavior | About statistical properties of data |
| Identification | Exclusion restrictions, instruments, functional form | Often relies on natural experiments, IV, regression discontinuity |
| Estimation methods | MLE, GMM, simulated methods | OLS, IV, matching, regression discontinuity |
| Data requirements | Often requires detailed microdata | Generally less detailed or aggregate data |
| Applications | Structural models of demand, dynamic programming, games | Reduced form impact evaluations, treatment effects |

- **Structural econometrics** focuses on estimating parameters of economic models
  derived from theory, allowing for counterfactual analysis and policy simulations.
- **Reduced form econometrics** focuses on estimating relationships directly from
  data without explicit reference to underlying economic models, often used for
  prediction or causal inference without structural interpretation.


````{admonition} Definition
:class: attention

Structural model is a disciplined abstraction for counterfactual analysis.
````

:::{div}
:class: discussion

- Is economics a falsifiable science according to Popper (1934)?
- How can we estimate the effect of a *large new* and *unique* policy?
- Can a model be useful without being realistic?
- Evidence-based (experimental) vs. model-based (structural) policy analysis -- in medicine?
:::


## Why structural econometrics?

Four things a structural approach buys you, each of them a consequence of committing
to an explicit model.

**Internal consistency.** Rational agents facing constraints; uncertainty stated as an
explicit probability distribution; well-defined equilibrium concepts (competitive,
Nash, and so on); an explicit data-generating process; estimation grounded in the LLN
and the CLT.

**Elegance and transparency.** Every step can be independently verified, and there is
little room for researcher discretion — though the numerical implementation can still
hide problems.

**Causality.** A model-based concept of causality, resting on assumptions that are
stated rather than implied.

**Counterfactuals.** Generated by the model itself. They are valid only within the
maintained structure, and taking them outside it requires further external validity
assumptions.

### Components of a structural estimation project

1. **Economic model** derived from theory — optimizing agents, possibly with bounded
   rationality, dynamics, observed and unobserved heterogeneity, and either
   equilibrium conditions or strategic interaction between agents.
2. **Data** generated by that model — cross-section, time series or panel; individual
   or aggregate; often censored, truncated or incomplete.
3. **Preliminary data analysis** — cleaning, descriptive statistics, visualization,
   and reduced form estimates that feed back into how the model is built.
4. **Estimation method** — maximum likelihood (full or limited information), GMM,
   simulated methods (simulated MLE, method of simulated moments), or Bayesian.
5. **Identification strategy** — exclusion restrictions, functional form assumptions,
   instruments for endogenous variables, policy invariance.
6. **Counterfactual simulations** — the estimated model as a synthetic laboratory for
   welfare effects, market outcomes and policy analysis.


## Prototype dynamic discrete choice model

### Choices

- Periods: $t = 1, \dots, T$, possibly $T = \infty$
- Actions: $j = 1, \dots, J$
- Indicators: $d_{jt} \in \{0,1\}$

$$
\sum_{j=1}^J d_{jt} = 1, \; \forall t
$$

Mutual exclusivity is not restrictive: combinations can be redefined as distinct actions.

### States and transitions

Let the state be $z_t \in \mathcal{Z}$. This is all the information that is relevant
for the decision at time $t$.

Transition probabilities when action $j$ is chosen at period $t$

$$
f_{jt}(z_{t+1} \mid z_t)
$$

State spaces may be large but are often sparse.

### Preferences and expected utility

Flow/current/instantaneous utility at time period $t$ when action $j$ is chosen

$$
u_{jt}(z_t)
$$

Discount factor

$$
\beta \in (0,1)
$$

Expected utility

$$
\E\left\{\left.
\sum_{t=1}^T \sum_{j=1}^J \beta^{t-1} d_{jt} u_{jt}(z_t)
\right | z_1
\right\}
$$

### Value functions and Bellman equation

Define the optimal policy $d_t^\star(z_t)$ as a vector of zeros and one, indicating
the most desirable action.

The value function conditions on optimal behavior in all future periods; it is the
maximal attainable expected utility from period $t$ on

$$
V_t(z_t)
=
\E\left\{\left.
\sum_{\tau=t}^T
\sum_{j=1}^J
\beta^{\tau-t}
d_{j\tau}^\star(z_\tau)
u_{j\tau}(z_\tau)
\right| z_t
\right\}
$$

Bellman equation:

$$
V_t(z_t)
=
\sum_{j=1}^J
d_{jt}^\star
\left[
u_{jt}(z_t)
+
\beta
\sum_{z'}
V_{t+1}(z')
f_{jt}(z' \mid z_t)
\right]
$$

We will see in Part II how the Bellman equation can be solved and value functions
computed numerically.

Define the choice-specific value:

$$
v_{jt}(z_t)
=
u_{jt}(z_t)
+
\beta
\sum_{z'}
V_{t+1}(z')
f_{jt}(z' \mid z_t)
$$

By definition the optimal choice is:

$$
d_{jt}^\star(z_t)
=
\mathbf{1}
\left\{
v_{jt}(z_t)
\ge
v_{kt}(z_t)\,
\;\forall k
\right\}
$$

### Why unobserved heterogeneity is needed

If agents with identical observed states *are observed in the data* to choose differently

- the model implies indifference between actions
- all actions appear optimal
- the model loses empirical content!

Therefore fully observed heterogeneity is useless for data analysis.

### Unobserved heterogeneity framework

Decompose the state:

$$
z_t = (x_t, e_t)
$$

- $x_t$: observed by both agents and econometrician
- $e_t$: unobserved by econometrician, but observed by agents

The objective becomes predicting **choice probabilities**, not individual choices.

### Data generating process

Observed data are states and corresponding choices:

$$
(x_1, d_1, \dots, x_T, d_T),
$$

with the individual observations given by

$$
(x_1^{(n)}, d_1^{(n)}, \dots, x_T^{(n)}, d_T^{(n)}), \; n = 1, \dots, N
$$

The likelihood integrates out unobservables:

$$
\Pr(d_1, x_2, \dots, d_T \mid x_1)
=
\int \cdots \int
\prod_t
\sum_{j=1}^J d_{jt}
\Pr(d_t \mid x_t, e_t)
\Pr(x_{t+1} \mid x_t, d_t)
\, de_1 \cdots de_T
$$

- A huge multidimensional integral in the general case!
- We will see how Rust's assumptions simplify this drastically

### Maximum likelihood estimation

Let $\theta$ index utilities, transitions, and $\beta$.

$$
\hat{\theta}_{ML}
=
\arg\max_\theta
\frac{1}{N}
\sum_{n=1}^N
\log
\Pr(\text{data}_n \mid x_{1n}; \theta)
$$

Early applications include Miller (1984) and Wolpin (1984).

Other estimation approaches:

- Two-step methods based on *conditional choice probabilities* estimated directly
  from the data (CCP methods) (Hotz–Miller, Aguirregabiria–Mira)
- GMM
- Method of simulated moments (MSM)
- Calibration (no standard errors)

## Multiple decision makers $\rightarrow$ equilibrium models

### Macro style models with aggregate states

- Infinitely many agents
- Individual actions do not affect aggregate states
- Aggregate states affect individual payoffs and transitions
- Aggregate states evolve according to the collective behavior of all agents

### Dynamic Markov games

- Finite number of agents
- Individual actions affect payoffs and transitions of all other agents
- Joint individual actions affect payoffs and transitions
- Equilibrium defined by mutual best responses
  - Nash
  - Bayesian Nash
  - Markov perfect equilibrium (MPE)
  - Oblivious equilibrium, etc.

````{admonition} Example of policy analysis based on structural estimation
:class: tip

{cite:t}`iruc2` 
“Equilibrium Trade in Automobiles,” Journal of Political Economy

- Policy: restructuring of car registration and fuel taxes in Denmark
- Structural model: dynamic discrete choice model of car ownership and usage
- Equilibrium: used car prices adjust to balance supply and demand in the secondary market
- Estimation: MLE using Danish register data on car ownership and usage

```{image} _static/slides_iruc2/eqb_ESAM1.png
:width: 80%
:align: center
:class: img-thin-border
```

```{image} _static/slides_iruc2/eqb_ESAM10.png
:width: 80%
:align: center
:class: img-thin-border
```

```{image} _static/slides_iruc2/eqb_ESAM26.png
:width: 80%
:align: center
:class: img-thin-border
```

```{image} _static/slides_iruc2/eqb_ESAM42.png
:width: 80%
:align: center
:class: img-thin-border
```

```{image} _static/slides_iruc2/eqb_ESAM43.png
:width: 80%
:align: center
:class: img-thin-border
```

```{image} _static/slides_iruc2/eqb_ESAM44.png
:width: 80%
:align: center
:class: img-thin-border
```

```{image} _static/slides_iruc2/eqb_ESAM45.png
:width: 80%
:align: center
:class: img-thin-border
```

```{image} _static/slides_iruc2/eqb_ESAM46.png
:width: 80%
:align: center
:class: img-thin-border
```

```{image} _static/slides_iruc2/eqb_ESAM47.png
:width: 80%
:align: center
:class: img-thin-border
```

{download}`Download complete slide deck <_static/pdf/eqb_ESAM.pdf>`
````

(1_intro_references)=
````{admonition} References and additional resources
:class: note

- 📖 "Economic Theory and Measurement: A Twenty Year Research Report, 1932–1952",
  report by the Cowles Commission, University of Chicago, 1952
  — [download pdf](https://cowles.yale.edu/sites/default/files/2022-08/r1932-52.pdf)

- 📖 {cite:t}`keaneStructuralVsAtheoretic2010` "Structural vs. atheoretic approaches
  to econometrics", *Journal of Econometrics*

- 📖 {cite:t}`wolpin2013LimitsInferenceTheory` "The Limits of Inference without
  Theory", The MIT Press

- 📖 {cite:t}`rustLimitsInferenceTheory2014` "The Limits of Inference Theory: A Review
  of Wolpin (2013)", *Journal of Economic Literature*

- 📖 {cite:t}`sargent2024CritiqueConsequence` "Critique and consequence", *Journal of
  Monetary Economics*, 2024

- 🎥 Michael Keane's lecture on structural estimation at BFI, University of Chicago
  [https://youtu.be/0hazaPBAYWE](https://youtu.be/0hazaPBAYWE)
````

<!-- **Photo credits.** Freely licensed portraits from Wikimedia Commons:
Leonid Hurwicz © Dong Oh, University of Minnesota
([CC BY 3.0](https://creativecommons.org/licenses/by/3.0));
Trygve Haavelmo, University of Oslo (public domain);
Tjalling Koopmans © Eric Koch / Anefo, Nationaal Archief NL
([CC0](https://creativecommons.org/publicdomain/zero/1.0));
Robert E. Lucas Jr. © Centro de Estudios Públicos
([CC BY 3.0](https://creativecommons.org/licenses/by/3.0));
John Rust © F. Iskhakov
([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0)).

The remaining portraits are institutional photographs reproduced here for
identification in a teaching context, and remain the copyright of their respective
holders: Jacob Marschak (Cowles Foundation archives, Yale University),
V. Joseph Hotz (Duke University), Robert A. Miller (Carnegie Mellon
University), Steven Berry (Yale University), James Levinsohn (Yale Jackson School),
Ariel Pakes (NBER), Richard Ericson (East Carolina University), Victor Aguirregabiria
(University of Toronto), Pedro Mira (CEMFI). -->
