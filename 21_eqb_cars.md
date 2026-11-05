---
title: 📖 Equilibrium models and trade in automobiles
short_title: 📖 Equilibrium models
subtitle: Class 21 — Thursday, November 5
downloads:
  - file: 21_eqb_cars.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

Today the course turns from one decision maker to many. When the choices of many agents
add up to an aggregate that each of them takes as given, solving the model means finding a
fixed point: first in a small market of platforms, then in the Danish market for cars.

````{seealso} Key reading

{cite:t}`iruc2` "Equilibrium Trade in Automobiles", *Journal of Political Economy*
130(10), 2534–2593. The model, the existence theorems, the equilibrium solver and the
estimates for Denmark in the second half of this class all come from this paper.
````

````{hint} Running the code for this lecture
:class: dropdown

The code for this class is in the folder `session21-nov5/` of the course **code
repository**.

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

(eqb-many-agents)=
## One decision maker → multiple decision makers

Every model in the course so far had **one decision maker** in a given (though stochastic) environment:

- Harold Zurcher replaced bus engines against a fixed mileage process and a fixed
  replacement cost, in the [bus engine model of Class 9](9_zurcher.md)
- a consumer saved at a given interest rate out of a given income, and chose when to
  retire, in the [retirement model of Class 18](18_dcegm.md)

We solved these models with VFI, Newton–Kantorovich, EGM and
DC-EGM, and estimated them with NFXP, CCP, NPL and MSM. 

However, in most markets part of
the *environment is made by the other decision makers*:

- the price of a used car depends on how many owners sell
- the value of a phone network depends on how many people are on it
- the profit of a firm that enters a market depends on how many rivals enter too

So, we now turn to **multiple agent** models, where the environment for the decision making depends on the choices of the other agents. 

The course will cover two kinds of models:

1. **Non-strategic interaction of many agents.** Every agent is small, takes an aggregate
   as given — a price, a market share — and solves a single-agent problem. The aggregate
   is affected by all the choices, and it has to be the one the agents took as given. This
   class and [the next one](22_eqb_practice.md).
2. **Strategic interaction between agents**, in dynamic models known as **dynamic
   games**. A finite number of players each take into account the choices of each other, 
   the environment evolves conditional on all the decisions made, and the equilibrium is a 
   mutual best responce.
   

In both cases we are talking about **equilibrium** as the solution of the model, which is a 
**fixed point** of some *equilibrium operator*. Fundamentally:

1. the equilibrium operator is rarely a contraction, and there can be several equilibria
2. an estimator needs the equilibrium at every trial value of the parameters: a second
  nested fixed point
3. a counterfactual policy moves the equilibrium, and with it the environment of every
  agent, not only their behavior

```{code-cell} python3
:tags: [remove-cell]

import numpy as np
import matplotlib.pyplot as plt
```

```{code-cell} python3
:tags: [remove-cell, raises-exception]

from pathlib import Path
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(14, 12), layout='constrained')
ax.set(xlim=(-6.5, 6.5), ylim=(-5.2, 5.6), aspect='equal'); ax.axis('off')
fig.canvas.draw()                                             # fix the layout before measuring the text
R = 3.6                                                       # radius of the loop
pos = lambda deg, r=R: (r * np.cos(np.radians(deg)), r * np.sin(np.radians(deg)))
box = dict(boxstyle='round,pad=0.25', fc='white', ec='black', lw=1.5)
dec = dict(boxstyle='round,pad=0.25', fc='#e6f2e6', ec='tab:green', lw=2.5)   # the decision variables
for deg, txt, style in ((90, 'market shares\n$s_i, \\; i = 1, \\dots, m$', box),
                        (-30, 'utilities\n$u_{ij} = c_{ij} + \\lambda s_i$', box),
                        (-150, 'choices of consumers\n$P_{ij} = \\dfrac{\\exp(u_{ij})}{\\sum_{k} \\exp(u_{kj})}$', dec)):
    t = ax.text(*pos(deg), txt, ha='center', va='center', fontsize=28, zorder=3)
    bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    ax.add_patch(FancyBboxPatch((bb.x0, bb.y0), bb.width, bb.height, zorder=2, **style))   # bbox= misses tall math
gap = {90: 30, -30: 20, -150: 30, -270: 30}                   # half-widths of the boxes, in degrees
arrow = dict(arrowstyle='-|>', lw=2, color='black', mutation_scale=24)
for a, label, ha in ((90, 'network effect,\nstrength $\\lambda$', 'left'),          # clockwise arcs from node to node
                     (-30, 'logit, EV1 taste\nshocks $\\varepsilon_i$', 'center'),
                     (-150, 'aggregation\n$s_i = \\sum_j p_j P_{ij}$', 'right')):
    t = np.linspace(a - gap[a], a - 120 + gap[a - 120], 60)   # the arc between two boxes, clear of both
    x, y = pos(t)
    ax.plot(x, y, color='black', lw=2)
    ax.annotate('', xy=pos(t[-1] - 2), xytext=(x[-1], y[-1]), arrowprops=arrow)
    ax.text(*pos(a - 60, R + 0.5), label, ha=ha, va='top' if ha == 'center' else 'center', fontsize=26)
ax.text(0, 0, '$s = \\Phi(s)$', ha='center', va='center', fontsize=44, color='tab:red')
Path('_static/eqb').mkdir(exist_ok=True)
fig.savefig('_static/eqb/platform_loop.png', dpi=150, bbox_inches='tight')   # shown in the example box
plt.close(fig)

```

`````{tip} Example: platform competition

(eqb-platform-model)=
A **platform** is a product or a service with **network effects**
{cite:p}`katz1985NetworkExternalitiesCompetition`: its value to a user grows with the
number of other users:
- game consoles
- interface standards
- social media
- two-sided marketplaces
- peer-to-peer services

The setup:

- $m$ platforms, $i = 1, \dots, m$
- a **unit mass** of consumers of $n$ types, $j = 1, \dots, n$, 
- $p_j$ the share of type $j$ in the population
- $c_{ij}$ the **valuation** of platform $i$ by a consumer of type $j$
- $s_i$ the **market share** of platform $i$, the fraction of all consumers on it

The utility of a consumer of type $j$ from platform $i$ is

$$
u_{ij} = c_{ij} + \lambda s_i ,
$$

where $\lambda$ is the strength of the network effect.

Each consumer also draws an extreme value type 1 taste shock $\varepsilon_i$ for every
platform, i.i.d. across consumers and platforms, and joins the platform with the highest
$u_{ij} + \varepsilon_i$. The choice probabilities are logit, as in
[Class 6](6_logit.md#probabilistic-choice):

$$
P_{ij} = \frac{\exp(u_{ij})}{\sum_{k=1}^m \exp(u_{kj})} .
$$

$P_{ij}$ is the probability that a consumer of type $j$ chooses platform $i$, the CCP

Each consumer chooses once, and a unit mass of them turns the probabilities into shares:

$$
s_i = \sum_{j=1}^n p_j P_{ij}, \qquad s = P p ,
$$

a matrix product with $p$ the column vector of type shares.

Combining the three expressions gives $m$ equations in the $m$ market shares:

$$
s_i = \Phi_i(s) = \sum_{j=1}^n p_j \frac{\exp(c_{ij} + \lambda s_i)}{\sum_{k=1}^m \exp(c_{kj} + \lambda s_k)},
\qquad i = 1, \dots, m .
$$

In **equilibrium** the market shares $s^\star$ solve $s^\star = \Phi(s^\star)$: every
consumer chooses the best platform given the shares $s^\star$, and these choices add up
to the shares $s^\star$. The loop that the equilibrium closes:

```{image} _static/eqb/platform_loop.png
:width: 75%
:align: center
:alt: The fixed point loop of the platform market: shares, utilities, choices
```

The expectations of the consumers are fulfilled in equilibrium: nobody would choose
differently knowing the shares that result. 

The **decision variables** are the choice
probabilities $P_{ij}$, in the green box of the diagram: they are the only place where
consumers optimize. Everything else is bookkeeping.

The fixed point can be computed at any node of the loop:

- in the **shares**, $s = \Phi(s)$
- in the **utilities**, $u_{ij} = c_{ij} + \lambda \sum_t p_t P_{it}(u)$
- in the **choice probabilities**, $P = \Lambda(c + \lambda P p)$ with $\Lambda$ the logit
  map

All three have the same solutions. 

Interestingly there can be *multiple solutions* depending on the value of $\lambda$, we will look at this  [in the next class](22_eqb_practice.md#eqb-successive-approximations).

`````

:::{div}
:class: discussion

- Where in this model are the expectations of the consumers, and what makes them rational?
- Which of the three formulations above would you choose to solve numerically?
:::





A market for new and used cars is an equilibrium model built on top of a dynamic discrete
choice problem: every owner solves a replacement problem in the spirit of the
[bus engine model of Class 9](9_zurcher.md), and the prices of used cars clear the market.

The rest of the class follows the presentation of the paper, from the theory to a
counterfactual tax reform in Denmark.


(eqbjpe-motivation)=
## Finding the better way to improve taxation of cars in Denmark

How much is a Volvo in Denmark?

A new Volvo XC90 on the Danish website of the manufacturer has been listed (some years ago) 
for 1,348,091 DKK $\approx$ \$204,816!

```{image} _static/eqb-jpe/volvo_xc90_price.png
:width: 80%
:align: center
:alt: The Danish price of the Volvo XC90
```

The difference is
the Danish registration tax of 180%, plus 25% VAT. 

The price of a Toyota Avensis, relative
to the EU average net price, is about 180 in Denmark and at most 120 in the other countries,
while the Danish price before taxes is among the lowest:

```{image} _static/eqb-jpe/car_price_by_eu_country.png
:width: 70%
:align: center
:alt: Car prices by EU country
```

Meanwhile, car taxes are a large part of the Danish budget:

- 30–50 billion DKK of annual revenue, about 2–3% of GDP and 4–7% of the total tax revenue
- most of it comes from the taxation of the registration and the ownership of new cars

At the same time the externalities of transport are rarely priced right
{cite:p}`parry2005DoesBritainUnited`: congestion is underpriced, and it is not clear that
the taxes match the externalities of air pollution.

What the paper does:

- **policy question**: what are the effects of moving car taxes from purchase to usage,
  that is to fuel?
- **method**: the primary and the secondary markets interact, and the prices of used
  cars are often not observed
- **theory**: a characterization of the equilibrium and a proof of its existence
- **application**: a tractable model with transactions, scrappage, heterogeneous consumers
  and cars, and a flexible utility, with 131 parameters estimated from 39 million
  observations in under 15 minutes
- **conclusion**: the Danish taxes are above the top of the Laffer curve, but a "naive"
  model overstates how far, and could lead to a policy that is bad for the tax revenue and
  for the environment. There is room to reduce CO2 and to raise both the tax revenue and
  welfare at the same time

The paper builds on a line of models of car markets:

- {cite:t}`manski1983AnalysisEquilibriumAutomobile` and {cite:t}`berkovec1985NewCarSales`:
  equilibrium in the secondary market
- {cite:t}`rust1985StationaryEquilibriumMarket`: a dynamic model of equilibrium prices and
  quantities
- {cite:t}`konishi2002ExistenceStationaryEquilibrium`: transaction costs, and a proof of
  existence of the equilibrium
- {cite:t}`stolyarov2002TurnoverUsedDurables`: consumer heterogeneity and stochastic
  transaction costs
- {cite:t}`gavazza2014QuantitativeAnalysisUsedcar`: equilibrium in the secondary market
  with heterogeneous consumers and varying transaction costs
- {cite:t}`esteban2007DurablegoodsOligopolySecondary`: the effects of durability and of
  secondary markets on the behavior of the producers

:::{div}
:class: discussion

- Why would a tax on the registration of new cars change the prices of used cars?
- What does a model need to say about the revenue of a fuel tax that could replace a part of the
  registration tax?
:::

(eqbjpe-consumer)=
## The consumer's problem

The model extends the stationary flow equilibrium of
{cite:t}`rust1985StationaryEquilibriumMarket`: a dynamic discrete choice equilibrium model
with a unit mass of homogeneous consumers, no transaction costs, and a continuous quality
of the durable good, the odometer reading. The extensions:

1. an outside option of having no car
2. transaction costs
3. endogenous scrappage at any car age
4. several kinds of consumer heterogeneity (idiosyncratic unobserved and time-invariant and time-varying observed)
5. multiple makes and models of cars of different vintages
6. a flexible utility, which allows for inclusion of driving and the demand for fuel in the application

The baseline setup has idiosyncratic consumer heterogeneity only = i.i.d. taste shocks:

- a unit mass of consumers who live forever
- cars of $J$ makes and models, $i \in \{1, \dots, J\}$, and of age
  $a \in \{0, 1, \dots, \bar a\}$; a car must be scrapped at age $\bar a$
- a car owner can **trade** the current car $(i, a)$ for a car $(j, d)$, **keep** it
  ($d = \kappa$, if $a < \bar a$), or **purge** it and stay without a car, $i = \varnothing$
- when a car is replaced or purged, there is a further **scrappage** choice: **sell** the
  car, $s = 0_s$, at the market price $P(i, a)$, or **scrap** it, $s = 1_s$, at the fixed
  price $\underline P(i)$

The choices of a car owner form a tree:

```{image} _static/eqb-jpe/tree.png
:width: 90%
:align: center
:alt: Choice tree of a car owner
```

The tree is the nested logit specification of the GEV distribution of the taste shocks
$\varepsilon$. 

In the shown example there are three nests at the top (purge, keep, trade), two intermediate
levels in the trade nest (make and age), and the scrappage choice of the existing car
whenever it is traded.

- Recall though that the nesting can collapse if the scale parameters are equal, and the GEV distribution reduces to the EV1 distribution of the logit model.
- In application we only have two levels of the tree, with the scrappage choice separated from the rest, and the trade nest collapsed to a single level.

The timing within a period:

1. a consumer enters the period with a car of make $i$ and age $a$, or without a car,
   $i = \varnothing$
2. the trading or keeping choice $j \in \{\kappa, \varnothing, 1, \dots, J\}$,
   $d \in \{0, 1, \dots, \bar a - 1\}$, and the scrappage choice $s \in \{0_s, 1_s\}$ are
   made **immediately** at the start of the period
3. the chosen car $(j, d)$ is used during the period, and has an accident with probability
   $\alpha$
4. by the start of the next period the car has aged to $a = d + 1$ with probability
   $1 - \alpha$, and is in the scrappage state $a = \bar a$ with probability $\alpha$

Nobody starts a period with a brand new car, so the state $a$ takes the values
$a \in \{1, \dots, \bar a\}$.

The utility of a consumer is

$$
\text{utility} = u(i, a) - \mu \big[ \text{operating costs} + \text{trade and transaction costs} \big] + \varepsilon ,
$$

where:

- the utility of the car $u(i, a)$ decreases with age, because the services of an older
  car are worse and its maintenance costs more
- $\mu$ is the marginal utility of money
- the taste shocks $\varepsilon$ are generalized extreme value (GEV) to allow for nesting (still, analytic expectations and choice probabilities)

A consumer without a car, or with a car of the terminal age $\bar a$, chooses between no
car and every tradable car:

$$
\begin{aligned}
V(\varnothing, \varepsilon) &= \max\Big[ v(\varnothing, \varnothing) + \varepsilon(\varnothing);\;
\max_{j \in \{1, \dots, J\},\, d \in \{0, \dots, \bar a - 1\}} \big[ v(\varnothing, j, d) + \varepsilon(j, d) \big] \Big] \\
V(i, \bar a, \varepsilon) &= \max\Big[ v(i, \bar a, \varnothing) + \varepsilon(\varnothing);\;
\max_{j \in \{1, \dots, J\},\, d \in \{0, \dots, \bar a - 1\}} \big[ v(i, \bar a, j, d) + \varepsilon(j, d) \big] \Big]
\end{aligned}
$$

Here $v(\varnothing, j, d)$ and $v(i, \bar a, j, d)$ are the values of trading to a car of make
$j$ and age $d$.

An owner of a car $(i, a)$ with $a < \bar a$ can also keep it, and chooses whether to sell or
to scrap the car that is replaced:

$$
V(i, a, \varepsilon) = \max \left\{
\begin{array}{l}
v(i, a, \kappa) + \varepsilon(\kappa) ; \\[4pt]
\max_{s \in \{0_s, 1_s\}} \big[ v(i, a, \varnothing, s) + \varepsilon(\varnothing, s) \big] ; \\[4pt]
\max_{j \in \{1, \dots, J\},\, d \in \{0, \dots, \bar a - 1\},\, s \in \{0_s, 1_s\}} \big[ v(i, a, j, d, s) + \varepsilon(j, d, s) \big]
\end{array}
\right.
$$

The choice-specific value functions are the current utility plus the discounted expected
value. For the consumers without a car and with a terminal car:

$$
\begin{aligned}
v(\varnothing, \varnothing) &= u(\varnothing) + \beta EV(\varnothing) \\
v(\varnothing, j, d) &= u(j, d) - \mu \big[ P(j, d) + T_b(P, j, d) \big] + \beta (1 - \alpha) EV(j, d + 1) + \beta \alpha EV(j, \bar a) \\
v(i, \bar a, \varnothing) &= u(\varnothing) + \mu \underline P(i) + \beta EV(\varnothing) \\
v(i, \bar a, j, d) &= u(j, d) - \mu \big[ P(j, d) - \underline P(i) + T_b(P, j, d) \big] + \beta (1 - \alpha) EV(j, d + 1) + \beta \alpha EV(j, \bar a)
\end{aligned}
$$

For the owners of a car of age $a < \bar a$:

$$
\begin{aligned}
v(i, a, \varnothing, 1_s) &= u(\varnothing) + \mu \underline P(i) + \beta EV(\varnothing) \\
v(i, a, \varnothing, 0_s) &= u(\varnothing) + \mu \big[ P(i, a) - T_s(P, i, a) \big] + \beta EV(\varnothing) \\
v(i, a, \kappa) &= u(i, a) + \beta (1 - \alpha) EV(i, a + 1) + \beta \alpha EV(i, \bar a) \\
v(i, a, j, d, 1_s) &= u(j, d) - \mu \big[ P(j, d) - \underline P(i) + T_b(P, j, d) \big] + \beta (1 - \alpha) EV(j, d + 1) + \beta \alpha EV(j, \bar a) \\
v(i, a, j, d, 0_s) &= u(j, d) - \mu \big[ P(j, d) - P(i, a) + T_s(P, i, a) + T_b(P, j, d) \big] + \beta (1 - \alpha) EV(j, d + 1) + \beta \alpha EV(j, \bar a)
\end{aligned}
$$

$T_b$ and $T_s$ are the transaction costs of buying and selling, which may depend on the
prices.

The expected value function integrates the taste shocks out,

$$
EV(i, a) = \int V(i, a, \varepsilon) f(\varepsilon \mid i, a) \, d\varepsilon ,
$$

and has an **analytic form under GEV**, a composition of logsum functions. In the simplest
case, when GEV reduces to EV1,

$$
EV(i, a) = \sigma \log \Big( \sum_{j, d, s} \exp \frac{v(i, a, j, d, s)}{\sigma} \Big) .
$$

After plugging in the choice-specific values, the expected values solve a Bellman
equation, given the prices:

$$
EV(P) = \Gamma \big( EV(P), P \big) .
$$

This is the Rust engine replacement model on steroids. 

For every value of $P$ the $EV$ is the fixed point of a
contraction mapping, so it can be found by globally convergent successive approximations (VFI),
but it is much better to use Newton's method in function space, the
[Newton–Kantorovich iterations of Class 9](9_zurcher.md#nk-iterations).

The solution $\big( EV(i, 1), \dots, EV(i, \bar a) \big)$ gives the choice-specific values
$v(i, a, j, d, s)$ for all choices, and these give **analytic nested logit choice
probabilities**. In the simplest case

$$
\Pi(j, d, s \mid i, a) = \frac{\exp \{ v(i, a, j, d, s) / \sigma \}}{\sum_{j', d', s'} \exp \{ v(i, a, j', d', s') / \sigma \}} .
$$

With additively separable transaction costs the sell-or-scrap probability is separable
from the rest. 

We use $\sigma$ to denote the scale parameter of the EV1 distribution, and imply that $\sigma$ is a vector of scale parameters for the GEV distribution, one for each level of the tree, or even for each nest.

All the choice probabilities depend on the market prices $P$, through the values.

:::{div}
:class: discussion

- Why does the model need a scrap price $\underline P(i)$ at all, when cars of age
  $\bar a$ must be scrapped?
- Imagine you have coded up this specification. How much work would it be to change the timing assumptions in the model, and rewrite the code?
:::

(eqbjpe-ownership)=
## The ownership distribution

The aggregate state of the economy is the distribution of cars over the consumers:

- $0 \leqslant q_{i,a} \leqslant 1$ is the fraction of owners of make $i$ cars of age $a$,
  $i \in \{1, \dots, J\}$, $a \in \{1, \dots, \bar a\}$
- $0 \leqslant q_\varnothing \leqslant 1$ is the fraction of consumers without a car

The **ownership distribution** is the vector

$$
q = \Big( (q_{11}, \dots, q_{1 \bar a}), \dots, (q_{J1}, \dots, q_{J \bar a}), q_\varnothing \Big) \in \mathbb{R}^{J \bar a + 1} .
$$

It is measured at the beginning of the period, before trading, and it changes from period
to period through trading, the deterministic aging of the cars, and the stochastic
accidents.

Trading is a Markov transition, collected in the $(J \bar a + 1) \times (J \bar a + 1)$
**trade transition matrix**

$$
\Omega(P) =
\begin{bmatrix}
\Delta_{11}(P) + \Lambda_1(P) & \Delta_{12}(P) & \dots & \Delta_{1J}(P) & \Delta_{1\varnothing}(P) \\
\Delta_{21}(P) & \Delta_{22}(P) + \Lambda_2(P) & \dots & \Delta_{2J}(P) & \Delta_{2\varnothing}(P) \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
\Delta_{J1}(P) & \Delta_{J2}(P) & \dots & \Delta_{JJ}(P) + \Lambda_J(P) & \Delta_{J\varnothing}(P) \\
\Delta_{\varnothing 1}(P) & \Delta_{\varnothing 2}(P) & \dots & \Delta_{\varnothing J}(P) & \Pi(\varnothing \mid \varnothing, P)
\end{bmatrix} ,
$$

so that $q \, \Omega(P)$ is the distribution of cars **after the trading phase**. The
$\bar a \times \bar a$ blocks hold the trading components $\Delta_{ij}(P)$ and the keeping
components $\Lambda_i(P)$:

$$
\Delta_{ij}(P) =
\begin{bmatrix}
\Pi(j, 1 \mid i, 1) & \dots & \Pi(j, \bar a - 1 \mid i, 1) & \Pi(j, 0 \mid i, 1) \\
\vdots & \ddots & \vdots & \vdots \\
\Pi(j, 1 \mid i, \bar a) & \dots & \Pi(j, \bar a - 1 \mid i, \bar a) & \Pi(j, 0 \mid i, \bar a)
\end{bmatrix} ,
$$
$$
\Lambda_i(P) =
\begin{bmatrix}
\Pi(\kappa \mid i, 1) & \dots & 0 & 0 \\
\vdots & \ddots & \vdots & \vdots \\
0 & \dots & \Pi(\kappa \mid i, \bar a - 1) & 0 \\
0 & \dots & 0 & 0
\end{bmatrix} .
$$

The last column of each block belongs to the cars of age $\bar a$, but it collects the
choice probabilities of the **new** cars: nobody holds a car of age $\bar a$ after trading,
so the slot is free. 

The last row and column of $\Omega(P)$ hold the purchases by the
consumers without a car and the transitions of the owners to no car:

$$
\Delta_{\varnothing j}(P) = \big[ \Pi(j, 1 \mid \varnothing), \dots, \Pi(j, \bar a - 1 \mid \varnothing), \Pi(j, 0 \mid \varnothing) \big] ,
$$
$$
\Delta_{i \varnothing}(P) = \big[ \Pi(\varnothing \mid i, 1), \dots, \Pi(\varnothing \mid i, \bar a) \big]' .
$$

Aging and accidents are another Markov transition, the **physical transition matrix**
$$
Q = \text{diag}(Q_1, \dots, Q_J, 1)
$$

with the blocks

$$
Q_j =
\begin{bmatrix}
0 & 1 - \alpha & \dots & 0 & \alpha \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \dots & 1 - \alpha & \alpha \\
0 & 0 & \dots & 0 & 1 \\
1 - \alpha & 0 & \dots & 0 & \alpha
\end{bmatrix} .
$$

A car ages by one year with probability $1 - \alpha$ and is lost in an accident with
probability $\alpha$. The last row applies to the new cars and restores the order of the
ages.

The whole idea about the matrices $\Omega(P)$ and $Q$ is that they are **linear operators** on the ownership:

- given the initial distribution $q$ and the prices $P$ in period $t$
- $q \, \Omega(P)$ gives the distribution of ownership immediately after trading phase in period $t$, and
- $q \, \Omega(P) Q$ gives the resulting ownership distribution by the end of the period, and thus at the beginning of period $t + 1$. 


The ownership distribution $q$ is time invariant when

$$
q = q \, \Omega(P) Q,
$$

in other words it is the stationary distribution of a Markov chain $\Omega(P) Q$, similar to the
[ergodic distribution of mileage in Class 10](10_nfxp.md#nfxp-counterfactuals).

````{attention} Theorem (uniqueness of the stationary ownership distribution)
Let $\sigma > 0$ (component-wise if $\sigma$ is a vector). Then the stationary ownership distribution is unique.
````

Proof: With a positive scale of the GEV shocks the choice probabilities have full support, so the
transition matrix $\Omega(P) Q$ is irreducible and aperiodic, and the uniqueness follows
from the fundamental theorem of Markov chains.

(eqbjpe-equilibrium)=
## Stationary equilibrium

The market has three sides:

- an infinitely elastic supply of new cars at the prices $\overline P(j)$
- an infinitely elastic demand for scrap cars at the prices $\underline P(j)$
- $J(\bar a - 1)$ tradable used cars at the prices $P = \big( P(1, 1), \dots, P(J, \bar a - 1) \big)$

````{attention} Definition (stationary equilibrium)
A **stationary equilibrium** is a pair $(q, P)$ such that

1. consumers maximize their expected discounted utility
2. the secondary market clears for all tradable cars
3. the ownership distribution is time invariant
````

The **demand** for cars of make $j$ and age $d$ sums the choice probabilities of all the
consumers who buy them:

$$
D_{jd}(P, q) = \Pi(j, d \mid \varnothing, P) \, q_\varnothing + \sum_{i=1}^J \sum_{a=1}^{\bar a} \Pi(j, d \mid i, a, P) \, q_{ia} .
$$

With a continuum of consumers and a finite number of tradable goods the choice
probabilities are market shares, and in equilibrium all the intended choices take place.

The **supply** comes from the owners who do not keep their car and do not scrap it:

$$
S_{jd}(P, q) = \big( 1 - \Pi(\kappa \mid j, d, P) \big) \big( 1 - \Pi(1_s \mid j, d, P) \big) \, q_{jd} .
$$

Market clearing is the condition of zero excess demand,

$$
ED(P, q) = D(P, q) - S(P, q) = 0 ,
$$

with $D$ and $S$ the vectors of $D_{jd}$ and $S_{jd}$ over all $J(\bar a - 1)$ tradable
cars. Therefore:

- a system of $J(\bar a - 1)$ equations, linear in $q$ and nonlinear in the prices
- $J \bar a + 1$ unknowns in the ownership distribution
- $J(\bar a - 1)$ unknown prices

The stationarity condition supplies the missing equations for $q$.

````{attention} Theorem (existence of the equilibrium)
The stationary equilibrium $(q^\star, P^\star)$ of the automobile economy with
idiosyncratically heterogeneous consumers exists, and in equilibrium

$$
q^\star = q^\star \, \Omega(P^\star) Q, \qquad 0 = ED(P^\star, q^\star) .
$$
````

The theorem only gives existence. $q^\star$ is unique given the prices, but the
uniqueness of $P^\star$ is not known, although the computations have shown no sign of
multiplicity.

The equilibrium also has the **flow property**: for every make $j$ the scrapped cars are
replaced by new ones,

$$
\underbrace{\sum_{a=1}^{\bar a - 1} \Pi(1_s \mid j, a, P) \big( 1 - \Pi(\kappa \mid j, a, P) \big) q^\star_{ja} + q^\star_{j \bar a}}_{\text{all scrapped cars of make } j}
= \underbrace{\sum_{i=1}^J \sum_{a=1}^{\bar a} \Pi(j, 0 \mid i, a, P) \, q^\star_{ia}}_{\text{all new cars of make } j} .
$$

It follows from the stationarity of $q^\star$, which holds for its subvectors as well. It
holds for the outside good too, which follows from the structure of $\Omega(P^\star)$ and
$Q$. So the equilibrium is a **stationary flow equilibrium**.

:::{div}
:class: discussion

- The excess demand has $J(\bar a - 1)$ equations, but the unknowns $(q, P)$ number
  $J \bar a + 1 + J(\bar a - 1)$. Where do the other equations come from?
- What would multiplicity of equilibrium prices mean for the counterfactual simulations
  below?
:::

(eqbjpe-computation)=
## Computing the equilibrium

The proof of existence suggests a straightforward algorithm.

````{attention} Algorithm (stationary equilibrium of the car market)

**Input:** the model fundamentals, including the new car prices $\overline P$ and the scrap
prices $\underline P$, and starting values for the used car prices $P$.

**Steps:**

1. **Consumer problem.** For given prices $P$, solve the DP problem for $EV(P)$ and form
   the trade matrix $\Omega(P)$.
2. **Ownership distribution.** The stationary distribution $q(P)$ solves
   $q = q \, \Omega(P) Q$.
3. **Excess demand.** Substituting $q(P)$ makes the excess demand $ED(P, q(P))$ a function
   of the prices only.
4. **Market clearing.** Solve $ED(P, q(P)) = 0$ for the prices numerically.
5. **Equilibrium.** Compute the stationary ownership distribution $q^\star = q(P^\star)$ at
   the solution $P^\star$.
````

In other words, we have a **double fixed point problem**: 
- the consumer problem is a fixed point in the Bellman equation (inner loop), and 
- the market clearing is a fixed point in the prices (outer loop)

Add *one more outer loop* over parameter values in estimation $\implies$ we have **tripple loop** structure!

The keys to success:

- an efficient gradient-based solver, **Newton's method**
- analytic derivatives, which make it quick and precise
- precise, theoretically founded starting values, which make it robust

Newton's method needs smoothness. A series of lemmas in the paper proves that the
following objects are continuously differentiable in the prices $P$, and in the
structural parameters and other fundamentals such as $\alpha$, $\overline P$ and
$\underline P$:

- the fixed point of the Bellman equation $EV$
- the choice-specific values $v(i, a, j, d, s)$
- the choice probabilities $\Pi(j, d, s \mid i, a)$
- the invariant distribution of the transition matrix $\Omega(P) Q$
- the excess demand $ED(P, q)$

Newton's method is then used three times: 
1. to solve the DP problem (Newton–Kantorovich), 
2. to solve for the equilibrium prices, and 
3. to maximize the likelihood. 

The derivatives come from
repeated application of the chain rule aka *back propagation*

The run times of the equilibrium solver in Matlab, by the number of car types $J$ and the
maximal age $\bar a$:

| | $J = 1$ | $J = 2$ | $J = 5$ | $J = 10$ | $J = 25$ |
| :-- | --: | --: | --: | --: | --: |
| $\bar a = 20$ | 0.081 s | 0.141 s | 0.674 s | 4.103 s | 1 min 6.9 s |
| $\bar a = 50$ | 0.162 s | 0.720 s | 11.566 s | 1 min 12.3 s | |
| $\bar a = 100$ | 0.680 s | 4.407 s | 1 min 29.8 s | | |

This is fast enough to nest the solver in other iterative algorithms: structural
estimation, models of the primary market, and so on.

But a better implementation is in lower-level languages such as C++ or Fortran, or in Python JAX package providing automatic differentiation and parallelization (for multiple cores or GPU) in addition

## Persistent consumer heterogeneity

The model extends to:
- time-invariant types,
- time-variant types, and 
- the combination of the two. 

The paper proves existence in each
case, and the computational cost is **linear in the number of types**. With consumer types
$\tau = 1, \dots, N$:

- the type fractions, or the stationary distribution of time-variant types, are
  $f = (f_1, \dots, f_N)$
- the consumer problem is solved separately for each type $\tau$, which gives the trade
  matrices $\Omega_\tau(P)$, the ownership distributions $q_\tau$ and the excess demands
  $ED(P, q_\tau)$
- the equilibrium conditions become

$$
q_\tau^\star = q_\tau^\star \, \Omega_\tau(P^\star) Q \quad \text{for all } \tau, \qquad 0 = \sum_{\tau=1}^N ED(P^\star, q_\tau^\star) ,
$$

with the **market clearing condition integrated over the types**.

Heterogeneity allows **sorting** of consumers into ages and types of cars: the rich hold
newer and better cars, the poor hold older and worse ones. There are gains from trade, cars
survive longer, and high transaction costs suppress trade and lower the scrappage age. An
example with two types of consumers, with normal transaction costs:

```{image} _static/eqb-jpe/example_holdings_tc_0.png
:width: 60%
:align: center
:alt: Ownership by two consumer types, normal transaction costs
```

and with high transaction costs:

```{image} _static/eqb-jpe/example_holdings_tc_10.png
:width: 60%
:align: center
:alt: Ownership by two consumer types, high transaction costs
```

The consumers sort in both regimes, and the transaction costs affect the two types
differently. The gains from trade between the rich and the poor (see pictures [here](1_intro.md#intro-eqb-cars-example))

## Doubly nested fixed point estimation (DNFXP)

The full solution estimator for the equilibrium model is a **double fixed point**
estimator. With $\theta$ the structural parameters and data on car ownership and observed
household characteristics $x$, the log-likelihood is

$$
L(\theta) = \sum_\tau \sum_{x'} \sum_{x} \log \big( \Pi(x' \mid x, \tau, \theta) \big)
\underbrace{\Pi(x' \mid x, \tau) \, q_\tau(x) f(\tau)}_{\sim \text{ observed } x \rightarrow x'} ,
$$

and its computation has three nested loops:

1. **outer**: maximization of the likelihood over $\theta$
2. **middle**: the equilibrium solver for the prices $P(\theta)$
3. **inner**: the DP solver for the fixed point of the Bellman operator $EV(P, \theta)$

It is the [nested fixed point algorithm of Class 10](10_nfxp.md#nfxp-nested-loop) with
one more loop in the middle.

What the theory delivers is an empirical engine for equilibrium models of secondary
markets of durable goods:

- computationally tractable in real empirical applications
- flexible preferences and transaction costs, from which the heterogeneous preferences
  for every type of car of every age are recovered
- **no data on secondary market prices are needed**: the prices are computed in the model
- **the identity of car owners is not needed**: panel microdata on car ownership and
  trading can be represented by counts of transitions conditional on the consumer types

:::{div}
:class: discussion

- Why does the likelihood need no data on the prices of used cars?
- Which of the three loops dominates the run time, and what do analytic derivatives save in
  each of them?
:::



(eqbjpe-denmark)=
## The Danish car market

The proposed Danish reform lowers the registration taxes and raises the usage taxes, by
road charging or a fuel tax. The outcomes of interest, which have to be captured at the
same time:

- the equilibrium dynamics of car ownership and of the choice of car type: new car sales
  and trade in the secondary market, the age of the fleet and scrappage, the value of the
  car stock
- driving, the demand for fuel, and emissions
- redistribution and welfare

The counterfactual simulation:

1. estimate the model on Danish register data
2. **cut the registration tax rates for new vehicles by half**
3. at the same time **increase the fuel tax** rate so that the revenue is unchanged
4. compute the economic, welfare and environmental implications

To address the proposed reform, we have to include **driving** into the model $\to$ get a handle on the effect of fuel taxes and changes in emissions/driving behavior. 

The approach is to have a **static** driving decision, to work with the **indirect utility** which already *opimizes out* driving choice.

:::{div}
:class: discussion

- What assumptions make this approach feasible?
:::

The utility function includes driving:

$$
u(a, x) = u_{\text{car}}(a) + u_{\text{drive}}(a, x) + \mu [\text{trade} + \text{transaction cost}] ,
$$

with the utility of ownership without driving, $x = 0$, and the net utility of driving

$$
u_{\text{car}}(a) = \alpha_0 + \alpha_1 a + \alpha_2 a^2 ,
\qquad
u_{\text{drive}}(a, x) = (\gamma_0 + \gamma_1 a) x - \mu p x + \frac{\phi}{2} x^2 .
$$

Here $x$ is the number of kilometers driven in a period and $p$ is the cost of fuel per
kilometer, taxes included. 

The parameters $\theta = (\alpha_0, \alpha_1, \alpha_2, \gamma_0, \gamma_1, \mu, \phi)$
may be specific to each consumer type and car type, a dependence suppressed in the
notation.

> The key assumption: the probability of an accident and the other physical deterioration of a car do not depend on driving $x$. 

Driving is then a **static subproblem** of the DP
problem and can be solved separately. 

Solving the FOC in $x$, the structural driving equation implied by the
utility function is

$$
x^\star(p, a) = -\frac{1}{\phi} \big[ \gamma_0 + \gamma_1 a - \mu p \big] .
$$

Identification of $\theta$ uses two kinds of observations: the trading of cars by the
consumers, and their driving. 

The linear **reduced form driving equation** that can be estimated directly from the data, is

$$
x = d_0 + d_1 a + d_2 p ,
$$

where the following transformation of the structural parameters is used:

$$
d_0 = -\frac{\gamma_0}{\phi}, \quad d_1 = -\frac{\gamma_1}{\phi}, \quad d_2 = \frac{\mu}{\phi} ,
$$

Clearly, if we new $\phi$ or $\mu$, for example, we could back out the structural parameters from reduced form estimates.

Reduced form parameters $d_0$, $d_1$, $d_2$ are identified from data on driving,
car age and the price per kilometer. The variation in $p$ across car types comes mainly
from the different fuel efficiency of different types and ages of cars.

Alternatively, plugging the optimal driving into the utility gives the indirect utility of car
ownership, quadratic in age:

$$
u \big( a, x^\star(p, a) \big) = \alpha_0 + \alpha_1 a + \alpha_2 a^2 - \frac{1}{2 \phi} \big[ \gamma_0 + \gamma_1 a - \mu p \big]^2
= u_0 + u_1 a + u_2 a^2 ,
$$

with

$$
u_0 = \alpha_0 - \frac{1}{2 \phi} \big[ \gamma_0 - \mu p \big]^2 ,
\quad
u_1 = \alpha_1 - \frac{\gamma_1}{\phi} \big[ \gamma_0 - \mu p \big] ,
\quad
u_2 = \alpha_2 - \frac{\gamma_1^2}{2 \phi} .
$$

$(u_0, u_1, u_2)$ and $\mu$ are identified from the dynamic
discrete choice model of car trading.

Given the estimates of $\mu$, $(u_0, u_1, u_2)$ from DNFXP and $(d_0, d_1, d_2)$ from OLS of the driving equation,
the structural parameters $(\alpha_0, \alpha_1, \alpha_2, \gamma_0, \gamma_1, \phi)$ are
identified separately. 

Thus the whole estimation approach has three steps:

1. least squares estimation of the reduced form driving parameters $(d_0, d_1, d_2)$
2. maximum likelihood estimation of the equilibrium model: the reduced form utility
   parameters $(u_0, u_1, u_2)$, the marginal utility of money $\mu$, the transaction costs
   of purchases and sales, and the coefficients of a binary logit model of accident rates,
   $\alpha(a) = \Lambda(\alpha_j + \alpha^a_j a)$
3. backing out the structural parameters

### Consumer heterogeneity and car types specification

The consumers come in eight types:
- by work distance (WD), 
- couple or single, and 
- rich or poor; 

The cars have four types:
- light or heavy, and 
- green (efficient) or brown (otherwise). 

### Estimation results

The driving regression, with driving in 1,000 km per year as the dependent variable and standard errors in parentheses, on 19,635,940 driving periods:

| | | Estimate | |
| :-- | :-- | --: | --: |
| $\gamma_0$ | Intercept | 19.07 | (0.58) |
| $\hat\gamma^a_1 / \phi_\tau$ | Car age | −0.1325 | (0.03) |
| $\hat\gamma^a_2 / \phi_\tau$ | Car age squared | −0.001975 | (0.00) |
| $\hat\gamma_\tau / \phi_\tau$ | Intercept, Low WD, Couple, Rich | 4.43 | (0.70) |
| $\hat\gamma_\tau / \phi_\tau$ | Intercept, Low WD, Single, Poor | −3.752 | (1.24) |
| $\hat\gamma_\tau / \phi_\tau$ | Intercept, Low WD, Single, Rich | −0.0325 | (0.79) |
| $\hat\gamma_\tau / \phi_\tau$ | Intercept, High WD, Couple, Poor | 9.825 | (0.79) |
| $\hat\gamma_\tau / \phi_\tau$ | Intercept, High WD, Couple, Rich | 12.33 | (0.74) |
| $\hat\gamma_\tau / \phi_\tau$ | Intercept, High WD, Single, Poor | 6.436 | (1.52) |
| $\hat\gamma_\tau / \phi_\tau$ | Intercept, High WD, Single, Rich | 12.23 | (1.27) |
| $\hat\gamma_j / \phi_\tau$ | Car dummy: light, green | −1.994 | (0.15) |
| $\hat\gamma_j / \phi_\tau$ | Car dummy: heavy, brown | 4.345 | (0.15) |
| $\hat\gamma_j / \phi_\tau$ | Car dummy: heavy, green | 3.606 | (0.14) |
| $\mu / \phi$ | Price (common) | −7.074 | (0.84) |
| $\hat\mu_\tau / \phi_\tau$ | Price, Low WD, Couple, Rich | −4.111 | (1.02) |
| $\hat\mu_\tau / \phi_\tau$ | Price, Low WD, Single, Poor | 4.732 | (1.84) |
| $\hat\mu_\tau / \phi_\tau$ | Price, Low WD, Single, Rich | 0.2781 | (1.16) |
| $\hat\mu_\tau / \phi_\tau$ | Price, High WD, Couple, Poor | −6.41 | (1.17) |
| $\hat\mu_\tau / \phi_\tau$ | Price, High WD, Couple, Rich | −9.892 | (1.09) |
| $\hat\mu_\tau / \phi_\tau$ | Price, High WD, Single, Poor | −1.714 | (2.29) |
| $\hat\mu_\tau / \phi_\tau$ | Price, High WD, Single, Rich | −9.007 | (1.91) |

The maximum likelihood estimates of the equilibrium model, standard errors in parentheses.

The marginal utility of money $\mu_\tau$, the utility costs of transacting (common, and the
extra cost for consumers without a car), the intercept $u_{\tau,j,0}$ and the age
coefficient $u_{\tau,j,1}$ of the quadratic indirect utility of car ownership:

| Consumer type | $\mu_\tau$ | Transaction cost, common | Transaction cost, no car |
| :-- | --: | --: | --: |
| Low WD, Couple, Poor | 0.1131 (0.0007) | 6.5944 (0.0226) | 1.7899 (0.0029) |
| Low WD, Couple, Rich | 0.1119 (0.0007) | 6.4425 (0.0229) | 1.0719 (0.0030) |
| Low WD, Single, Poor | 0.0941 (0.0007) | 6.5457 (0.0186) | 3.0816 (0.0045) |
| Low WD, Single, Rich | 0.1077 (0.0007) | 6.6036 (0.0212) | 2.5769 (0.0031) |
| High WD, Couple, Poor | 0.1036 (0.0006) | 6.2644 (0.0207) | 0.7793 (0.0036) |
| High WD, Couple, Rich | 0.1155 (0.0007) | 6.4237 (0.0237) | 0.1742 (0.0045) |
| High WD, Single, Poor | 0.0920 (0.0008) | 6.0303 (0.0182) | 2.3383 (0.0066) |
| High WD, Single, Rich | 0.1081 (0.0008) | 6.2786 (0.0220) | 1.7884 (0.0064) |

| $u_{\tau,j,0}$ | light, brown | light, green | heavy, brown | heavy, green |
| :-- | --: | --: | --: | --: |
| Low WD, Couple, Poor | 3.6490 (0.0178) | 3.1132 (0.0179) | 5.1535 (0.0281) | 4.7760 (0.0257) |
| Low WD, Couple, Rich | 4.0324 (0.0172) | 3.4657 (0.0173) | 5.7492 (0.0271) | 5.3478 (0.0247) |
| Low WD, Single, Poor | 2.4042 (0.0191) | 2.1504 (0.0178) | 3.5823 (0.0298) | 3.2068 (0.0274) |
| Low WD, Single, Rich | 3.2454 (0.0175) | 2.8222 (0.0174) | 4.6934 (0.0275) | 4.2829 (0.0252) |
| High WD, Couple, Poor | 3.8821 (0.0164) | 3.4351 (0.0164) | 5.3199 (0.0258) | 5.0561 (0.0235) |
| High WD, Couple, Rich | 4.7620 (0.0182) | 4.3185 (0.0183) | 6.5617 (0.0285) | 6.2375 (0.0260) |
| High WD, Single, Poor | 2.6685 (0.0206) | 2.4290 (0.0188) | 3.7554 (0.0324) | 3.5129 (0.0295) |
| High WD, Single, Rich | 3.5538 (0.0204) | 3.1741 (0.0194) | 4.9946 (0.0319) | 4.6985 (0.0291) |

| $u_{\tau,j,1}$ | light, brown | light, green | heavy, brown | heavy, green |
| :-- | --: | --: | --: | --: |
| Low WD, Couple, Poor | −0.1459 (0.0009) | −0.0922 (0.0011) | −0.2196 (0.0014) | −0.1717 (0.0012) |
| Low WD, Couple, Rich | −0.1586 (0.0009) | −0.0985 (0.0011) | −0.2411 (0.0013) | −0.1953 (0.0012) |
| Low WD, Single, Poor | −0.0984 (0.0009) | −0.0615 (0.0010) | −0.1600 (0.0014) | −0.1128 (0.0012) |
| Low WD, Single, Rich | −0.1308 (0.0009) | −0.0841 (0.0011) | −0.2056 (0.0014) | −0.1544 (0.0011) |
| High WD, Couple, Poor | −0.1390 (0.0008) | −0.0825 (0.0010) | −0.2134 (0.0013) | −0.1696 (0.0011) |
| High WD, Couple, Rich | −0.1642 (0.0009) | −0.1069 (0.0011) | −0.2551 (0.0014) | −0.2074 (0.0012) |
| High WD, Single, Poor | −0.1108 (0.0010) | −0.0707 (0.0010) | −0.1732 (0.0016) | −0.1273 (0.0013) |
| High WD, Single, Rich | −0.1493 (0.0010) | −0.1007 (0.0011) | −0.2208 (0.0016) | −0.1735 (0.0013) |

The accident rates $\alpha(a) = \Lambda(\alpha_j + \alpha^a_j a)$ by car type:

| | light, brown | light, green | heavy, brown | heavy, green |
| :-- | --: | --: | --: | --: |
| Intercept $\alpha_j$ | −5.6248 (0.0119) | −6.0443 (0.0105) | −5.6728 (0.0095) | −5.7826 (0.0088) |
| Age slope $\alpha^a_j$ | 0.1804 (0.0015) | 0.2216 (0.0011) | 0.2020 (0.0009) | 0.2048 (0.0009) |

What the estimates say about preferences and willingness to pay:

- a rich couple with a low work distance is willing to pay 36,040 DKK for a new light
  brown car for a year, against 32,256 DKK for a poor household
- rich households are willing to pay more for any type of car than poor households
- all households prefer heavy cars to light ones and brown cars to green ones, which gives
  the ordering heavy brown ≻ heavy green ≻ light brown ≻ light green
- households with a high work distance are willing to pay more than those with a low one
- couples generally have a higher willingness to pay than singles

The model captures the key features of the Danish households: 
- poor households are much
more likely not to own a car than rich ones,
- couples are more likely to own cars than
singles, and 
- households with a high work distance are relatively more likely to own cars.

For driving: 
- households with a high work distance drive much more than those with a low
one, and 
- more so if they are rich. 

The model implies fuel price elasticities between
−0.10 and −0.60 across households, close to the average elasticity of −0.30 found by
{cite:t}`gillingham2019TaleTwoTails`.

### Goodness of fit

The fit of the aggregate market shares:

```{image} _static/eqb-jpe/fit_marketshares_agg.png
:width: 60%
:align: center
:alt: Model fit, aggregate market shares
```

and of the market shares by household type:

```{image} _static/eqb-jpe/fit_marketshares_45.png
:width: 55%
:align: center
:alt: Model fit, household-specific market shares
```

The actual and predicted probabilities of purchase:

```{image} _static/eqb-jpe/fit_purchases.png
:width: 60%
:align: center
:alt: Actual and predicted probability of purchase
```

and of keeping the car, by car type:

```{image} _static/eqb-jpe/fit_keep.png
:width: 60%
:align: center
:alt: Actual and predicted probability of keeping, by car type
```

The ownership distribution in the data,

```{image} _static/eqb-jpe/pt_dist_data.png
:width: 80%
:align: center
:alt: Observed ownership distribution
```

and the one predicted by the model:

```{image} _static/eqb-jpe/pt_dist_model.png
:width: 80%
:align: center
:alt: Predicted ownership distribution
```

The sorting of households into cars in the data,

```{image} _static/eqb-jpe/holdings_data.png
:width: 90%
:align: center
:alt: Observed sorting
```

and in the model:

```{image} _static/eqb-jpe/holdings_model.png
:width: 90%
:align: center
:alt: Predicted sorting
```

Both the scrappage rates and the equilibrium prices of used cars show **zig-zag
patterns** in the age of the car. The scrappage rates in the data and in the model:

```{image} _static/eqb-jpe/fit_scrap.png
:width: 65%
:align: center
:alt: Scrappage rates, data against model
```

and the secondary market prices predicted by the model:

```{image} _static/eqb-jpe/prices_model.png
:width: 55%
:align: center
:alt: Predicted secondary market prices
```

:::{div}
:class: discussion

- What do you make of the estimated zig-zags in the scrappage rates?
:::

(eqbjpe-counterfactual)=
## Counterfactual: halving the registration tax

Halving the registration tax lowers the price of a new car by 25.6% for the normal car and
by 27.1% for the luxury car. Four scenarios:

1. **baseline**: the calibration under the Danish tax rates of 2008
2. **naive, expected**: a **non-equilibrium** simulation in which the prices of new and
   used cars all fall proportionally, between 25.6% for the cheapest and 26.8% for the most
   expensive car, and the fuel tax rises to keep the total tax revenue **unchanged**
3. **naive, realized**: the same taxes, with the market equilibrium imposed; the revenue is
   no longer neutral
4. **sophisticated policy maker**: **equilibrium and revenue neutral**

The results:

| | Baseline | Naive, expected | Naive, realized | Sophisticated |
| :-- | --: | --: | --: | --: |
| ***Policy choice variables*** | | | | |
| Registration tax (bottom rate) | 1.050 | 0.525 | 0.525 | 0.525 |
| Registration tax (top rate) | 1.800 | 0.900 | 0.900 | 0.900 |
| Fuel tax (share of pump price) | 0.573 | 0.760 | 0.760 | 0.728 |
| ***Exogenous prices*** | | | | |
| Price, light, brown (1000 DKK) | 174.902 | 130.110 | 130.110 | 130.110 |
| Price, light, green (1000 DKK) | 144.551 | 107.532 | 107.532 | 107.532 |
| Price, heavy, brown (1000 DKK) | 299.452 | 216.760 | 216.760 | 216.760 |
| Price, heavy, green (1000 DKK) | 253.397 | 185.508 | 185.508 | 185.508 |
| Fuel price (DKK/l) | 8.322 | 14.824 | 14.824 | 13.054 |
| ***Outcomes*** | | | | |
| Social surplus (1000 DKK) | 8.831 | 10.851 | 7.705 | 9.632 |
| Total tax revenue (1000 DKK) | 9.302 | 9.302 | 7.113 | 9.302 |
| Fuel tax revenue (1000 DKK) | 4.281 | 5.039 | 4.788 | 6.191 |
| Car tax revenue (1000 DKK) | 5.021 | 4.263 | 2.325 | 3.111 |
| Non-CO2 externalities (1000 DKK) | 6.749 | 3.312 | 3.182 | 4.782 |
| Externalities (1000 DKK) | 7.372 | 3.622 | 3.476 | 5.234 |
| Consumer surplus (1000 DKK) | 6.901 | 5.172 | 4.069 | 5.564 |
| CO2 (ton) | 2.148 | 1.069 | 1.016 | 1.559 |
| VKT (1000 km) | 10.858 | 5.329 | 5.118 | 7.693 |
| E(car age) | 6.505 | 2.911 | 4.228 | 5.473 |
| Pr(no car) | 0.367 | 0.546 | 0.548 | 0.415 |

What to read in the table:

- the naive policy maker raises the fuel tax from 57.3% to 76.0% of the pump price and
  expects the total tax revenue to stay at 9,302 DKK; in equilibrium it falls to 7,113 DKK,
  **23.5% less than expected**, because the car tax revenue falls to 2,325 DKK instead of the
  expected 4,263 DKK
- the sophisticated policy maker reaches revenue neutrality with a fuel tax of only 72.8%
- and gets a higher social surplus than in the baseline, 9,632 against 8,831 DKK, with CO2
  emissions down from 2.148 to 1.559 tons

The prices of used cars in the four scenarios:

```{image} _static/eqb-jpe/prices_car4_total_revenue.png
:width: 70%
:align: center
:alt: Equilibrium prices of used cars in the four scenarios
```

The naive scenario assumes a proportional change in all prices, but in equilibrium **used
car prices fall non-proportionally**.


## Counterfactual policy map

Varying the two tax rates over a grid, both measured relative to the baseline level of 1,
traces the Laffer curve of the total tax revenue, from the sales of new cars and from fuel:

```{image} _static/eqb-jpe/laffer_3d.png
:width: 70%
:align: center
:alt: Laffer curves for the registration tax and the fuel tax
```

The CO2 emissions on the same grid:

```{image} _static/eqb-jpe/laffer_3d_co2.png
:width: 70%
:align: center
:alt: CO2 emissions against the registration tax and the fuel tax
```

And the social welfare, excluding CO2:

```{image} _static/eqb-jpe/laffer_3d_welfare_ex_co2.png
:width: 70%
:align: center
:alt: Social welfare excluding CO2 against the registration tax and the fuel tax
```

The level curves through the baseline show the trade-off between CO2 emissions and
welfare:

```{image} _static/eqb-jpe/laffer_contours.png
:width: 90%
:align: center
:alt: Level curves of welfare, CO2 and tax revenue through the baseline
```

In the green region every tax combination has lower CO2 emissions, higher welfare and
higher tax revenue than the baseline: a lower registration tax and a higher fuel tax.
The revenue maximum and the welfare maxima, at carbon prices of 50 and 250 dollars per ton,
all have a much lower registration tax and a higher fuel tax than the baseline.

The conclusions of the paper:

- **theory**: a characterization and a proof of existence of the equilibrium in a
  tractable model of primary and secondary markets
- **application**: a tractable model with transactions, scrappage, heterogeneous consumers
  and cars, and a flexible utility, with 131 parameters estimated from 39 million
  observations in under 15 minutes
- **policy**: the Danish taxes are above the top of the Laffer curve, but a "naive" model
  overestimates the strength of this effect, and could lead to a policy that is bad for the
  tax revenue and for the environment. There is an opportunity to reduce CO2 and to raise
  the tax revenue and social welfare at the same time

:::{div}
:class: discussion

- Why does the car tax revenue fall so much more in equilibrium than the naive policy maker
  expects?
- The sophisticated policy maker needs the equilibrium model for every trial value of the
  fuel tax. Which loop of the estimator does that add, and which does it remove?
:::




````{note} References and additional resources

(eqbjpe_references)=
- 📖 {cite:t}`iruc2` "Equilibrium Trade in Automobiles"
- 📖 {cite:t}`katz1985NetworkExternalitiesCompetition` "Network Externalities, Competition, and Compatibility"
- 📖 {cite:t}`rochet2003PlatformCompetitionTwosided` "Platform Competition in Two-Sided Markets"
- 📖 {cite:t}`rust1985StationaryEquilibriumMarket` "Stationary Equilibrium in a Market for Durable Assets"
- 📖 {cite:t}`manski1983AnalysisEquilibriumAutomobile` "Analysis of equilibrium automobile holdings in Israel with aggregate discrete choice models"
- 📖 {cite:t}`berkovec1985NewCarSales` "New Car Sales and Used Car Stocks: A Model of the Automobile Market"
- 📖 {cite:t}`konishi2002ExistenceStationaryEquilibrium` "Existence of stationary equilibrium in the markets for new and used durable goods"
- 📖 {cite:t}`stolyarov2002TurnoverUsedDurables` "Turnover of Used Durables in a Stationary Equilibrium: Are Older Goods Traded More?"
- 📖 {cite:t}`gavazza2014QuantitativeAnalysisUsedcar` "A Quantitative Analysis of the Used-Car Market"
- 📖 {cite:t}`esteban2007DurablegoodsOligopolySecondary` "Durable-goods oligopoly with secondary markets: the case of automobiles"
- 📖 {cite:t}`parry2005DoesBritainUnited` "Does Britain or the United States Have the Right Gasoline Tax?"
- 📖 {cite:t}`gillingham2019TaleTwoTails` "A tale of two tails: Commuting and the fuel price response in driving"
````
