---
title: 📖 Dynamic entry games
short_title: 📖 Entry games
subtitle: Class 23 — Thursday, November 12
downloads:
  - file: 23_entry_games.md
    title: MyST Markdown
---

Today the agents interact strategically: firms decide whether to be active in a market,
and the profit of each depends on how many rivals are active. We set up the dynamic entry
game of Aguirregabiria and Mira, review how it is estimated, and see why multiple equilibria
are a problem for every one of those estimators.

````{seealso} Key reading

{cite:t}`aguirregabiriaSequentialEstimationDynamic2007` "Sequential Estimation of Dynamic
Discrete Games", *Econometrica* 75(1), 1–53. The dynamic entry game, its equilibrium
conditions in the space of choice probabilities, the NPL estimator for games and the
Chilean retail application of this class all come from this paper.

{cite:t}`egesdal2015EstimatingDynamicDiscretechoice` "Estimating dynamic discrete-choice
games of incomplete information", *Quantitative Economics* 6(3), 567–597. Maximum
likelihood for the dynamic entry game by MPEC, and Monte Carlo evidence on how two-step,
NPL and MPEC fare when the game has multiple equilibria.
````

````{note} Guest speaker: Bertel Schjerning

Prof. Bertel Schjerning, University of Copenhagen,
[schjerning.com](https://www.schjerning.com/), joins the class with a lecture on dynamic
games.

{button}`📄 Slides: Dynamic games (PDF) <https://github.com/bschjerning/dp_ucph/blob/main/4_dynamic_games/2_DynamicGames.pdf>`
````

(games-intro)=
## Dynamic games

In [Class 21](21_eqb_cars.md#eqb-many-agents) the course split the study of many decision
makers into two directions. The equilibrium models of Classes 21 and 22 had many small
agents, each taking an aggregate as given. In a **game** there are a few players, and each
of them takes into account how the others act and react.

The setup:

- several decision makers, the **players**, make discrete choices over time
- each maximizes the expected discounted value of payoffs, as in a single-agent model
- each anticipates the consequences of current actions, and the current and future actions
  of the other players
- the solution of the model is a profile of strategies that are **mutual best responses**:
  an equilibrium

The parts of a dynamic game are those of a single-agent model, with one more question for
each:

1. **State space.** Classified as in single-agent models, but now we also ask what each
   player knows: which state variables are **common knowledge** and which are **private
   information**.
2. **Actions.** As in single-agent models. A policy function is now called a
   **strategy**, and the collection of the strategies of all players is a **strategy
   profile**.
3. **Payoffs.** As the utility in single-agent models, but they depend on the actions of
   all players: this is the **strategic interaction**.
4. **Transitions.** Typically Markovian, and they depend on the actions of all players.
   They describe the beliefs of the players about the future, which may or may not be
   common.

(games-mpe)=
### Markov perfect equilibrium

The equilibrium concept used in almost all empirical work on dynamic games is due to
{cite:t}`maskin1988TheoryDynamicOligopoly` and {cite:t}`maskin2001MarkovPerfectEquilibriumb`.

````{attention} Definition (Markov perfect equilibrium, informally)
A **Markov perfect equilibrium (MPE)** is a profile of strategies that depend only on the
payoff-relevant state, together with the value functions they imply, such that every
player maximizes the expected discounted payoff given the strategies of all other players.
````

Two things hold in an MPE: the Bellman equation of every player holds, and every
player's strategy is the $\arg\max$ of that Bellman equation given the strategies of the
others.

What is new compared with a single agent?

- **Existence** is no longer automatic, and **uniqueness** is the exception, not the rule
- the contraction mapping arguments of [Class 8](8_dp_infinite.md#dptheory) do not apply
  to the joint problem of several players

Ways to solve for an MPE:

- successive approximations on the players' Bellman equations, solving the players'
  problems in turn {cite:p}`pakesComputingMarkovPerfectNash1994,ericsonMarkovPerfectIndustryDynamics1995`
- the algebraic approach of {cite:t}`juddFindingAllPureStrategy2012`, which exploits the
  polynomial structure of the equilibrium equations
- homotopy methods, which follow a path of solutions from an easy problem to the one of
  interest {cite:p}`borkovsky2008UsersGuideSolving`
- methods for special classes of games, such as directional dynamic games and the
  recursive lexicographical search of {cite:t}`rls`

In general, finding all MPE of a game is impossible: no method guarantees it, as
[Class 22](22_eqb_practice.md#eqb-all-equilibria) showed for a much simpler fixed point.

:::{div}
:class: discussion

- Which difficulties do you expect in estimation, compared with a single-agent model?
- What does a player need to know about the other players to compute a best response?
:::

(games-am2007)=
## The dynamic entry game of Aguirregabiria and Mira

Five firms decide every period whether to be active in a market. An illustration of the
data:

| Period | Market size | Firm 1 | Firm 2 | Firm 3 | Firm 4 | Firm 5 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| 1 | 3 | 0 | 1 | 0 | 0 | 1 |
| 2 | 4 | 0 | 1 | 0 | 1 | 1 |
| 3 | 5 | 0 | 1 | 0 | 0 | 1 |
| 4 | 5 | 1 | 1 | 0 | 0 | 0 |
| 5 | 5 | 1 | 1 | 0 | 0 | 1 |
| 6 | 6 | 1 | 1 | 1 | 1 | 1 |

The decisions are simultaneous, and each firm conditions on the market size, the
decisions of all firms in the previous period, and its own private shocks.

The setup:

- discrete time, infinite horizon, $t = 1, 2, \dots$
- $N$ players, $i \in \mathcal{I} = \{1, \dots, N\}$
- the market size $s_t \in \mathcal{S} = \{s_1, \dots, s_L\}$, observed by all players,
  with an exogenous stationary transition $f_{\mathcal{S}}(s_{t+1} \mid s_t)$
- the action $a_{it} \in \mathcal{A} = \{0, 1\}$ of player $i$: inactive or active
- $a_t = (a_{1t}, \dots, a_{Nt})$ the actions of all players, and $a_{-it}$ the actions of
  all players other than $i$

The state variables:

- the common knowledge state $x_t = (s_t, a_{t-1})$: the market size and who was active
  last period
- the private shocks $\varepsilon_{it} = \{\varepsilon_{it}(a)\}_{a \in \mathcal{A}}$,
  i.i.d. extreme value type 1 across actions, players and time; the other players know
  only their distribution $g$

The **conditional independence** assumption of
[Class 9](9_zurcher.md) carries over, with the actions of all players in place of one
decision:

$$
p\big(x_{t+1} = (s', a'), \varepsilon_{i,t+1} \mid x_t = (s, \tilde a), \varepsilon_{it}, a_t\big)
= f_{\mathcal{S}}(s' \mid s) \, \mathbf{1}\{a' = a_t\} \, g(\varepsilon_{i,t+1}) .
$$

Next period's common state is this period's market size draw and this period's actions.

(games-am2007-payoffs)=
### Payoffs

The payoff of player $i$ is a common knowledge part plus the private shock,
$\Pi_i(a_{it}, a_{-it}, x_t; \theta) + \varepsilon_{it}(a_{it})$. The common knowledge part
is zero for an inactive firm, and for an active firm

$$
\Pi_i(1, a_{-it}, x_t; \theta) = \theta^{RS} s_t - \theta^{RN} \log\Big(1 + \sum_{j \neq i} a_{jt}\Big)
- \theta^{FC}_i - \theta^{EC} (1 - a_{i,t-1}) .
$$

The four terms:

- $\theta^{RS} s_t$: variable profit grows with the market size
- $\theta^{RN} \log(1 + \sum_{j \neq i} a_{jt})$: the **strategic interaction**, the loss
  of profit to the active rivals
- $\theta^{FC}_i$: the fixed operating cost of firm $i$
- $\theta^{EC}$: the sunk **entry cost**, paid only by a firm that was inactive last period

Player $i$ chooses the sequence of actions that maximizes

$$
\mathbb{E}\Big[\sum_{\tau = t}^{\infty} \beta^{\tau - t}
\big(\Pi_i(a_{i\tau}, a_{-i\tau}, x_\tau; \theta) + \varepsilon_{i\tau}(a_{i\tau})\big)
\,\Big|\, x_t, \varepsilon_{it}\Big] ,
$$

where the expectation is also over the actions of the other players.

(games-equilibrium-conditions)=
### The equilibrium in the space of choice probabilities

An MPE is characterized in terms of the common knowledge state $x$, exactly as the CCP
methods of [Class 12](12_ccp.md#ccp-choice-probabilities) characterize the solution of a
single-agent model:

- $P_i(a_i \mid x)$, the conditional choice probability of player $i$ at state $x$
- $V_i(x)$, the integrated value function of player $i$ at state $x$
- $\mathbf{P}$ and $\mathbf{V}$, all of them stacked over players, actions and states

From the point of view of player $i$, the other players are part of the environment, and
their choice probabilities are what player $i$ needs to know about them. Given
$\mathbf{P}$, the **expected payoff** of action $a_i$ averages over the actions of the rivals:

$$
\pi_i(a_i \mid x, \theta) = \sum_{a_{-i} \in \mathcal{A}^{N-1}}
\Big[\prod_{j \neq i} P_j(a_j \mid x)\Big] \, \Pi_i(a_i, a_{-i}, x; \theta) ,
$$

and the transition of the common state is

$$
f^{\mathbf{P}}_{\mathcal{X}}\big(x' = (s', a') \mid x\big)
= \Big[\prod_{j=1}^N P_j(a'_j \mid x)\Big] f_{\mathcal{S}}(s' \mid s) .
$$

The equilibrium conditions come in two systems.

**System I: Bellman optimality.** For all players $i$ and states $x$,

$$
V_i(x) = \sum_{a_i \in \mathcal{A}} P_i(a_i \mid x) \big[\pi_i(a_i \mid x, \theta) + e_i^{\mathbf{P}}(a_i \mid x)\big]
+ \beta \sum_{x' \in \mathcal{X}} V_i(x') \, f^{\mathbf{P}}_{\mathcal{X}}(x' \mid x) ,
$$

where $e_i^{\mathbf{P}}(a_i \mid x) = \gamma - \log P_i(a_i \mid x)$ is the expected value
of the shock given that $a_i$ is chosen, with $\gamma$ the Euler–Mascheroni constant. This
is the Hotz–Miller representation of the integrated value function, linear in $V_i$ for
given $\mathbf{P}$.

**System II: Bayes–Nash best responses.** For all players $i$, actions $a$ and states $x$,

$$
P_i(a \mid x) = \frac{\exp\big[v_i(a \mid x)\big]}{\sum_{k \in \mathcal{A}} \exp\big[v_i(k \mid x)\big]},
\qquad
v_i(a \mid x) = \pi_i(a \mid x, \theta) + \beta \sum_{x' \in \mathcal{X}} V_i(x') \, f^{\mathbf{P}}_i(x' \mid x, a) ,
$$

with the transition conditional on the own action,
$f^{\mathbf{P}}_i(x' \mid x, a_i) = f_{\mathcal{S}}(s' \mid s) \, \mathbf{1}\{a'_i = a_i\}
\prod_{j \neq i} P_j(a'_j \mid x)$.

Write the two systems compactly as $\mathbf{V} = \Psi^V(\mathbf{V}, \mathbf{P}, \theta)$
and $\mathbf{P} = \Psi^P(\mathbf{V}, \mathbf{P}, \theta)$.

````{attention} Definition (the set of MPE)
The set of Markov perfect equilibria of the entry game at the parameters $\theta$ is

$$
SOL(\Psi, \theta) = \big\{(\mathbf{P}, \mathbf{V}) \,:\, \mathbf{V} = \Psi^V(\mathbf{V}, \mathbf{P}, \theta),\
\mathbf{P} = \Psi^P(\mathbf{V}, \mathbf{P}, \theta)\big\} .
$$
````

For a single agent the set has exactly one element, the fixed point of a contraction. For
a game it can have several, and their number can change with $\theta$.

:::{div}
:class: discussion

- Which value function is $V_i(x)$: the expected value, the integrated value, or the
  choice-specific value?
- How large is the state space with $N = 5$ firms and $|\mathcal{S}| = 5$ market sizes, and
  how many unknowns do the two systems have?
:::

(games-estimation)=
## Estimation

The data generating process:

- $\theta_0$ are the true structural parameters, and $(\mathbf{P}^0, \mathbf{V}^0)$ is an
  MPE at $\theta_0$
- data $\mathbf{Z} = \{\bar a^{mt}, \bar x^{mt}\}$ come from $M$ independent markets
  observed for $T$ periods: the common state and the actions of all players

The assumptions (Assumption 5 in the paper):

1. if there are multiple MPE, **only one equilibrium is played in the data**
2. the same equilibrium is played outside the sample periods
3. the markets are independent
4. the usual identification conditions hold

(games-ml)=
### Maximum likelihood

With $(\mathbf{P}^\ell(\theta), \mathbf{V}^\ell(\theta))$ the $\ell$-th element of
$SOL(\Psi, \theta)$, the log-likelihood picks the equilibrium that fits the data best:

$$
\mathcal{L}(\mathbf{Z}, \theta) = \max_{\ell} \ \frac{1}{M} \sum_{i=1}^N \sum_{m=1}^M \sum_{t=1}^T
\log P^\ell_i(\bar a^{mt}_i \mid \bar x^{mt}; \theta) .
$$

This is the [nested fixed point estimator of Class 10](10_nfxp.md#nfxp-nested-loop) with
a harder inner loop: for every trial value of $\theta$ it needs **all** equilibria of the
game, and that is generally impossible.

The same estimator can be written as a constrained optimization problem over
$(\theta, \mathbf{P}, \mathbf{V})$, with the equilibrium conditions as constraints:

$$
\max_{\theta, \mathbf{P}, \mathbf{V}} \ \frac{1}{M} \sum_{i, m, t} \log P_i(\bar a^{mt}_i \mid \bar x^{mt})
\quad \text{subject to} \quad
\mathbf{V} = \Psi^V(\mathbf{V}, \mathbf{P}, \theta), \ \mathbf{P} = \Psi^P(\mathbf{V}, \mathbf{P}, \theta) .
$$

This is **MPEC**, mathematical programming with equilibrium constraints. The constraints
need to hold only at the solution, so the game is never solved for all equilibria. It is
the topic of the [next class](24_mpec.md).

(games-two-step)=
### Two-step and NPL estimators

The two-step idea of [Class 12](12_ccp.md#ccp-estimation) carries over directly. Suppose
the equilibrium choice probabilities were known: then the equilibrium conditions would
pin down $\theta$ without solving the game.

**Two-step pseudo maximum likelihood (2S-PML):**

1. estimate the CCPs $\hat{\mathbf{P}}$ from the data, nonparametrically
2. given $\hat{\mathbf{P}}$, System I is linear in $\mathbf{V}$, and for each player

   $$
   \mathbf{V}_i = \big[I - \beta F^{\hat{\mathbf{P}}}_{\mathcal{X}}\big]^{-1}
   \sum_{a \in \mathcal{A}} \hat{\mathbf{P}}_i(a) * \big[\boldsymbol\pi_i(a, \theta) + \mathbf{e}_i^{\hat{\mathbf{P}}}(a)\big]
   \equiv \Gamma_i(\theta, \hat{\mathbf{P}}) ,
   $$

   with $*$ the element-wise product over states; then maximize the pseudo-likelihood
   $\mathcal{L}\big(\mathbf{Z}, \Psi^P(\Gamma(\theta, \hat{\mathbf{P}}), \hat{\mathbf{P}}, \theta)\big)$
   over $\theta$

The second step is a logit with the rivals' behavior and the continuation values computed
from $\hat{\mathbf{P}}$. It is fast, but the first-step estimates of $\mathbf{P}$ enter
nonlinearly, and with the CCPs of many players at many states the finite-sample bias can
be large.

**Nested pseudo-likelihood (NPL)** iterates the second step, as in
[Class 14](14_npl.md) for a single agent: starting from $\tilde{\mathbf{P}}_0 = \hat{\mathbf{P}}$,
for $K = 1, 2, \dots$

1. $\tilde\theta_K = \arg\max_\theta \mathcal{L}\big(\mathbf{Z}, \Psi^P(\Gamma(\theta, \tilde{\mathbf{P}}_{K-1}), \tilde{\mathbf{P}}_{K-1}, \theta)\big)$
2. $\tilde{\mathbf{P}}_K = \Psi^P(\Gamma(\tilde\theta_K, \tilde{\mathbf{P}}_{K-1}), \tilde{\mathbf{P}}_{K-1}, \tilde\theta_K)$

until both converge. A fixed point of the iterations is the **NPL estimator**. For a single
agent the [zero Jacobian property](14_npl.md#npl-zero-jacobian) made the iterations
converge; in a game it does not hold, and the update in step 2 is a best response
iteration on the CCPs.

The other estimators of dynamic games, in brief:

- two-step estimators based on simulation or on the equilibrium conditions directly,
  {cite:t}`bajariEstimatingDynamicModels2007`, {cite:t}`pakesSimpleEstimatorsParameters2007`
  and the asymptotic least squares of {cite:t}`pesendorfer2008AsymptoticLeastSquares`:
  they avoid any solution of the game, at the price of efficiency and small-sample bias
- the efficient pseudo-likelihood (EPL) of {cite:t}`dearing2025EfficientConvergentSequential`,
  which replaces the best response update of NPL by a Newton-type step: more stable, but
  convergence is still not guaranteed
- MPEC {cite:p}`suConstrainedOptimizationApproaches2012,egesdal2015EstimatingDynamicDiscretechoice`:
  full maximum likelihood without solving the game at every trial $\theta$, with local
  optima as the main risk
- the nested recursive lexicographical search of {cite:t}`nrls`, maximum likelihood over
  all equilibria for the class of directional dynamic games

:::{div}
:class: discussion

- 2S-PML and NPL need consistent first-stage CCPs. Which assumption of the data
  generating process makes them consistent?
- Why does NPL not need to solve the game, while maximum likelihood does?
:::

(games-chile)=
## Entry and exit in Chilean retail

{cite:t}`aguirregabiriaSequentialEstimationDynamic2007` estimate the entry game for five
retail industries in Chile:

- the data are a census of Chilean firms for tax purposes from the *Servicio de Impuestos
  Internos*, annual, 1994–1999: firm identifier, five-digit industry, annual sales in 12
  categories, and the *comuna* (county) of the firm
- combined with the annual population of each comuna, 1990–2003
- the markets are comunas, excluding the large cities to avoid interaction between markets
- the market size $s_{mt}$ is the population of comuna $m$ in year $t$, an AR(1) in logs,
  discretized, with the transition matrix estimated from the data
- a separate model for each industry, and a market effect $\omega_m$ added to the profit
  of active firms: discretized unobserved heterogeneity across markets
- $N = 5$ players, $|\mathcal{S}| = 5$, so $|\mathcal{X}| = 5 \times 2^5 = 160$ states;
  $\beta = 0.95$; symmetric equilibria

The descriptive statistics:

```{image} _static/games/AM2007TableVI.png
:width: 80%
:align: center
:alt: Descriptive statistics of five retail industries in Chile, AM 2007 Table VI
```

What stands out:

- there are many restaurants and few gas stations and bookstores
- market concentration is lowest among restaurants
- turnover is very high in all five industries
- survival is more likely for gas stations than in the other industries

What explains these facts? Economies of scale (lower fixed costs for restaurants?), sunk
entry costs (lower for restaurants?), and strategic interaction (is product
differentiation possible for gas stations?). The structural estimates separate the three:

```{image} _static/games/AM_estimates.png
:width: 80%
:align: center
:alt: Structural estimates of the entry game for five industries
```

The raw parameters are not comparable across industries, so the paper normalizes them by
the variable profit of a monopolist in the market of median size, $\theta^{RS} \ln S_{med}$:

```{image} _static/games/AM_normalizedparams.png
:width: 80%
:align: center
:alt: Normalized parameters for cross-industry comparison
```

The four normalized quantities:

- $\theta^{FC} / (\theta^{RS} \ln S_{med})$: fixed operating costs relative to a monopolist's
  variable profit in the median market
- $\theta^{EC} / (\theta^{RS} \ln S_{med})$: sunk entry costs relative to the same
- $\theta^{RN} \ln 2 / (\theta^{RS} \ln S_{med})$: the percentage loss of variable profit
  per firm when a monopoly becomes a duopoly
- $\sigma_\omega^2 / (\theta_{RS}^2 \operatorname{var}(\ln S) + \sigma_\omega^2 + 1)$: the
  share of the cross-market variation in monopoly profits explained by the unobserved
  market type

The findings:

- fixed operating costs are a large part of total profits, and economies of scale are
  smallest for restaurants, which explains their large number
- the strategic interaction is significant in all industries, but small among restaurants
  and bookstores, probably because of product differentiation
- sunk entry costs are significant in all industries, but smaller than the annual fixed
  costs; they are largest for gas stations, which explains their low turnover

:::{div}
:class: discussion

- The model assumes one equilibrium is played in all markets of an industry. How would
  you test it?
- Which counterfactual would you run with these estimates?
:::

(games-multiplicity)=
## Why multiple equilibria are a problem

All the estimators above assume that one equilibrium is played in the data, and each of
them runs into trouble when the model has several:

- **maximum likelihood** needs all equilibria at every trial $\theta$, and as the number
  of equilibria changes with $\theta$, the likelihood can jump
- **MPEC** avoids solving for all equilibria, but the optimizer may converge to an
  equilibrium that is not the one in the data
- **two-step estimators** do not solve the game at all, but they inherit the bias of the
  first stage
- **NPL** updates the CCPs by best response iterations: [successive approximations of
  Class 22](22_eqb_practice.md#eqb-sa-multiple), which converge only to equilibria that are
  stable under best responses


(mpec-egesdal)=
## Monte Carlo evidence

{cite:t}`egesdal2015EstimatingDynamicDiscretechoice` compare the estimators on the full
game of Aguirregabiria and Mira:

- $N = 5$ firms with fixed costs $\theta^{FC}_0 = (1.9, 1.8, 1.7, 1.6, 1.5)$ and entry cost
  $\theta^{EC}_0 = 1$, the market size transition of the code above
- **experiment 1**: $|\mathcal{S}| = 5$, with $(\theta^{RS}, \theta^{RN}) = (1, 2)$ in case 3
  and $(2, 4)$ in case 4
- **experiment 2**: $|\mathcal{S}| = 10$ and $15$, cases 5 and 6, with 4800 constraints and
  4808 unknowns in the largest
- $M = 400$ markets, $T = 10$ periods; 100 data sets per case in experiment 1 and 50 in
  experiment 2; MPEC from 10 and 5 starting points; NPL and its modification NPL-$\Lambda$
  with at most 100 iterations
- MPEC is solved by KNITRO through AMPL, with exact derivatives and sparse matrices

The share of data sets for which each estimator converged:

```{image} _static/games/dgame_converged.png
:width: 80%
:align: center
:alt: Percentage of data sets solved by MLE, 2S-PML, NPL and NPL-Lambda
```

The average time per run:

```{image} _static/games/dgame_cputime.png
:width: 80%
:align: center
:alt: Solve time per run for the four estimators
```

The estimates in experiment 1:

```{image} _static/games/dgame_exp1.png
:width: 90%
:align: center
:alt: Monte Carlo estimates in experiment 1, cases 3 and 4
```

and in experiment 2:

```{image} _static/games/dgame_exp2.png
:width: 90%
:align: center
:alt: Monte Carlo estimates in experiment 2, 10 and 15 market sizes
```

The findings:

- maximum likelihood by MPEC converges for every data set and recovers the parameters,
  at a cost that grows with the state space but stays within minutes
- 2S-PML always converges and is the fastest, but $\theta^{RN}$ and $\theta^{RS}$ are badly
  biased, in every case
- NPL converges for about half of the data sets in cases 3 and 5 and for none in cases 4
  and 6; NPL-$\Lambda$, which damps the best response update, does better, but is not
  reliable either
- open questions: MPEC for larger state spaces, and its reliability when the data come
  from a model with many equilibria, which the directional dynamic games of the next
  classes address



**We will formally set up and play with MPEC estimation in the next class.**


````{note} References and additional resources

(entry-games-references)=
- 📖 {cite:t}`aguirregabiriaSequentialEstimationDynamic2007` "Sequential Estimation of Dynamic Discrete Games"
- 📖 {cite:t}`pesendorfer2010SequentialEstimationDynamic` "Sequential Estimation of Dynamic Discrete Games: A Comment"
- 📖 {cite:t}`egesdal2015EstimatingDynamicDiscretechoice` "Estimating dynamic discrete-choice games of incomplete information"
- 📖 {cite:t}`aguirregabiriaImposingEquilibriumRestrictions2021` "Imposing equilibrium restrictions in the estimation of dynamic discrete games"
- 📖 {cite:t}`doraszelski2010TheoryRegularMarkov` "A theory of regular Markov perfect equilibria in dynamic stochastic games: Genericity, stability, and purification"
- 📖 {cite:t}`toivanen2005MarketStructureEntry` "Market Structure and Entry: Where's the Beef?"
- 📄 Jason Blevins and Minhae Kim, "Nested Pseudo-Likelihood Estimation of Dynamic Discrete Choice Games: A Replication and Retrospective" [jblevins.org/research/am_2007](https://jblevins.org/research/am_2007)
````
