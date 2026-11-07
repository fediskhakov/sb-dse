---
title: "🔬 Practice: MPEC for static and dynamic entry games"
short_title: "🔬 MPEC"
subtitle: Class 24 — Tuesday, November 17
downloads:
  - file: 24_mpec.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

<span class="tag wip">🚧 🚧 🚧 Not the final version 🚧 🚧 🚧</span>

Maximum likelihood for a game needs all equilibria at every trial value of the parameters,
as [Class 23](23_entry_games.md#games-static-ml) showed. Today we estimate the static and
the dynamic entry games by MPEC, which writes the equilibrium conditions as constraints
of the likelihood maximization and never solves the game for all equilibria.

````{seealso} Key reading

{cite:t}`suConstrainedOptimizationApproaches2012` "Constrained Optimization Approaches to
Estimation of Structural Models", *Econometrica* 80(5), 2213–2230. The MPEC formulation of
a structural estimator, its equivalence with the nested fixed point algorithm, and the
idea this class applies to the static and dynamic entry games.
````

````{hint} Running the code for this lecture
:class: dropdown

The code for this class is in the folder `session24-nov17/` of the course **code
repository**. The notebook `mpec.ipynb` has all the code of this page.

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

```{code-cell} python3
import time, warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize, root
warnings.simplefilter('ignore', RuntimeWarning)               # log(0) at trial points the optimizers reject
```

(mpec-idea)=
## Estimation as constrained optimization

Every structural estimator of the course maximizes a criterion that depends on the
parameters $\theta$ and on endogenous objects $\sigma$ the model determines: the expected
value function of the bus model, the equilibrium prices of the car market, the choice
probabilities of a game. The model ties them together by a system of equations
$G(\theta, \sigma) = 0$.

**MPEC**, mathematical programming with equilibrium constraints
{cite:p}`suConstrainedOptimizationApproaches2012`, maximizes over both:

$$
\max_{\theta, \sigma} \ \mathcal{L}(\theta, \sigma; \text{data}) \quad \text{subject to} \quad G(\theta, \sigma) = 0 .
$$

The [nested fixed point algorithm of Class 10](10_nfxp.md#nfxp-nested-loop) solves the same
problem by eliminating $\sigma$: when $G(\theta, \sigma) = 0$ has a unique solution
$\sigma(\theta)$, the constrained problem reduces to $\max_\theta \mathcal{L}(\theta, \sigma(\theta))$.
Rust's NFXP manual describes the algorithm exactly this way. Both formulations have the same
solutions; they differ in how the computer gets there.

What changes with MPEC:

- the constraints have to hold only at the solution, not at every trial value of $\theta$:
  no inner loop, and no inner loop tolerance
- the objective and the constraints are smooth in $(\theta, \sigma)$ even when
  $\sigma(\theta)$ is not, for example when the number of equilibria changes with $\theta$
- the number of unknowns grows from $\dim \theta$ to $\dim \theta + \dim \sigma$, so MPEC
  needs a solver for large constrained problems, such as KNITRO or IPOPT, with exact
  derivatives and the **sparsity pattern** of the constraint Jacobian

Where it has been applied:

- the bus engine model {cite:p}`suConstrainedOptimizationApproaches2012`; with the
  Newton–Kantorovich inner loop of [Class 9](9_zurcher.md), NFXP is as fast as MPEC
  there {cite:p}`ecma_comment`
- random coefficients demand {cite:p}`dube2012ImprovingNumericalPerformance`: the inner loop
  of BLP inverts market shares by a contraction that is slow, and a loose inner tolerance
  makes the outer gradient wrong and the optimizer stop at a point that is not a maximum;
  MPEC adds the demand shocks as unknowns and the share equations as sparse constraints
- games {cite:p}`suEstimatingDiscretechoiceGames2014,egesdal2015EstimatingDynamicDiscretechoice`,
  where the inner loop of maximum likelihood would need all equilibria

:::{div}
:class: discussion

- When does eliminating $\sigma$ make the problem easier, and when harder?
- What does a constrained optimizer do at an intermediate iterate where the constraints
  are violated? Is such an iterate meaningful economically?
:::

(mpec-static)=
## MPEC for the static entry game

The game of [Class 23](23_entry_games.md#games-static): two firms, entry payoff
$[\alpha + p_{-j}(\beta - \alpha)] x_j$, logit best responses $\Psi_a$, $\Psi_b$. The model
object holds the parameters and the best responses; `equilibria` finds all equilibria with
the second-order best response, the one-dimensional trick of
[Class 22](22_eqb_practice.md#eqb-network-strength).

```{code-cell} python3
class entry_game:
    '''Static entry game of Su (2014): two firms, entry payoff [alpha + p_other (beta - alpha)] x'''

    def __init__(self, alpha=5.0, beta=-11.0, x=(0.52, 0.22)):
        self.alpha, self.beta = alpha, beta                   # monopoly and duopoly profit per unit of type
        self.x = np.asarray(x, dtype=float)                   # observed types of firms a and b

    def br(self, p_other, x, theta=None):
        '''Logit best response: entry probability given the belief p_other about the rival'''
        a, b = (self.alpha, self.beta) if theta is None else theta
        return 1 / (1 + np.exp(-a * x + p_other * x * (a - b)))

    def Psi(self, p, theta=None):
        '''Equilibrium mapping (p_a, p_b) -> (Psi_a(p_b), Psi_b(p_a))'''
        return np.array([self.br(p[1], self.x[0], theta), self.br(p[0], self.x[1], theta)])

    def equilibria(self, theta=None, n=1001):
        '''All equilibria, as roots of Psi_a(Psi_b(p_a)) - p_a bracketed on a grid and refined by brentq'''
        g = lambda pa: self.br(self.br(pa, self.x[1], theta), self.x[0], theta) - pa
        grid = np.linspace(0, 1, n)
        gv = g(grid)                                          # vectorized: the whole grid at once
        pa = [brentq(g, lo, hi, xtol=1e-14)
              for lo, hi, glo, ghi in zip(grid[:-1], grid[1:], gv[:-1], gv[1:]) if glo * ghi < 0]
        return np.array([[q, self.br(q, self.x[1], theta)] for q in pa])

game = entry_game()
eq = game.equilibria()
print('equilibria (p_a, p_b):', eq.round(4), sep='\n')
```

The data are $N = 1000$ plays of one equilibrium, and the likelihood depends on them only
through the number of entries of each firm:

```{code-cell} python3
def simulate(p, N=1000, seed=0):
    '''N plays of the game in which the firms enter with probabilities p = (p_a, p_b)'''
    rng = np.random.default_rng(seed)
    return (rng.uniform(size=(N, 2)) < p).astype(int)

def loglik(p, d):
    '''Log-likelihood of entry data d (N x 2) given the entry probabilities p'''
    n1 = d.sum(axis=0)                                        # number of entries of each firm
    return np.sum(n1 * np.log(p) + (d.shape[0] - n1) * np.log(1 - p))

data = [simulate(eq[k], seed=k) for k in range(3)]            # one data set from each equilibrium
for k, d in enumerate(data):
    print(f'equilibrium {k + 1}: true p = {eq[k].round(3)}, entry frequencies {d.mean(axis=0)}')
```

The NFXP estimator of Class 23 for comparison: for every trial $(\alpha, \beta)$ it finds
all equilibria and keeps the best one. The likelihood jumps, so it is maximized by
Nelder–Mead, which needs no derivatives:

```{code-cell} python3
def nfxp(d, game, theta0=(5, -11)):
    '''NFXP: maximize the likelihood of the best fitting equilibrium at every trial theta'''
    ll = lambda th: max(loglik(np.clip(p, 1e-12, 1 - 1e-12), d) for p in game.equilibria(th))
    return minimize(lambda th: -ll(th), theta0, method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-8))

for k, d in enumerate(data):
    r = nfxp(d, game)
    print(f'equilibrium {k + 1}: NFXP (alpha, beta) = {r.x.round(3)}, log-likelihood {-r.fun:.2f}')
```

The MPEC formulation of the same estimator has four unknowns, $(\alpha, \beta, p_a, p_b)$:

$$
\max_{\alpha, \beta, p_a, p_b} \ \log \mathcal{L}(p_a, p_b; X)
\quad \text{subject to} \quad
p_a = \Psi_a(p_b, x_a; \alpha, \beta), \quad p_b = \Psi_b(p_a, x_b; \alpha, \beta), \quad 0 \leqslant p_a, p_b \leqslant 1 .
$$

The objective depends on the probabilities only, and its gradient is that of two binary
logits. SciPy's SLSQP handles equality constraints and bounds:

```{code-cell} python3
def mpec_static(d, game, z0):
    '''MPEC for the static entry game: maximize over z = (alpha, beta, p_a, p_b) subject to p = Psi(p, theta)'''
    n1 = d.sum(axis=0); n0 = d.shape[0] - n1
    obj = lambda z: -np.sum(n1 * np.log(z[2:]) + n0 * np.log(1 - z[2:]))
    jac = lambda z: np.r_[0, 0, -(n1 / z[2:] - n0 / (1 - z[2:]))]   # the likelihood does not depend on theta directly
    con = {'type': 'eq', 'fun': lambda z: z[2:] - game.Psi(z[2:], z[:2])}
    bounds = [(None, None)] * 2 + [(1e-8, 1 - 1e-8)] * 2      # probabilities stay inside (0, 1)
    return minimize(obj, z0, jac=jac, constraints=[con], bounds=bounds, method='SLSQP',
                    options=dict(ftol=1e-12, maxiter=500))

starts = ([1, -1, 0.5, 0.5], [8, -5, 0.2, 0.8], [5, -11, 0.5, 0.5])
for k, d in enumerate(data):
    for z0 in starts:
        r = mpec_static(d, game, np.array(z0, dtype=float))
        print(f'eq {k + 1}, start {z0}: {"converged" if r.success else "failed"} in {r.nit:2d} iterations, '
              f'(alpha, beta, p_a, p_b) = {r.x.round(3)}, log-likelihood {-r.fun:.2f}')
```

MPEC reaches the NFXP estimates from every starting point, for all three equilibria,
including the unstable equilibrium 2, in at most 40 iterations. It never solved the game:
the probabilities and the parameters moved together, and the constraints were satisfied
only at the end.

The Monte Carlo results of {cite:t}`suEstimatingDiscretechoiceGames2014` for MPEC, 1000
samples from each equilibrium, the same design as the NFXP results of Class 23:

```{image} _static/games/mc_eqb1_mpec.png
:width: 80%
:align: center
:alt: Monte Carlo distribution of the MPEC estimates, equilibrium 1
```

```{image} _static/games/mc_eqb2_mpec.png
:width: 80%
:align: center
:alt: Monte Carlo distribution of the MPEC estimates, equilibrium 2
```

```{image} _static/games/mc_eqb3_mpec.png
:width: 80%
:align: center
:alt: Monte Carlo distribution of the MPEC estimates, equilibrium 3
```

The distributions are those of NFXP, centered at the truth for every equilibrium. Unlike
NPL in [Class 23](23_entry_games.md#games-static-two-step), MPEC does not care whether the
equilibrium in the data is stable under best responses.

:::{div}
:class: discussion

- MPEC found equilibrium 2 although it is unstable. What plays the role of the best
  response iterations in MPEC, and why does stability not matter for it?
- What would MPEC do if started at the probabilities of equilibrium 1 with data from
  equilibrium 3?
:::

(mpec-markets)=
## Many markets

The design of {cite:t}`suEstimatingDiscretechoiceGames2014` with many markets:

- 25 markets with the types $(x^m_a, x^m_b)$ on a grid of 5 points in $[0.12, 0.87]$ for
  each firm, and the same $(\alpha_0, \beta_0) = (5, -11)$
- in each market the equilibrium that generates the data is drawn at random, so different
  markets play different equilibria
- $T = 50$ observations per market

Most markets have three equilibria. Over a finer grid of types:

```{image} _static/games/neqb.png
:width: 70%
:align: center
:alt: Number of equilibria as a function of the types of the two firms
```

The NFXP likelihood is a sum over markets, so its inner loop picks the best equilibrium
in each market separately. The MPEC formulation has the probabilities of every market
as unknowns, $2 + 2M$ of them:

$$
\max_{\alpha, \beta, \{p^m_a, p^m_b\}} \ \sum_{m=1}^M \log \mathcal{L}(p^m_a, p^m_b; X^m)
\quad \text{subject to} \quad
p^m_a = \Psi_a(p^m_b, x^m_a; \alpha, \beta), \ p^m_b = \Psi_b(p^m_a, x^m_b; \alpha, \beta), \ m = 1, \dots, M .
$$

```{code-cell} python3
xs = np.linspace(0.12, 0.87, 5)
markets = [entry_game(x=(xa, xb)) for xa in xs for xb in xs]  # 25 markets, one game object each
print('equilibria per market:', [len(m.equilibria()) for m in markets])

def simulate_markets(markets, T=50, seed=0):
    '''Data from M markets, each playing a randomly chosen equilibrium T times: (P, d) with d of shape T x M x 2'''
    rng = np.random.default_rng(seed)
    P = np.array([(e := m.equilibria())[rng.integers(len(e))] for m in markets])
    return P, (rng.uniform(size=(T, len(markets), 2)) < P).astype(int)

def nfxp_markets(d, markets, theta0=(5, -11)):
    '''NFXP with many markets: the best fitting equilibrium in every market'''
    ll = lambda th: sum(max(loglik(np.clip(p, 1e-12, 1 - 1e-12), d[:, m]) for p in g.equilibria(th))
                        for m, g in enumerate(markets))
    return minimize(lambda th: -ll(th), theta0, method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-8))
```

The MPEC estimator stacks the constraints of all markets:

```{code-cell} python3
def mpec_markets(d, markets, z0):
    '''MPEC with many markets: unknowns z = (alpha, beta, p^1_a, p^1_b, ..., p^M_a, p^M_b)'''
    M = len(markets)
    n1 = d.sum(axis=0); n0 = d.shape[0] - n1                  # entries by market and firm, M x 2
    P = lambda z: z[2:].reshape(M, 2)
    obj = lambda z: -np.sum(n1 * np.log(P(z)) + n0 * np.log(1 - P(z)))
    jac = lambda z: np.r_[0, 0, -(n1 / P(z) - n0 / (1 - P(z))).ravel()]
    con = lambda z: np.concatenate([P(z)[m] - g.Psi(P(z)[m], z[:2]) for m, g in enumerate(markets)])
    bounds = [(None, None)] * 2 + [(1e-8, 1 - 1e-8)] * 2 * M
    return minimize(obj, z0, jac=jac, constraints=[{'type': 'eq', 'fun': con}], bounds=bounds,
                    method='SLSQP', options=dict(ftol=1e-10, maxiter=1000))

t0, res = time.time(), []
for rep in range(40):
    P0, d = simulate_markets(markets, seed=rep)
    a = nfxp_markets(d, markets)
    z0 = np.r_[5, -11, d.mean(axis=0).ravel().clip(0.01, 0.99)]   # start at the truth and the frequencies
    b = mpec_markets(d, markets, z0)
    res.append([*a.x, *b.x[:2], np.abs(b.x[2:].reshape(-1, 2) - P0).max()])
res = np.array(res)
print(f'{len(res)} samples in {time.time() - t0:.0f} sec')
print(f'NFXP: mean (alpha, beta) {res[:, :2].mean(axis=0).round(3)}, worst |beta - beta0| {np.abs(res[:, 1] + 11).max():.2f}')
print(f'MPEC: mean (alpha, beta) {res[:, 2:4].mean(axis=0).round(3)}, worst |beta - beta0| {np.abs(res[:, 3] + 11).max():.2f}')
print(f'MPEC: samples with an equilibrium probability off by more than 0.1: {np.sum(res[:, 4] > 0.1)}')
```

```{code-cell} python3
:tags: [hide-input]

fig, ax = plt.subplots(2, 1, figsize=(12, 8), layout='constrained')
for j, (name, tv) in enumerate((('alpha', 5), ('beta', -11))):
    bins = np.linspace(res[:, [j, j + 2]].min(), res[:, [j, j + 2]].max(), 40)
    ax[j].hist(res[:, j], bins=bins, color='tab:blue', alpha=0.6, label='NFXP')
    ax[j].hist(res[:, j + 2], bins=bins, color='tab:red', alpha=0.6, label='MPEC')
    ax[j].axvline(tv, color='black', lw=1.5)
    ax[j].set(xlabel=f'estimate of {name}', ylabel='samples'); ax[j].legend()
plt.show()
```

Both estimators start at the true parameters. NFXP stays close to them in every sample;
MPEC lands in a **local maximum** in some samples, with the probabilities of some markets
at an equilibrium other than the one that generated the data. The results of
{cite:t}`suEstimatingDiscretechoiceGames2014`, 25 markets with $T = 50$, and 2 markets with
$T = 625$:

```{image} _static/games/mc_t50m25_mpec_true.png
:width: 80%
:align: center
:alt: Monte Carlo distribution of the MPEC estimates with 25 markets
```

```{image} _static/games/mc_t625m2_mpec_true.png
:width: 80%
:align: center
:alt: Monte Carlo distribution of the MPEC estimates with 2 markets
```

The trade-off between the two estimators:

- **NFXP** has 2 unknowns and can tell which equilibrium each market plays, but it needs
  all equilibria in every market at every trial value: easy here, very hard in a dynamic
  game
- **MPEC** has $2 + 2M$ unknowns and never solves the game, but every market adds
  equilibria in which the optimizer can get stuck, and with many markets the local maxima
  multiply
- both are full information maximum likelihood: the same estimator, computed differently

The implementations here use SLSQP with finite difference constraint Jacobians. Su used
AMPL and KNITRO, with exact derivatives and sparse linear algebra, and more starting points.

:::{div}
:class: discussion

- How would you choose starting values for MPEC with many markets?
- With $M = 25$ markets of three equilibria each, how many combinations of equilibria are
  there? What does that say about the number of local maxima?
:::

(mpec-dynamic)=
## MPEC for the dynamic entry game

The MPEC formulation of the [dynamic entry game](23_entry_games.md#games-equilibrium-conditions)
of Aguirregabiria and Mira has the choice probabilities $\mathbf{P}$ and the value
functions $\mathbf{V}$ of all players as unknowns, and Systems I and II as constraints:

$$
\max_{\theta, \mathbf{P}, \mathbf{V}} \ \sum_{i=1}^N \sum_{m=1}^M \sum_{t=1}^T \log P_i(\bar a^{mt}_i \mid \bar x^{mt})
\quad \text{subject to} \quad
\mathbf{V} = \Psi^V(\mathbf{V}, \mathbf{P}, \theta), \quad
\mathbf{P} = \Psi^P(\mathbf{V}, \mathbf{P}, \theta) .
$$

The size of the problem with $N$ firms and $L$ market sizes: $|\mathcal{X}| = L \cdot 2^N$
states, and $2 N |\mathcal{X}|$ constraints. The game of the paper, with $N = 5$ and
$L = 5$, has 160 states and 1600 constraints; with $L = 15$ it has 4800.

The code below uses $N = 3$ firms and $L = 5$ market sizes, 40 states, so that SciPy can
handle it in class. The fixed cost is common to all firms, and
$\theta = (\theta^{RS}, \theta^{RN}, \theta^{FC}, \theta^{EC})$. The first part of the
model object enumerates the action profiles and computes the expected payoffs and the
transitions given $\mathbf{P}$:

```{code-cell} python3
class dynamic_entry:
    '''Dynamic entry game of Aguirregabiria and Mira (2007): N symmetric firms, L market sizes'''

    def __init__(self, N=3, L=5, beta=0.95, theta=(1.0, 2.0, 1.7, 1.0)):
        self.N, self.L, self.beta = N, L, beta
        self.theta = np.array(theta, dtype=float)             # (RS, RN, FC, EC)
        self.A = np.array([[(k >> i) & 1 for i in range(N)] for k in range(2**N)])   # action profiles, 2^N x N
        self.nA, self.X = 2**N, L * 2**N                      # states x = (s, a_prev): s major, a_prev minor
        self.s = np.repeat(np.arange(1, L + 1), self.nA)      # market size at each state
        self.aprev = np.tile(self.A, (L, 1))                  # last period's actions at each state, X x N
        self.fS = 0.6 * np.eye(L) + 0.2 * np.eye(L, k=1) + 0.2 * np.eye(L, k=-1)
        self.fS[0, 0] = self.fS[-1, -1] = 0.8                 # market size transition of Egesdal, Lai and Su
        self.rivals = self.A.sum(axis=1)[:, None] - self.A    # active rivals of firm i in each profile, 2^N x N

    def profiles(self, P):
        '''Probability of every action profile at every state, X x 2^N, given P (N x X) the entry probabilities'''
        return np.prod(np.where(self.A[None] == 1, P.T[:, None, :], 1 - P.T[:, None, :]), axis=2)

    def payoff(self, P, theta):
        '''Expected payoff of entry pi_i(1|x), N x X, averaging over the rivals; staying out pays zero'''
        RS, RN, FC, EC = theta
        Elog = self.profiles(P) @ np.log(1 + self.rivals)     # E log(1 + active rivals), X x N
        return (RS * self.s[:, None] - RN * Elog - FC - EC * (1 - self.aprev)).T

    def transition(self, P):
        '''Transition of the common state F(x'|x), X x X: next market size times next action profile'''
        return (self.fS[self.s - 1][:, :, None] * self.profiles(P)[:, None, :]).reshape(self.X, self.X)
```

The second part holds the two equilibrium systems and the simulator. $\Gamma$ solves System
I for $\mathbf{V}$ given $\mathbf{P}$, a linear system, as in the
[two-step estimator of Class 23](23_entry_games.md#games-two-step):

```{code-cell} python3
class dynamic_entry(dynamic_entry):

    def flow(self, P, theta):
        '''Expected flow payoff including the shocks: sum_a P_i(a|x) [pi_i(a|x) + gamma - log P_i(a|x)]'''
        return P * (self.payoff(P, theta) + np.euler_gamma - np.log(P)) + (1 - P) * (np.euler_gamma - np.log(1 - P))

    def Psi_V(self, V, P, theta):
        '''System I, the Bellman equations: right hand side, N x X'''
        return self.flow(P, theta) + self.beta * V @ self.transition(P).T

    def Psi_P(self, V, P, theta):
        '''System II, the logit best responses given the rivals' P and the own continuation values'''
        EV = np.einsum('xt,ita->ixa', self.fS[self.s - 1], V.reshape(self.N, self.L, self.nA))  # E_s' V_i(s', a')
        Pa = self.profiles(P)
        cont = [np.einsum('xa,ixa->ix', Pa, EV * (self.A == ai).T[:, None, :]) / (P if ai else 1 - P)
                for ai in (0, 1)]                             # E V_i(x') given own action a_i, rivals from P
        return 1 / (1 + np.exp(-(self.payoff(P, theta) + self.beta * (cont[1] - cont[0]))))

    def Gamma(self, P, theta):
        '''V given P: System I is linear in V'''
        return np.linalg.solve(np.eye(self.X) - self.beta * self.transition(P), self.flow(P, theta).T).T

    def simulate(self, P, M=400, T=10, seed=0):
        '''Data from M markets over T periods: counts n[i, x, a] of firm i choosing a at state x'''
        rng = np.random.default_rng(seed)
        s, a = rng.integers(self.L, size=M), np.zeros((M, self.N), dtype=int)   # all firms start out of the market
        n = np.zeros((self.N, self.X, 2))
        for t in range(T):
            x = s * self.nA + a @ (2 ** np.arange(self.N))    # index of the state (s, a_prev)
            a = (rng.uniform(size=(M, self.N)) < P[:, x].T).astype(int)
            for i in range(self.N): np.add.at(n[i], (x, a[:, i]), 1)
            s = np.array([rng.choice(self.L, p=self.fS[k]) for k in s])
        return n
```

To generate data we need one equilibrium at the true parameters. Successive approximations
on the best responses, $\mathbf{P} \leftarrow \Psi^P(\Gamma(\mathbf{P}), \mathbf{P})$, are the
natural solver; Newton's method on the same fixed point is the robust one:

```{code-cell} python3
def solve_mpe(g, method='newton', tol=1e-12, maxiter=2000):
    '''An MPE of the game g at its true parameters, by successive approximations or Newton (scipy root)'''
    BR = lambda P: g.Psi_P(g.Gamma(P, g.theta), P, g.theta)   # best response to P, with V solved from System I
    P = np.full((g.N, g.X), 0.5)
    if method == 'newton':
        r = root(lambda p: p - BR(p.reshape(g.N, g.X)).ravel(), P.ravel(), method='hybr', tol=tol)
        return r.x.reshape(g.N, g.X)
    for k in range(maxiter):
        P1 = BR(P)
        if np.max(np.abs(P1 - P)) < tol: return P1
        P = P1
    raise RuntimeError(f'Successive approximations failed to converge in {maxiter} iterations')

def br_radius(g, P, h=1e-6):
    '''Spectral radius of the Jacobian of the best response map at P, by central differences'''
    BR = lambda p: g.Psi_P(g.Gamma(p.reshape(g.N, g.X), g.theta), p.reshape(g.N, g.X), g.theta).ravel()
    J = np.array([(BR(P.ravel() + h * e) - BR(P.ravel() - h * e)) / (2 * h) for e in np.eye(P.size)]).T
    return np.max(np.abs(np.linalg.eigvals(J)))
```

Two cases from {cite:t}`egesdal2015EstimatingDynamicDiscretechoice`: the strength of the
market size and of the competition effect doubled, $(\theta^{RS}, \theta^{RN}) = (1, 2)$
and $(2, 4)$:

```{code-cell} python3
games = {case: dynamic_entry(theta=(*rsrn, 1.7, 1.0)) for case, rsrn in (('case 3', (1, 2)), ('case 4', (2, 4)))}
P_true = {}
for case, g in games.items():
    P_true[case] = solve_mpe(g, 'newton')
    V = g.Gamma(P_true[case], g.theta)
    resid = np.abs(g.Psi_P(V, P_true[case], g.theta) - P_true[case]).max()
    print(f'{case}: theta = {g.theta}, residual {resid:.1e}, spectral radius of the best response map {br_radius(g, P_true[case]):.3f}')
    try:
        solve_mpe(g, 'sa'); print('  successive approximations converge')
    except RuntimeError as e:
        print('  ', e)
```

In case 4 the equilibrium is **unstable under best responses**: the spectral radius is
above one, and successive approximations, which converged in case 3, never find it. This
is equilibrium 2 of the static game again, now in 120 dimensions.

The three estimators of the game: 2S-PML and NPL from
[Class 23](23_entry_games.md#games-two-step), and MPEC with the $3 \times 40$ probabilities
and the $3 \times 40$ values as unknowns, 244 in total:

```{code-cell} python3
def ll_counts(P, n):
    '''Log-likelihood from the counts n[i, x, a]'''
    return np.sum(n[..., 1] * np.log(P) + n[..., 0] * np.log(1 - P))

def pml(n, Phat, g, theta0):
    '''Pseudo-likelihood step: theta maximizing the likelihood of Psi_P(Gamma(theta, Phat), Phat, theta)'''
    f = lambda th: -ll_counts(np.clip(g.Psi_P(g.Gamma(Phat, th), Phat, th), 1e-10, 1 - 1e-10), n)
    return minimize(f, theta0, method='BFGS').x

def npl(n, Phat, g, theta0, K=100, tol=1e-8):
    '''NPL iterations: pseudo-likelihood step, then a best response update of the CCPs'''
    P, th = Phat, theta0
    for k in range(K):
        th = pml(n, P, g, th)
        P1 = g.Psi_P(g.Gamma(P, th), P, th)
        if np.max(np.abs(P1 - P)) < tol: return th, k + 1
        P = P1
    return th, None                                           # no convergence in K iterations

def mpec(n, Phat, g, theta0):
    '''MPEC: maximize the likelihood over (theta, P, V) subject to Systems I and II'''
    N, X = g.N, g.X
    split = lambda z: (z[:4], z[4:4 + N * X].reshape(N, X), z[4 + N * X:].reshape(N, X))
    obj = lambda z: -ll_counts(split(z)[1], n)
    jac = lambda z: np.r_[np.zeros(4), -(n[..., 1] / split(z)[1] - n[..., 0] / (1 - split(z)[1])).ravel(), np.zeros(N * X)]
    def con(z):
        th, P, V = split(z)
        return np.r_[(V - g.Psi_V(V, P, th)).ravel(), (P - g.Psi_P(V, P, th)).ravel()]
    z0 = np.r_[theta0, Phat.ravel(), g.Gamma(Phat, theta0).ravel()]   # start from the first stage CCPs
    bounds = [(None, None)] * 4 + [(1e-6, 1 - 1e-6)] * N * X + [(None, None)] * N * X
    r = minimize(obj, z0, jac=jac, constraints=[{'type': 'eq', 'fun': con}], bounds=bounds,
                 method='SLSQP', options=dict(maxiter=500, ftol=1e-10))
    return r.x[:4], r.success
```

One data set of $M = 400$ markets observed for $T = 10$ periods for each case, the sample
size of the paper. The first stage CCPs are the frequencies, smoothed to stay inside
$(0, 1)$:

```{code-cell} python3
for case, g in games.items():
    n = g.simulate(P_true[case], seed=1)
    Phat = (n[..., 1] + 0.5) / (n.sum(axis=-1) + 1)           # smoothed frequency estimator of the CCPs
    t0 = time.time(); th_pml = pml(n, Phat, g, np.ones(4)); t1 = time.time()
    th_npl, k = npl(n, Phat, g, th_pml); t2 = time.time()
    th_mpec, ok = mpec(n, Phat, g, th_pml); t3 = time.time()
    print(f'{case}, truth (RS, RN, FC, EC) = {g.theta}')
    print(f'  2S-PML {th_pml.round(3)}  {t1 - t0:5.1f} sec')
    print(f'  NPL    {th_npl.round(3)}  {t2 - t1:5.1f} sec, ' + (f'converged in {k} iterations' if k else 'did not converge'))
    print(f'  MPEC   {th_mpec.round(3)}  {t3 - t2:5.1f} sec, success {ok}')
```

The results reproduce the findings of the paper on a smaller game:

- 2S-PML is fast but biased: $\theta^{RS}$ and $\theta^{RN}$, the two parameters that
  depend on the rivals' behavior, are both underestimated
- NPL corrects the bias when it converges, in case 3, and does not converge in case 4,
  where the equilibrium in the data is unstable under the best response update
- MPEC recovers the parameters in both cases, from the 2S-PML starting values

:::{div}
:class: discussion

- Which entries of the constraint Jacobian are not zero? Sketch its sparsity pattern for
  the constraints of System I and of System II.
- MPEC started from the 2S-PML estimates. Would it still find the right equilibrium in
  case 4 from a poor start?
:::

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

````{warning} Practical task: a Monte Carlo study of the dynamic entry game

(task-mpec-practice)=
Pull the class code and open the notebook `mpec.ipynb` in `session24-nov17/`:

```bash
cd sb-dse-code && git pull
# first time: git clone https://github.com/fediskhakov/sb-dse-code.git
```

1. Run 20 data sets for each of the two cases and report the mean and the standard
   deviation of the 2S-PML, NPL and MPEC estimates, the share of NPL runs that converge,
   and the run times.
2. Start MPEC in case 4 from five random starting values instead of the 2S-PML estimates.
   How often does it reach the same maximum, and is the equilibrium the one in the data?
3. Write the Jacobian of the constraints of System I analytically, plot the sparsity
   pattern of the full constraint Jacobian with `plt.spy`, and compare the run time of
   SLSQP with the exact and the finite difference Jacobians.
4. Increase the number of firms to $N = 4$ and then $N = 5$ with firm-specific fixed costs,
   as in the paper. How do the run times of the three estimators grow?
````

````{note} References and additional resources

(mpec-references)=
- 📖 {cite:t}`suConstrainedOptimizationApproaches2012` "Constrained Optimization Approaches to Estimation of Structural Models"
- 📖 {cite:t}`egesdal2015EstimatingDynamicDiscretechoice` "Estimating dynamic discrete-choice games of incomplete information", MPEC for the dynamic entry game and the Monte Carlo comparison of this class
- 📖 {cite:t}`dube2012ImprovingNumericalPerformance` "Improving the Numerical Performance of Static and Dynamic Aggregate Discrete Choice Random Coefficients Demand Estimation", MPEC for BLP demand estimation
- 📖 {cite:t}`ecma_comment` "Constrained optimization approaches to estimation of structural models: Comment"
- 📖 {cite:t}`suEstimatingDiscretechoiceGames2014` "Estimating discrete-choice games of incomplete information: Simple static examples"
- 📖 {cite:t}`aguirregabiriaSequentialEstimationDynamic2007` "Sequential Estimation of Dynamic Discrete Games"
- 📄 SciPy documentation of constrained minimization with SLSQP and trust-constr [docs.scipy.org/doc/scipy/tutorial/optimize.html](https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize)
````
