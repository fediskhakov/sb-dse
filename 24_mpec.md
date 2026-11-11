---
title: "🔬 Practice: Mathematical programming with equilibrium constraints (MPEC)"
short_title: "🔬 MPEC"
subtitle: Class 24 — Tuesday, November 17
downloads:
  - file: 24_mpec.md
    title: MyST Markdown
    
kernelspec:
  name: python3
  display_name: Python 3
---

Today we estimate the static entry game of {cite:t}`suEstimatingDiscretechoiceGames2014` and
see the [trouble with multiple equilibria from Class 23](23_entry_games.md#games-multiplicity)
up close: the likelihood has cliffs, and MPEC avoids them by never solving for all equilibria.

````{seealso} Key reading

{cite:t}`suConstrainedOptimizationApproaches2012` "Constrained Optimization Approaches to
Estimation of Structural Models", *Econometrica* 80(5), 2213–2230. The MPEC formulation of
a structural estimator, its equivalence with the nested fixed point algorithm, and the
idea this class applies to the static entry game.

{cite:t}`suEstimatingDiscretechoiceGames2014` "Estimating discrete-choice games of
incomplete information: Simple static examples", *Quantitative Marketing and Economics*
12(2), 167–207. The static entry game of this class, its multiple equilibria, and how MPEC and other estimators perform on it.
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

(mpec-idea)=
## Maximum likelihood as constrained optimization

The likelihood of a structural model depends on the parameters $\theta$ and on objects
$\sigma$ the model determines, such as the expected value function of the bus model or the
equilibrium choice probabilities of a game. The model ties them together by equations
$G(\theta, \sigma) = 0$, so maximum likelihood is a constrained optimization problem:

$$
\begin{aligned}
\max_{\theta, \sigma} \quad & \mathcal{L}(\theta, \sigma; \text{data}) \\
\text{subject to} \quad & G(\theta, \sigma) = 0 .
\end{aligned}
$$

There are two ways to solve it:

- the [nested fixed point algorithm of Class 10](10_nfxp.md#nfxp-nested-loop) eliminates
  $\sigma$: an inner loop solves $G(\theta, \sigma) = 0$ for $\sigma(\theta)$ at every trial
  $\theta$, and the outer loop maximizes $\mathcal{L}(\theta, \sigma(\theta))$ over $\theta$
  alone
- **MPEC**, mathematical programming with equilibrium constraints
  {cite:p}`suConstrainedOptimizationApproaches2012`, gives the whole problem to a
  constrained optimizer: $\theta$ and $\sigma$ are unknowns together, and the constraints
  have to hold only at the solution

Both have the same solutions. In a game the difference matters: $G(\theta, \sigma) = 0$ can
have several solutions, and the inner loop of NFXP has to find all of them at every trial
$\theta$.

(games-static)=
## A static entry game

The example is from {cite:t}`suEstimatingDiscretechoiceGames2014`. Two firms, $a$ and $b$,
decide simultaneously whether to enter a market, $d_a, d_b \in \{0, 1\}$. The payoff of
firm $a$ is

$$
u_a(d_a, d_b, x_a, \varepsilon_a) =
\begin{cases}
\big[\alpha + d_b (\beta - \alpha)\big] x_a + \varepsilon_{a1}, & d_a = 1, \\
\varepsilon_{a0}, & d_a = 0,
\end{cases}
$$

and the payoff of firm $b$ is symmetric. Here:

- $\alpha x_a$ is the monopoly profit of firm $a$, and $\beta x_a$ its duopoly profit
- $x = (x_a, x_b)$ are the observed types of the firms, common knowledge
- $\theta = (\alpha, \beta)$ are the structural parameters to estimate
- $\varepsilon = (\varepsilon_{a0}, \varepsilon_{a1}, \varepsilon_{b0}, \varepsilon_{b1})$
  are private information, observed by each firm about itself and by neither the
  opponent nor the econometrician: a game of **incomplete information**, solved by a
  **Bayesian Nash equilibrium**

Each firm forms beliefs about the entry of the other, $p_a = \Pr\{d_a = 1\}$ and
$p_b = \Pr\{d_b = 1\}$, and its expected payoff from entering is
$[\alpha + p_b (\beta - \alpha)] x_a$. With i.i.d. EV1 shocks the best responses are logit:

$$
\begin{aligned}
p_a &= \frac{1}{1 + \exp\big[-\alpha x_a + p_b x_a (\alpha - \beta)\big]} \equiv \Psi_a(p_b, x_a; \theta), \\
p_b &= \frac{1}{1 + \exp\big[-\alpha x_b + p_a x_b (\alpha - \beta)\big]} \equiv \Psi_b(p_a, x_b; \theta) .
\end{aligned}
$$

The pair defines the equilibrium mapping $\Psi(p_a, p_b) = \big(\Psi_a(p_b), \Psi_b(p_a)\big)$
on $[0, 1]^2$.

````{attention} Definition (Bayesian Nash equilibrium of the entry game)
A **Bayesian Nash equilibrium** is a pair of entry probabilities that are mutual best
responses, $(p_a^\star, p_b^\star) = \Psi(p_a^\star, p_b^\star)$: the beliefs of each firm
about the other are consistent with the other's behavior.
````

$\Psi$ is a continuous map of the square into itself, so by Brouwer's theorem an
equilibrium exists. Uniqueness is another matter.

The model object holds the parameters and the best responses. Substituting one best response
into the other reduces the equilibrium to one equation in $p_a$, the fixed point of the
**second-order best response** $p_a \mapsto \Psi_a(\Psi_b(p_a))$. Each best response falls
with the rival's entry probability, so this map is increasing in $p_a$, and `equilibria`
uses the [successive approximations and bracketing of Class 22](22_eqb_practice.md#eqb-all-equilibria):

- successive approximations from $p_a = 0$ and from $p_a = 1$ converge to the lowest and the
  highest equilibrium
- `brentq` between the two finds the unstable one in the middle, so the method finds at most
  three equilibria, enough for this game

The option `method='grid'` switches to the one-dimensional method of
[Class 22](22_eqb_practice.md#eqb-network-strength): a grid of sign changes refined by
`brentq`.

```{code-cell} python3
import time, warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import brentq, minimize
warnings.simplefilter('ignore', RuntimeWarning)               # log(0) at trial points the optimizers reject
plt.rcParams['legend.frameon'] = False                        # no border around the legends

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

    def equilibria(self, theta=None, method='sa', n=1001, tol=1e-13, maxiter=100000):
        '''All equilibria, as fixed points of the increasing map phi(p_a) = Psi_a(Psi_b(p_a)):
        method='sa'    successive approximations from both ends, brentq in between (at most three)
        method='grid'  sign changes of phi(p_a) - p_a on a grid of n points, refined by brentq'''
        phi = lambda pa: self.br(self.br(pa, self.x[1], theta), self.x[0], theta)
        g = lambda pa: phi(pa) - pa
        if method == 'grid':
            grid = np.linspace(0, 1, n)
            gv = g(grid)                                      # vectorized: the whole grid at once
            pa = [brentq(g, lo, hi, xtol=1e-14)
                  for lo, hi, glo, ghi in zip(grid[:-1], grid[1:], gv[:-1], gv[1:]) if glo * ghi < 0]
            pa = sorted(pa + list(grid[gv == 0]))             # a root exactly on the grid shows no sign change
        elif method == 'sa':
            pa = []
            for p in (0.0, 1.0):                              # from below: the lowest, from above: the highest
                for _ in range(maxiter):                      # slow near a fold, where the slope is close to one
                    p, p0 = phi(p), p
                    if abs(p - p0) < tol: break
                else:
                    raise RuntimeError(f'Successive approximations failed to converge at theta = {theta}')
                pa.append(p)
            lo, hi = pa
            if hi - lo < 1e-8:
                pa = [lo]                                     # a unique equilibrium
            elif g(lo + 1e-9) * g(hi - 1e-9) < 0:
                pa = [lo, brentq(g, lo + 1e-9, hi - 1e-9, xtol=1e-14), hi]   # the unstable one in between
        else:
            raise ValueError(f'Unknown solver: {method}')
        return np.array([[q, self.br(q, self.x[1], theta)] for q in pa])
```

(games-static-solution)=
### Three equilibria

The parameterization is a very contestable market:

- monopoly profits $\alpha x_j = 5 x_j$, duopoly profits $\beta x_j = -11 x_j$
- a firm makes money alone and loses it in a duopoly
- the firm types $(x_a, x_b) = (0.52, 0.22)$

```{code-cell} python3
game = entry_game()
eq = game.equilibria()
print('equilibria (p_a, p_b):', eq.round(4), sep='\n')
```

The equilibria are the intersections of the two best response curves:

```{code-cell} python3
:tags: [hide-input]

pv = np.linspace(0, 1, 101)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(game.br(pv, game.x[0]), pv, color='red', lw=1, label='$\\psi_a(p_b)$')
ax.plot(pv, game.br(pv, game.x[1]), color='blue', lw=1, label='$\\psi_b(p_a)$')
ax.plot(eq[:, 0], eq[:, 1], 's', mfc='none', mec='black', ms=7)
for pa, pb in eq:
    ax.text(pa + 0.03, pb + 0.03, f'({pa:.3f},{pb:.3f})', va='bottom')
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='$p_a$', ylabel='$p_b$', title='Best response functions, firm a and firm b')
ax.legend(loc='upper right')
plt.show()
```

There are three:

- **equilibrium 1**, $(p_a, p_b) = (0.030, 0.730)$: firm $b$ is the likely entrant
- **equilibrium 2**, $(0.616, 0.256)$: both firms enter with moderate probability
- **equilibrium 3**, $(0.774, 0.165)$: firm $a$ is the likely entrant

The second-order best response and the 45-degree line:

```{code-cell} python3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(pv, game.br(game.br(pv, game.x[1]), game.x[0]), color='blue', lw=1, label='$\\psi_a(\\psi_b(p_a))$')
ax.plot(pv, pv, color='tab:green', lw=1)
ax.plot(eq[:, 0], eq[:, 0], 's', mfc='none', mec='black', ms=7)
for pa in eq[:, 0]:
    ax.text(pa + 0.03, pa - 0.03, f'({pa:.3f})', va='bottom')
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='$p_a$', ylabel='$\\psi_a(\\psi_b(p_a))$',
       title='Second order best response function, firm a')
ax.legend(loc='upper right')
plt.show()
```

This is the picture of [Class 22](22_eqb_practice.md#eqb-network-strength) again: the
curve crosses the 45-degree line three times. At equilibria 1 and 3 it crosses from above
with slope below one, so they are **stable** under best response iterations; at
equilibrium 2 the slope is above one, and it is **unstable**. Successive approximations on
the best responses never find equilibrium 2, which is why `equilibria` brackets it.

:::{div}
:class: discussion

- Which equilibrium would you expect to see in the data, and why might a market play the
  unstable one?
- Can this game have more than three equilibria? What about a game with three firms?
:::

(games-static-ml)=
## Maximum likelihood with multiple equilibria

The Monte Carlo design of {cite:t}`suEstimatingDiscretechoiceGames2014`:

- the data are generated by a single equilibrium: the two firms play the same equilibrium
  $N = 1000$ times, with the types fixed at $(x_a, x_b) = (0.52, 0.22)$
- the true parameters are $(\alpha_0, \beta_0) = (5, -11)$
- the data are $X = \{d^n_a, d^n_b\}_{n=1}^{N}$, and the task is to recover $(\alpha, \beta)$

The log-likelihood is that of two binary logits:

$$
\begin{aligned}
\log \mathcal{L}(\alpha, \beta; X) = {} & \sum_{n=1}^N \Big(d^n_a \log p_a(\alpha, \beta) + (1 - d^n_a) \log\big[1 - p_a(\alpha, \beta)\big]\Big) \\
& + \sum_{n=1}^N \Big(d^n_b \log p_b(\alpha, \beta) + (1 - d^n_b) \log\big[1 - p_b(\alpha, \beta)\big]\Big) ,
\end{aligned}
$$

where $(p_a(\alpha, \beta), p_b(\alpha, \beta))$ solve the equilibrium equations. It depends
on the data only through the number of entries of each firm:

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

The nested fixed point estimator:

1. **outer loop**: maximize $\log \mathcal{L}(\alpha, \beta; X)$ over $(\alpha, \beta)$
2. **inner loop**: for the given $(\alpha, \beta)$, solve for **all** equilibria
   $(p^k_a, p^k_b)$, $k = 1, \dots, K$, and keep the one with the highest likelihood,
   $k^\star = \arg\max_k \log \mathcal{L}(p^k_a, p^k_b; X)$

The likelihood jumps, as we see next, so the outer loop uses Nelder–Mead, which needs no
derivatives:

```{code-cell} python3
def nfxp(d, game, theta0=(5, -11)):
    '''NFXP: maximize the likelihood of the best fitting equilibrium at every trial theta'''
    ll = lambda th: max(loglik(np.clip(p, 1e-12, 1 - 1e-12), d) for p in game.equilibria(th))
    return minimize(lambda th: -ll(th), theta0, method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-8))

for k, d in enumerate(data):
    r = nfxp(d, game)
    print(f'equilibrium {k + 1}: NFXP (alpha, beta) = {r.x.round(3)}, log-likelihood {-r.fun:.2f}')
```

(games-static-cliffs)=
### Likelihood cliffs

The log-likelihood of every equilibrium along $\alpha$, with $\beta = -11$ fixed, for the
three data sets. The NFXP likelihood is the upper envelope:

```{code-cell} python3
:tags: [hide-input]

alphas = np.linspace(0.5, 10, 381)
eqs = [game.equilibria((a, -11.0)) for a in alphas]           # all equilibria at every trial alpha
fig, ax = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
for k, d in enumerate(data):
    for a, e in zip(alphas, eqs):
        ax[k].plot([a] * len(e), [loglik(np.clip(p, 1e-12, 1 - 1e-12), d) for p in e], '.', color='0.6', ms=3)
    env = np.array([max(loglik(np.clip(p, 1e-12, 1 - 1e-12), d) for p in e) for e in eqs])
    jumps = np.flatnonzero(np.abs(np.diff(env)) > 100)       # the cliffs: the envelope switches to another equilibrium
    for i, (lo, hi) in enumerate(zip(np.r_[0, jumps + 1], np.r_[jumps + 1, len(env)])):
        ax[k].plot(alphas[lo:hi], env[lo:hi], color='tab:red', lw=2, label='NFXP: the best equilibrium' if i == 0 else None)
    for j in jumps:                                           # dotted connection across each discontinuity
        ax[k].plot(alphas[j:j + 2], env[j:j + 2], ':', color='tab:red', lw=2)
    ax[k].axvline(5, color='black', lw=1, ls='--', label='true alpha')
    ax[k].set(xlabel='alpha', title=f'data from equilibrium {k + 1}', ylim=(max(env) - 4000, max(env) + 200))
    ax[k].grid(color='0.9')
ax[0].plot([], [], '.', color='0.6', label='every equilibrium')
ax[0].set(ylabel='log-likelihood'); ax[0].legend(loc='lower right')
plt.show()
```

For $\alpha$ below about 4.8 the game has a unique equilibrium, close to equilibrium 1. It
fits data from equilibrium 1, and it cannot fit data from equilibria 2 or 3. Where the other
two equilibria appear, the NFXP likelihood jumps by about 2000 log points for the data from
equilibrium 2 and by about 2700 for the data from equilibrium 3, just below the true
$\alpha_0 = 5$.

The likelihood over both parameters, on a $51 \times 51$ grid over the ranges used by
{cite:t}`suEstimatingDiscretechoiceGames2014`. To separate the cliff from the true values,
the data for these figures are generated at $(\alpha_0, \beta_0) = (5.5, -11)$, a little
further from the cliff than $(5, -11)$. The faces that span a cliff are left out, so the
discontinuities show as gaps rather than as walls:

```{code-cell} python3
:tags: [hide-input]

theta0 = (5.5, -11.0)                                         # further from the cliff than (5, -11)
data3 = [simulate(e, seed=k) for k, e in enumerate(game.equilibria(theta0))]   # one data set from each equilibrium
A, B = np.meshgrid(np.linspace(0, 10, 51), np.linspace(-16, -6, 51))   # the alpha and beta ranges of Su (2014)
Z = np.empty((3,) + A.shape)
for i, j in np.ndindex(A.shape):
    e = game.equilibria((A[i, j], B[i, j]))                   # all equilibria at this trial (alpha, beta)
    for k, d in enumerate(data3):                             # NFXP: the best equilibrium for each data set
        Z[k, i, j] = max(loglik(np.clip(p, 1e-12, 1 - 1e-12), d) for p in e)

def cliff_surface(ax, A, B, Z, jump=300, mesh=1):
    '''Draw the surface Z(A, B) from its grid faces and mesh lines, both broken where they span a cliff'''
    v = np.stack([A, B, Z], axis=-1)
    faces = np.stack([v[:-1, :-1], v[:-1, 1:], v[1:, 1:], v[1:, :-1]], axis=2).reshape(-1, 4, 3)
    faces = faces[np.ptp(faces[:, :, 2], axis=1) < jump]      # a face on the vertical wall spans the jump
    norm = Normalize(np.percentile(Z, 2), Z.max())            # colors robust to a small low patch
    ax.add_collection3d(Poly3DCollection(faces, facecolors=plt.cm.viridis(norm(faces[:, :, 2].mean(axis=1))), zorder=2,
                                         edgecolor='none', shade=True, lightsource=LightSource(azdeg=225, altdeg=40)))
    lines = np.concatenate([np.stack([v[::mesh, :-1], v[::mesh, 1:]], axis=2).reshape(-1, 2, 3),   # every mesh-th row
                            np.stack([v[:-1, ::mesh], v[1:, ::mesh]], axis=2).reshape(-1, 2, 3)])  # every mesh-th column
    lines = lines[np.abs(lines[:, 1, 2] - lines[:, 0, 2]) < jump]
    ax.add_collection3d(Line3DCollection(lines, colors=(0, 0, 0, 0.6), linewidths=0.5, zorder=3))
    return faces[:, :, 2].min()                               # the lowest point actually drawn

for k in range(3):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(projection='3d', computed_zorder=False)   # draw in zorder: floor, stem, surface, mesh, marker
    z0, z1 = cliff_surface(ax, A, B, Z[k]), Z[k].max()
    ax.contour(A, B, Z[k], zdir='z', offset=z0, levels=np.linspace(z0, z1, 60), cmap='viridis',
               vmin=np.percentile(Z[k], 2), vmax=z1, linewidths=0.6, zorder=0)   # contours on the floor, in the colors of the surface
    zt = max(loglik(np.clip(p, 1e-12, 1 - 1e-12), data3[k]) for p in game.equilibria(theta0))
    ax.plot([theta0[0]] * 2, [theta0[1]] * 2, [z0, zt], '--', color='tab:red', lw=1.5, zorder=1)
    ax.plot([theta0[0]], [theta0[1]], [zt], 'o', color='tab:red', mec='white', mew=1.5, ms=9, zorder=10,
            label=f'true $(\\alpha, \\beta) = ({theta0[0]:g}, {theta0[1]:g})$')
    ax.set(xlim=(0, 10), ylim=(-16, -6), zlim=(z0, z1), xlabel='$\\alpha$', ylabel='$\\beta$', zlabel='log-likelihood',
           title=f'NFXP log-likelihood, data from equilibrium {k + 1}')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0)); axis._axinfo['grid']['color'] = (0, 0, 0, 0.1)
    ax.set_box_aspect((1, 1, 0.75), zoom=0.9)
    ax.zaxis.labelpad = 14                                    # keep the label clear of the tick labels
    ax.view_init(elev=28, azim=-128)                          # the viewpoint of the figures in Su (2014)
    ax.legend(loc='upper left')
    plt.show()
```

With data from equilibrium 1 the surface is a smooth hill with the maximum near the true
values, except at the back edge: on $\beta = -6$ the game has a single equilibrium for
$\alpha$ between 6.2 and 6.6, the branch that fits the data disappears, and the likelihood
drops by about 1000 log points. With data from equilibria 2 and 3 a cliff cuts across the
surface, and the true values sit on its upper side: along $\beta = -11$ the cliff is at
$\alpha \approx 4.8$, where the likelihood jumps by about 1700 and 3200 log points.

These are the **likelihood cliffs**. Therefore:

- the likelihood is **discontinuous** in the parameters, and gradient-based optimizers fail
  near the cliff
- the true parameters are close to the edge of the region with three equilibria, so the
  usual asymptotic theory, which needs a smooth likelihood around the truth, is in doubt
- the cliff is not a bug of the estimator: it is what the model says

When it finds all the equilibria at every trial value, NFXP works: in the Monte Carlo
experiments of {cite:t}`suEstimatingDiscretechoiceGames2014`, shown [next to those of MPEC
below](#mpec-static), its estimates are centered at the truth for every equilibrium,
including the unstable one. In this game finding all equilibria is easy; in larger games it
is not.

:::{div}
:class: discussion

- Is the NFXP likelihood smooth in $(\alpha, \beta)$? Are the objective and the
  constraints of MPEC smooth in $(\alpha, \beta, p_a, p_b)$?
- Can we tell from the data which equilibrium was played?
- Can we use the standard theorems for inference here?
:::

(mpec-static)=
## MPEC for the static entry game

MPEC computes the same maximum likelihood estimator as NFXP, with the entry probabilities as
unknowns next to the parameters. There are four unknowns, $z = (\alpha, \beta, p_a, p_b)$:

$$
\begin{aligned}
\max_{\alpha, \beta, p_a, p_b} \quad & \log \mathcal{L}(p_a, p_b; X) \\
\text{subject to} \quad & p_a = \Psi_a(p_b, x_a; \alpha, \beta), \\
& p_b = \Psi_b(p_a, x_b; \alpha, \beta), \\
& 0 \leqslant p_a, p_b \leqslant 1 .
\end{aligned}
$$

Three things to see in this formulation:

- the **objective depends on the probabilities only**: it is the likelihood of two binary
  logits,

  $$
  \log \mathcal{L}(p_a, p_b; X) = \sum_{j \in \{a, b\}} \big[n_j \log p_j + (N - n_j) \log(1 - p_j)\big] ,
  $$

  with $n_j$ the number of entries of firm $j$; the parameters enter only through the
  constraints
- the **feasible set** is the graph of the equilibrium correspondence,
  $\{(\theta, p) : p = \Psi(p; \theta)\}$. NFXP maximizes over the same set, first over
  the equilibria at a given $\theta$, then over $\theta$, so the global maxima of the two
  formulations coincide
- the objective and the two constraints are **smooth** in $(\theta, p)$, although the
  NFXP likelihood has cliffs: the graph of the correspondence is a smooth surface in four
  dimensions, and the cliffs appear only when the surface is projected on $\theta$ and the
  best equilibrium is picked at every point

With the Lagrange multipliers $\lambda = (\lambda_a, \lambda_b)$, the Lagrangian is

$$
\mathcal{H}(\theta, p, \lambda) = \log \mathcal{L}(p; X) + \lambda' \big[\Psi(p; \theta) - p\big] ,
$$

and an interior solution satisfies the first-order conditions

$$
\begin{aligned}
\nabla_\theta \Psi(p; \theta)' \lambda &= 0, \\
\nabla_p \log \mathcal{L}(p; X) + \big[\nabla_p \Psi(p; \theta) - I\big]' \lambda &= 0, \\
p - \Psi(p; \theta) &= 0 .
\end{aligned}
$$

The matrix $I - \nabla_p \Psi$ appears, not its spectral radius: the conditions hold at
equilibrium 2 as well as at equilibria 1 and 3. What MPEC needs is that $I - \nabla_p \Psi$
is invertible at the solution, a **regular** equilibrium, and not that the equilibrium is
stable under best responses. What it gives up is the guarantee of a global maximum: a local
solver converges to a local maximum, which can be a point on the wrong sheet of the
correspondence, another equilibrium. Several starting points are the remedy.

The gradient of the objective is simple, $n_j / p_j - (N - n_j) / (1 - p_j)$ with respect
to $p_j$ and zero with respect to $\theta$. SciPy's SLSQP handles the equality constraints
and the bounds, with a finite difference Jacobian of the constraints:

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
including the unstable equilibrium 2, in 12 to 41 iterations. It never solved the game:
the probabilities and the parameters moved together, and the constraints were satisfied
only at the end.

The Monte Carlo experiment of {cite:t}`suEstimatingDiscretechoiceGames2014`: 1000 samples
of $N = 1000$ plays from each equilibrium. NFXP starts at the true parameters, and MPEC
keeps the best of the three starting points above. The probabilities reported for NFXP are
those of the best fitting equilibrium at the estimate:

```{code-cell} python3
def mc_static(k, R=1000):
    '''R samples from equilibrium k estimated by NFXP and MPEC: array 2 x R x (alpha, beta, p_a, p_b)'''
    out = np.empty((2, R, 4))
    for r in range(R):
        d = simulate(eq[k], seed=1000 * (k + 1) + r)
        th = nfxp(d, game).x
        e = game.equilibria(th)
        out[0, r] = np.r_[th, e[np.argmax([loglik(np.clip(p, 1e-12, 1 - 1e-12), d) for p in e])]]
        fits = [mpec_static(d, game, np.array(z0, dtype=float)) for z0 in starts]
        out[1, r] = max((f for f in fits if f.success), key=lambda f: -f.fun, default=fits[-1]).x
    return out

t0 = time.time()
mc = [mc_static(k) for k in range(3)]
print(f'{3 * len(mc[0][0])} samples in {time.time() - t0:.0f} sec')
for k in range(3):
    print(f'equilibrium {k + 1}: MPEC away from the true p_a by more than 0.05 in '
          f'{np.sum(np.abs(mc[k][1, :, 2] - eq[k][0]) > 0.05)} samples')
```

NFXP on the left and MPEC on the right, the true values in the titles of the panels.
Equilibrium 1, stable under best responses:

```{code-cell} python3
:tags: [hide-input]

def mc_hist(ax, v, title, bins):
    '''Histogram of Monte Carlo estimates in the style of Su (2014)'''
    ax.hist(v, bins=bins, color='#352a87', edgecolor='0.2', lw=0.5)
    ax.set_title(title, fontweight='bold')

def plot_mc(k):
    '''NFXP and MPEC estimates for data from equilibrium k, side by side'''
    fig = plt.figure(figsize=(16, 6.3), layout='constrained')
    titles = ['True alpha=5.0', 'True beta=-11.0', f'True $p_a$={eq[k][0]:.3f}', f'True $p_b$={eq[k][1]:.3f}']
    for sf, name, est in zip(fig.subfigures(1, 2, wspace=0.06), ('NFXP', 'MPEC'), mc[k]):
        sf.suptitle(name, fontweight='bold', fontsize=18)
        for j, ax in enumerate(sf.subplots(2, 2).flat):
            mc_hist(ax, est[:, j], titles[j], 20)
    plt.show()

plot_mc(0)
```

Equilibrium 2, unstable:

```{code-cell} python3
:tags: [hide-input]

plot_mc(1)
```

Equilibrium 3, stable:

```{code-cell} python3
:tags: [hide-input]

plot_mc(2)
```

The two estimators give the same distributions, centered at the truth for every
equilibrium. MPEC does not care whether the equilibrium in the data is stable under best
responses.

:::{div}
:class: discussion

- MPEC found equilibrium 2 although it is unstable. What plays the role of the best
  response iterations in MPEC, and why does stability not matter for it?
- What would MPEC do if started at the probabilities of equilibrium 1 with data from
  equilibrium 3?
:::

(mpec-markets)=
## Many markets

Real data come from many markets, observed separately. The same game is played in each, but
not the same equilibrium:

- **markets differ** in the observed types of the firms $x^m = (x^m_a, x^m_b)$, so each
  market is its own game, with its own equilibrium mapping $\Psi(p; x^m, \theta)$
- **the number of equilibria differs** across markets: the same $(\alpha, \beta)$ give one
  equilibrium in some markets and three in others
- **the equilibrium played differs** across markets: each market plays one equilibrium in
  all its observations, but which one is not tied to the other markets
- **the structural parameters are common**: the estimator imposes the same $(\alpha, \beta)$
  in every market, and this restriction is what ties the markets together and lets the
  data of all of them identify $\theta$

The design of {cite:t}`suEstimatingDiscretechoiceGames2014` with many markets:

- 25 markets with the types $(x^m_a, x^m_b)$ on a grid of 5 points in $[0.12, 0.87]$ for
  each firm, and the same $(\alpha_0, \beta_0) = (5, -11)$
- in each market the equilibrium that generates the data is drawn at random among the
  equilibria of that market
- $T = 50$ observations per market

At the true parameters 19 of the 25 markets have three equilibria and 6 have one. Over a
finer grid of types:

```{code-cell} python3
:tags: [hide-input]

xg = np.linspace(0.04, 0.96, 24)                              # a finer grid of types for both firms
XA, XB = np.meshgrid(xg, xg)
NE = np.vectorize(lambda xa, xb: len(entry_game(x=(xa, xb)).equilibria()))(XA, XB)
fig = plt.figure(figsize=(9, 6.5))
ax = fig.add_subplot(projection='3d')
st = ax.stem(XA.ravel(), XB.ravel(), NE.ravel(), linefmt='--', markerfmt='o', basefmt=' ')
st.stemlines.set(color='blue', lw=0.6)
st.markerline.set(color='blue', ms=6)
ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 5), xlabel='$x_a$', ylabel='$x_b$', zlabel='# of Equilibria')
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_pane_color((1, 1, 1, 0)); axis._axinfo['grid'].update(linestyle=':', color=(0, 0, 0, 0.4))
ax.view_init(elev=30, azim=-127.5)                            # the default viewpoint of MATLAB
plt.show()
```

````{note} Numerical glitch in published paper
:class: dropdown

The same figure in {cite:t}`suEstimatingDiscretechoiceGames2014`:

```{image} _static/games/neqb.png
:width: 70%
:align: center
:alt: Number of equilibria as a function of the types of the two firms, the figure of the published paper
```
````

The NFXP likelihood is a sum over markets, so its inner loop picks the best equilibrium
in each market separately. The MPEC formulation has the probabilities of every market
as unknowns, $2 + 2M$ of them:

$$
\begin{aligned}
\max_{\alpha, \beta, \{p^m_a, p^m_b\}} \quad & \sum_{m=1}^M \log \mathcal{L}(p^m_a, p^m_b; X^m) \\
\text{subject to} \quad & p^m_a = \Psi_a(p^m_b, x^m_a; \alpha, \beta), \\
& p^m_b = \Psi_b(p^m_a, x^m_b; \alpha, \beta), \quad m = 1, \dots, M .
\end{aligned}
$$

The probabilities of each market have to be an equilibrium of that market's game, any one
of its equilibria; the markets are linked only through the common $(\alpha, \beta)$.

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

The MPEC estimator stacks the constraints of all markets. It keeps the parameters in a box,
$\alpha \in [0, 20]$ and $\beta \in [-20, 20]$, so that an iteration cannot run away:

```{code-cell} python3
def mpec_markets(d, markets, z0, theta_bounds=((0, 20), (-20, 20))):
    '''MPEC with many markets: unknowns z = (alpha, beta, p^1_a, p^1_b, ..., p^M_a, p^M_b)'''
    M = len(markets)
    n1 = d.sum(axis=0); n0 = d.shape[0] - n1                  # entries by market and firm, M x 2
    P = lambda z: z[2:].reshape(M, 2)
    obj = lambda z: -np.sum(n1 * np.log(P(z)) + n0 * np.log(1 - P(z)))
    jac = lambda z: np.r_[0, 0, -(n1 / P(z) - n0 / (1 - P(z))).ravel()]
    con = lambda z: np.concatenate([P(z)[m] - g.Psi(P(z)[m], z[:2]) for m, g in enumerate(markets)])
    bounds = list(theta_bounds) + [(1e-8, 1 - 1e-8)] * 2 * M
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
at an equilibrium other than the one that generated the data.

The experiment of {cite:t}`suEstimatingDiscretechoiceGames2014` with MPEC alone: 100
samples of 25 markets with $T = 50$. MPEC starts at the true parameters and the entry
frequencies:

```{code-cell} python3
def mc_markets(T, M=None, R=100, seed=0):
    '''R samples of MPEC estimates (alpha, beta): the 25 markets of the grid if M is None,
    otherwise M markets with the types of both firms drawn from the grid in every sample'''
    rng = np.random.default_rng(seed)
    est = np.empty((R, 2))
    for r in range(R):
        mk = markets if M is None else [entry_game(x=rng.choice(xs, size=2)) for _ in range(M)]
        P0, d = simulate_markets(mk, T=T, seed=1000 + r)
        z0 = np.r_[5, -11, d.mean(axis=0).ravel().clip(0.01, 0.99)]   # the truth and the frequencies
        est[r] = mpec_markets(d, mk, z0).x[:2]
    return est

def plot_mc_markets(est):
    '''Distribution of the MPEC estimates in the style of Su (2014)'''
    fig, ax = plt.subplots(2, 1, figsize=(10, 7.5), layout='constrained')
    for j, title in enumerate(('True alpha=5.0', 'True beta=-11.0')):
        mc_hist(ax[j], est[:, j], title, 60)
    plt.show()

est25 = mc_markets(T=50)
plot_mc_markets(est25)
```

With 2 markets and $T = 625$, the types of the two markets drawn at random in every sample:

```{code-cell} python3
:tags: [hide-input]

est2 = mc_markets(T=625, M=2)
plot_mc_markets(est2)
```

:::{div}
:class: discussion

- How many variables are in the constrained optimization problem?
- How would you choose starting values for MPEC with many markets?
- With $M = 25$ markets of three equilibria each, how many combinations of equilibria are
  there? What does that say about the number of local maxima?
:::

The trade-off between the two estimators:

- **NFXP** has 2 unknowns and can tell which equilibrium each market plays, but it needs
  all equilibria in every market at every trial value: easy here, very hard in larger games
- **MPEC** has $2 + 2M$ unknowns and never solves the game, but every market adds
  equilibria in which the optimizer can get stuck, and with many markets the local maxima
  multiply
- both are full information maximum likelihood: the same estimator, computed differently

The implementations here use SLSQP with finite difference constraint Jacobians. Su used
AMPL and KNITRO, with exact derivatives and sparse linear algebra, and more starting points.


(mpec-conclusions)=
## Conclusions

- Maximum likelihood, by NFXP or MPEC, is statistically efficient but computationally
  demanding: NFXP needs all equilibria at every trial value of the parameters.
- Multiple equilibria make the NFXP likelihood discontinuous, with cliffs where the number
  of equilibria changes, which complicates both the computation and the inference, and they
  raise the question of which equilibrium is in the data.
- MPEC computes the maximum likelihood estimator without solving for all equilibria, from a
  smooth problem, and does not care about stability; its risk is local maxima, which
  multiply with the number of markets.


The [next class](25_rls.md) talks about a class of **dynamic games** in which all equilibria can be found.

- We will see that MPEC may be confused in more complex models, and so we build yet a more specified estimator for that class of dynamic games.


````{note} References and additional resources

(mpec-references)=
- 📖 {cite:t}`suConstrainedOptimizationApproaches2012` "Constrained Optimization Approaches to Estimation of Structural Models"
- 📖 {cite:t}`suEstimatingDiscretechoiceGames2014` "Estimating discrete-choice games of incomplete information: Simple static examples"
- 📖 {cite:t}`dube2012ImprovingNumericalPerformance` "Improving the Numerical Performance of Static and Dynamic Aggregate Discrete Choice Random Coefficients Demand Estimation", MPEC for BLP demand estimation
- 📖 {cite:t}`ecma_comment` "Constrained optimization approaches to estimation of structural models: Comment"
- 📄 SciPy documentation of constrained minimization with SLSQP and trust-constr [docs.scipy.org/doc/scipy/tutorial/optimize.html](https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize)
````
