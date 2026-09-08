---
title: "🔬 Coding static discrete choice"
short_title: 🔬 Static logit
subtitle: Class 6 — Thursday, September 10
exports:
  - format: typst
    output: exports/6_logit.pdf
downloads:
  - file: 6_logit.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

The logit model is the workhorse of the whole course: every dynamic discrete choice model we solve and estimate later has a logit formula at its core, evaluated millions of times inside a fixed point iteration nested inside a likelihood maximization. 

- we should know where the formula comes from and what it assumes.
- we should know how to compute it so that it does not silently produce `nan` on the fifth iteration of the inner loop.

````{hint} Running the code for this lecture
:class: dropdown

The code for this class is in the folder `session06-sep10/` of the course **code
repository**. Today is a practical, so that folder holds the pre-code skeleton we fill
in together, and the completed version appears there afterwards.

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

# Where the logit model comes from

## Deterministic choice, and why levels do not matter

A decision maker (DM) in state $x$ picks $d$ from a finite set $D(x)$ to maximize
$u(x,d)$:

$$d^*(x) = \arg\max_{d \in D(x)} u(x,d)$$

Two implications, used below for identification and, immediately, for numerics.

1. **Only differences in utilities matter** — adding a constant changes nothing:

   $$d^*(x) =\arg\max_{d \in D(x)} u(x,d) = \arg\max_{d \in D(x)} \big[u(x,d) + \mathrm{const}(x) \big]$$

   - one alternative's utility has to be normalized, say to zero
   - an attribute of the DM, identical across alternatives, cannot enter on its own —
     only through cross-effects
   - and we may subtract the largest utility before exponentiating anything, which is
     the trick this chapter rests on

2. **The scale of utility is irrelevant** — multiplying by a positive constant changes
   nothing:

   $$d^*(x) =\arg\max_{d \in D(x)} u(x,d) = \arg\max_{d \in D(x)} \big[ \mathrm{const}(x)\cdot u(x,d) \big], \; \mathrm{const}(x)>0$$

   - the variance of the random component is a normalization, not a parameter
   - it returns below as the scale parameter $\sigma$

## Probabilistic choice

The econometrician cannot predict $d^*$, because the DM has private information
$\epsilon$:

$$d^*(x,\epsilon)= \arg\max_{d \in D(x)} [u(x,d)+\epsilon(d)]$$

The object we work with all course is the *conditional choice probability* (CCP)

$$P(d|x)= \hbox{Prob}\left\{d^*(x,\epsilon)=d|x\right\} =
\int_\epsilon I\left\{ d^*(x,\epsilon)=d\right\}q(\epsilon|x) d\epsilon,$$

where $I(\cdot)$ is the indicator function and $q(\epsilon|x)$ the density of $\epsilon$
given $x$.

````{note} How the model developed

1. {cite:t}`thurstone1959MeasurementValues` "The measurement of values" — mathematical
   models of choice in psychology, based on stimuli.
2. {cite:t}`luce1959IndividualChoiceBehavior` "Individual choice behavior" — derived the
   logit structure from the *independence of irrelevant alternatives* (IIA) property of
   choice, with no random utility anywhere.
3. {cite:t}`marschak1960` "Binary choice constraints and random utility indicators" —
   showed that the same structure is consistent with random utility maximization, which
   is where random utility models (RUM) come from.
4. {cite:t}`mcfadden1974ConditionalLogitAnalysisa` "Conditional logit analysis of
   qualitative choice behavior" — proved the converse: the logit formula for choice
   probabilities *necessarily* implies extreme value type I (EV1) distributed
   unobservables. This is the paper that brought discrete choice into econometrics, and
   the 2000 Nobel prize with it.

````

````{attention} Definition

Let $D(x)$ be a finite set of alternatives in state $x$, and $d\in B \subset D(x)$. The
choice model satisfies **independence of irrelevant alternatives (IIA)** if

$$P(d|x,B) \propto P(d|x,D),$$

where $P(d|x,A)$ denotes the choice probability when the choice set is $A\subset D(x)$.
````

Luce's own statement is equivalent — with $P(d_1,d_2)$ the probability of choosing $d_1$
over $d_2$ and $B=\{d_1,d_2\} \subset D(x)$, he required

$$
\frac{P(d_1,d_2)}{P(d_2,d_1)} = \frac{P(d_1|x,B)}{P(d_2|x,B)}, \; \forall B \in D(x)
$$

so the relative odds of two alternatives do not depend on what else is available.

````{attention} Definition

**Luce theorem.** If IIA holds, then there exist non-negative weights $v(x,d)$,
$d \in D(x)$, such that

$$P(d|x) = \frac{\exp v(d,x)}{\sum_{d' \in D(x)} \exp v(d',x)}$$
````

The logit structure thus comes out with no distributional assumption anywhere, and
without specifying $v(d,x)$ at all.

## Extreme value distribution and max-stability

Luce's structure comes out of a RUM when the unobserved components have the type I
extreme value (Gumbel) distribution:

| Notation | EV1($\mu$,$\sigma$) |
|----------|----------|
| Parameters | Location $\mu$, scale $\sigma$ |
| CDF | $F(x) = \exp\left(-\exp(-\tfrac{x-\mu}{\sigma})\right)$  |
| PDF | $f(x) = \tfrac{1}{\sigma}\exp\left(-\tfrac{x-\mu}{\sigma}-\exp(-\tfrac{x-\mu}{\sigma})\right)$ |
| Support | $x\in \mathbb{R}$ |
| Expectation | $\mathbb{E}(x) = \mu + \sigma \gamma$ |
| Mode | $M = \mu$ |
| Variance | $Var(x) = \frac{\sigma^2 \pi^2}{6}$ |
| Standard normalization | $\mu =0, \sigma = 1$ |

where $\gamma = \lim_{n \to \infty}\left( \sum_{k=1}^n \frac{1}{k} - \ln n \right) = 0.57721\dots$
is the Euler–Mascheroni constant (`np.euler_gamma`).

```{code-cell} python3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9, 4]

x = np.linspace(-4, 8, 500)
ev1_pdf = lambda x, mu=0., s=1.: np.exp(-(x-mu)/s - np.exp(-(x-mu)/s))/s
fig, ax = plt.subplots()
for mu, s, c in [(0., 1., '#1A6D91'), (2., 1., '#0E8A27'), (0., 2., '#CB1515')]:
    ax.plot(x, ev1_pdf(x, mu, s), lw=2, color=c, label=f'EV1($\\mu$={mu:g}, $\\sigma$={s:g})')
ax.legend(frameon=False)
ax.set_title('Type I extreme value (Gumbel) density')
ax.grid(True, alpha=.3)
plt.show()
```

- developed to describe the *maxima* of independent draws — the highest level of a
  river, the strongest earthquake in a century
- extreme value theory gives exactly three limiting distributions of maxima, hence
  "type I"
- The other types are Fréchet (type II) and Weibull (type III)
- **max-stability**, the property we need: the maximum of independent EV1 variables with
  a common scale is again EV1, with that same scale

For independent $\{X_i\}$ with CDFs $F_i$,
$F^{\mathrm{max}}_n(x) \equiv Pr\{\max_i X_i \leqslant x\} = \prod_{i=1}^n F_i(x)$, and
for $X_i \sim \mathrm{EV1}(\mu_i,\sigma)$ the product telescopes:

$$
\begin{array}{l}
F^{\mathrm{max}}_n(x) = \Pi_{i=1}^n \exp\left(-\exp(-\tfrac{x-\mu_i}{\sigma})\right) =\\
= \exp\left(- \sum_{i=1}^n \exp(-\tfrac{x-\mu_i}{\sigma})\right) =\\
= \exp\left(-\exp \ln \big[ \exp(-\tfrac{x}{\sigma}) \sum_{i=1}^n \exp \tfrac{\mu_i}{\sigma}\big]\right) =\\
= \exp\left(-\exp \left[ -\frac{x-{\color{red} \sigma \ln \sum_{i=1}^n \exp \tfrac{\mu_i}{\sigma}}}{\sigma} \right]\right)
\end{array}
$$

so the maximum is EV1 with the same scale $\sigma$ and location

$$
\mu^{\mathrm{max}}_n = \sigma \ln \sum_{i=1}^n \exp \tfrac{\mu_i}{\sigma}
$$

This is the **logsum** formula which describes the expected maximum of independent EV1 variables:

$$
X_i \sim \mathrm{EV1}(\mu_i, \sigma), i=1,\dots,n
\iff
\mathbb{E} \max_{i=1,\dots,n} \{X_i\} = \sigma \ln \sum_{i=1}^n \exp \tfrac{\mu_i}{\sigma} + \sigma \gamma
$$

Worth confirming once rather than believing:

```{code-cell} python3

rng = np.random.default_rng(42)
mu, sigma, n = np.array([0., 1., 2.5]), 1.5, 500_000
draws = mu + sigma * rng.gumbel(size=(n, mu.size))   # EV1(mu_i, sigma), independent
mx = draws.max(axis=1)                               # the maximum over alternatives
mu_max = sigma * np.log(np.sum(np.exp(mu/sigma)))    # logsum formula
print(f'simulated mean of the max = {mx.mean():.4f}')
print(f'logsum + sigma*gamma      = {mu_max + sigma*np.euler_gamma:.4f}')
print(f'simulated variance        = {mx.var():.4f}   theory = {sigma**2*np.pi**2/6:.4f}')
```

## The logit choice probabilities

{cite:t}`marschak1960`: Luce's structure obtains in a RUM whose $\epsilon(d)$ are
**i.i.d.** EV1 across alternatives,

$$d^*(x,\epsilon)= \arg\max_{d \in D(x)} [u(x,d)+\epsilon(d)], \; \epsilon(d) \sim \mathrm{EV1, i.i.d.}$$

- independence across alternatives is the whole assumption
- location and scale are free: $\check{X} = (X-\mu)/\sigma$ is EV1(0,1) whenever $X$ is
  EV1$(\mu,\sigma)$, and by points 1 and 2 above shifting and scaling utilities changes
  no choice

The direct derivation — condition on $\epsilon(d)$, use max-stability for the rest, then
substitute $t = -\exp(-\xi)$ — is in the reading. The general route for any RUM:

(wdz-theorem)=
````{attention} Definition

**Williams–Daly–Zachary theorem.** For a RUM with an absolutely continuous distribution
of the random terms,

$$d^*(x,\epsilon)= \arg\max_{d \in D(x)} [u(x,d)+\epsilon(d)],$$

the choice probabilities are the derivatives of the expected maximum:

$$
P(d|x) = \frac{\partial}{\partial u(d,x)} \mathbb{E}\left\{\max_{d' \in D(x)} \big[ u(d',x) +\epsilon(d') \big]\Big|x\right\}
$$
````

- the differentiated object $\mathbb{E}\max_{d'}[u(d',x)+\epsilon(d')]$ is McFadden's
  *social surplus* function {cite:p}`mcfadden1981econometric`
- the proof: differentiation passes under the integral, and the derivative of a max is
  the indicator of the argmax

$$
\begin{array}{l}
\frac{\partial}{\partial u(d,x)} {\displaystyle \int_\epsilon}
\max_{d' \in D(x)} \big[ u(d',x) +\epsilon(d') \big] q(\epsilon|x) d\epsilon =\\
=
{\displaystyle \int_\epsilon} I\left\{ d^*(x,\epsilon)=d\right\}q(\epsilon|x) d\epsilon
=P(d|x)
\end{array}
$$

For EV1(0,1) the expected maximum is the logsum, and one line of calculus gives the
**multinomial logit choice probability**:

$$
\frac{\partial}{\partial u(d,x)} \left[ \ln \sum_{d'\in D(x)} \exp u(d',x) + \gamma \right]
= \frac{\exp u(d,x)}{\sum_{d'\in D(x)} \exp u(d',x)} = P(d|x)
$$

Keeping $\sigma$ instead of normalizing it away — which we want in models with
non-linear utility — the same derivation yields

$$
P(d|x) = \frac{\exp \big(u(d,x)/\sigma\big)}{\sum_{d'\in D(x)} \exp \big(u(d',x)/\sigma\big)}
$$

- $\sigma \to \infty$: every exponent goes to zero, $P(d|x) \to 1/|D(x)|$, choice is
  uniformly random
- $\sigma \to 0$: the probabilities collapse onto the deterministic rule
  $P(d|x) \to I \big\{u(d,x) > u(d',x), \forall d' \neq d\big\}$
- both limits are numerically nasty, and we meet them again below
- $P(d|x) \in (0,1)$ and sums to one exactly — the two things we assert in the code

## The price of IIA

IIA is an axiom about behavior, and behavior does not always oblige.

````{tip} Example: the red bus – blue bus paradox

- A commuter can travel by "blue bus" $b$ or by car $c$, and $u(b,x) = u(c,x)$.
- Logit with IIA gives $P(b|x) = P(c|x) = 1/2$. So far so good.
- Now introduce a third alternative, a "red bus" $r$, identical to the blue bus except
  for the paint, so $u(r,x) = u(b,x) = u(c,x)$.
- Logit now predicts $P(b|x) = P(r|x) = P(c|x) = 1/3$.
- The sensible prediction is $P(b|x) = P(r|x) = 1/4$ and $P(c|x) = 1/2$: the new
  alternative should steal from the bus, not from the car.
````

The culprit is independence of $\epsilon$ across alternatives, plainly wrong when two
alternatives differ only in paint. Remedies, in increasing computational cost:

- **Nested / GEV logit** — relaxes the substitution pattern, keeps closed forms for the
  probabilities and the social surplus. The generating function
  $\mathcal{H}(w) = \sum_i w_i$ gives multinomial logit, a nested sum of powers gives
  nested logit {cite:p}`mcfadden1981econometric`
- **Mixed logit** — integrates over a distribution of taste coefficients; approximates
  any RUM, at the cost of simulating the integral
- **Probit** — joint normal errors, rich covariance, no closed form, multidimensional
  integration
- **Dynamics** — IIA on flow utilities does not carry over to alternative-specific
  values, because continuation values differ. One reason we tolerate EV1 all course

:::{div}
:class: discussion

- The scale parameter $\sigma$ is a normalization in a linear-utility model. Why is it not a normalization once utility is non-linear in parameters?
:::

# Numerical issues: three formulas and how they break

Everything above reduces to three expressions that our code has to evaluate, over and
over, inside solvers and likelihoods:

$$
P_i = \frac{\exp(v_i)}{\sum_j \exp(v_j)}, \qquad
\mathrm{LS} = \log \sum_j \exp(v_j), \qquad
\log P_i = v_i - \mathrm{LS}
$$

Mathematically they are innocuous. On a computer they are not, because `exp` is the
fastest way to leave the range of a floating point number.

## Exponentiation is dangerous

:::{div}
:class: discussion

Do you expect the following code to run without errors? What will it print?
:::

```{code-cell} python3
:tags: [hide-output]

a = 2.
for i in range(14):
    print(f'i={i:2d}   a={a:.3e}   exp(a)={np.exp(a):.3e}')
    a **= 1.25
```

IEEE 754 double precision runs out
at about $1.8\times10^{308}$, so `np.exp` overflows to `inf` as soon as its argument
exceeds

```{code-cell} python3

print(f'largest double      = {np.finfo(float).max:.6e}')
print(f'log of it           = {np.log(np.finfo(float).max):.6f}')
print(f'smallest normal     = {np.finfo(float).tiny:.6e}')
print(f'log of it           = {np.log(np.finfo(float).tiny):.6f}')
```

So utilities above roughly $709$ overflow and utilities below roughly $-745$ underflow to
zero. That sounds like a lot of utility until you remember that we divide by $\sigma$,
that $v$ in a dynamic model is a *discounted lifetime* value with $\beta = 0.9999$, and
that an optimizer exploring the parameter space will happily try $\beta_1 = 10^4$ on its
way to somewhere sensible. A single `inf` turns into `inf/inf = nan`, and `nan`
propagates silently through the whole fixed point.

```{code-cell} python3

def logit_naive(v, sigma=1.0, axis=0):
    '''Logit choice probabilities, straight from the formula'''
    e = np.exp(np.asarray(v)/sigma)
    return e/e.sum(axis=axis, keepdims=True)

v = np.array([1000., 1001., 999.])       # utility differences are just 1 and 2
with np.errstate(over='ignore', invalid='ignore'):
    print('naive :', logit_naive(v))
```

## De-maxing

The fix follows from the fact that only utility *differences* matter. For any constant
$M$,

$$
\frac{\exp(v_i)}{\sum_{j} \exp(v_j)} =
\frac{\exp(v_i-M)}{\sum_{j} \exp(v_j-M)}
$$

because $\exp(-M)$ cancels from numerator and denominator. Choosing $M = \max_j v_j$
means we only ever exponentiate non-positive numbers, so the largest term is exactly
$1$ and overflow is impossible. Underflow may still happen, but it replaces a
astronomically small number by exactly zero, which is harmless — and the denominator is
at least $1$, so we never divide by zero.

The same trick applies to the logsum, where the constant is added back afterwards:

$$
\log \left( \sum_{j} \exp(v_j) \right) =
M + \log \left( \sum_{j} \exp(v_j - M) \right)
$$

````{hint} The rule to remember

Always **de-max** the values that go into logit probabilities and logsums.
````

```{code-cell} python3

def logit(v, sigma=1.0, axis=0):
    '''Logit choice probabilities, computed with de-maxing.
    Alternatives along the given axis, any other dimensions are states.
    '''
    v = np.asarray(v, dtype=float)/sigma
    mx = np.max(v, axis=axis, keepdims=True)      # de-maxing constant
    e = np.exp(v - mx)                            # exponent of non-positive numbers
    return e/e.sum(axis=axis, keepdims=True)

def logsum(v, sigma=1.0, axis=0):
    '''Log-sum-exp = expected maximum of v + EV1 shocks, up to sigma*gamma'''
    v = np.asarray(v, dtype=float)/sigma
    mx = np.max(v, axis=axis, keepdims=True)
    ls = mx + np.log(np.sum(np.exp(v - mx), axis=axis, keepdims=True))
    return sigma*np.squeeze(ls, axis=axis)

print('de-maxed :', logit(v))
print('logsum   :', logsum(v))
print('naive:', np.log(np.sum(np.exp(v))))
```

The de-maxed version returns the same numbers a hand calculation would: utility
differences of $1$ and $2$ give probabilities $0.245, 0.665, 0.090$ regardless of the
level $1000$.

## Never take the log of a probability

The log-likelihood needs $\log P_i$, and the obvious `np.log(logit(v))` is a bad idea:
when a probability underflows to zero — which it will, for a badly-fitting parameter
guess or a small $\sigma$ — the log is $-\infty$ and the optimizer is handed a wall
instead of a gradient. Compute log-probabilities directly instead, where the subtraction
happens before any exponentiation can destroy information:

$$
\log P_i = \frac{v_i - \mathrm{LS}_\sigma(v)}{\sigma}, \qquad
\mathrm{LS}_\sigma(v) = \sigma \log \sum_j \exp(v_j/\sigma)
$$

```{code-cell} python3

def loglogit(v, sigma=1.0, axis=0):
    '''Log of the logit choice probabilities, without ever taking log of a probability'''
    v = np.asarray(v, dtype=float)
    return (v - np.expand_dims(logsum(v, sigma, axis), axis))/sigma

w = np.array([0., -40., -900.])            # third alternative is hopeless
with np.errstate(divide='ignore'):
    print('log(logit(w)) :', np.log(logit(w)))
print('loglogit(w)   :', loglogit(w))
```

The third alternative has log-probability $-900$, not $-\infty$: a small number that a
likelihood can work with, rather than a missing value that stops the optimizer.

## The scale parameter is a numerical hazard

Dividing by $\sigma$ before de-maxing is what keeps small $\sigma$ usable. As
$\sigma \to 0$ the scaled utilities diverge, but their *differences after de-maxing* stay
finite or underflow gracefully to zero.

```{code-cell} python3

u = np.array([1.0, 1.5, 0.8])
for s in [10., 1., 0.1, 1e-3, 1e-8]:
    with np.errstate(over='ignore', invalid='ignore'):
        naive = logit_naive(u, sigma=s)
    print(f'sigma={s:<8g} naive={np.array2string(naive, precision=3)}'
          f'   de-maxed={np.array2string(logit(u, sigma=s), precision=3)}')
```

At $\sigma=10^{-3}$ the naive formula is already `nan`, while the de-maxed one reports
the deterministic choice, which is the correct limit. Note also what the de-maxed answer
does *not* do: it never returns a probability outside $[0,1]$, and the row still sums to
one, which is the check to write into the code.

```{code-cell} python3

p = logit(np.array([[1., 2.], [3., 0.], [-2., 5.]]), sigma=0.4)   # 3 alternatives x 2 states
print(p, end='\n\n')
assert np.allclose(p.sum(axis=0), 1.0), 'choice probabilities must sum to one'
assert np.all(p >= 0.), 'choice probabilities must be non-negative'
print('sums over alternatives:', p.sum(axis=0))
```

````{hint} Rules to code by

1. Never exponentiate a raw utility. De-max along the choice axis first.
2. Divide by $\sigma$ *before* de-maxing, not after.
3. Keep log-probabilities as a separate function; never `log` a probability that may
   have underflowed.
4. Fix an axis convention and stick to it. In this course: **alternatives in rows,
   states in columns**, and every reduction uses `axis=0, keepdims=True`.
5. Assert that probabilities sum to one after every solve. It costs nothing and catches
   an axis mistake immediately.
6. `scipy.special` has tested `logsumexp`, `softmax` and `log_softmax` that do all of the
   above. Use them in production code — we write our own here because in dynamic models
   we will need the derivatives of these expressions too.
````

## Logit as a smoothing device

There is a second reason to have this machinery at hand, unrelated to discrete choice:
the logsum is a *smooth maximum*, and the logit probability is a *smooth indicator*. Many
objects in structural models are kinked or discontinuous — progressive tax schedules,
eligibility thresholds, the upper envelope of value functions in discrete-continuous
choice {cite:p}`egm`, the likelihood correspondence in games with multiple equilibria —
and a solver that needs derivatives cannot cope with them.

For two functions $f(x)$ and $g(x)$, the kinked $h(x) = \max\big(f(x), g(x)\big)$ is
approximated by

$$
\tilde{h}(x) = \sigma\log\big(\exp(f(x)/\sigma) + \exp(g(x)/\sigma)\big)
\to h(x) \;\text{ as }\;\sigma \to 0
$$

and the approximation error is *uniformly bounded and known*. Since $\exp(\cdot/\sigma)$
is monotone,

$$
\exp\frac{\tilde h(x) - h(x)}{\sigma} =
\frac{\exp(f(x)/\sigma) + \exp(g(x)/\sigma)}
{\exp(\max\big[f(x),g(x)\big]/\sigma)}
$$

If $f(x)\geqslant g(x)$ this collapses to $1 + \exp\frac{g(x) - f(x)}{\sigma} \leqslant 2$,
and symmetrically otherwise, with the bound $2$ attained exactly where $f(x) = g(x)$.
Hence

$$
\max_x \{ \tilde h(x) - h(x) \} = \sigma \ln2
$$

so the error is controlled by choosing $\sigma$ — and note that this is the same $\sigma$
that has an economic interpretation as the scale of the taste shocks. Smoothing a kink
and adding an EV1 shock are the same operation.

```{code-cell} python3
:tags: [hide-input]

x = np.linspace(-2, 4, 1000)
f1 = 2 * np.exp(-0.5 * (x - 0.5)**2) + 0.3 * x**2 - 1
f2 = -0.4 * x**3 + 1.2 * x**2 + 0.5 * x + 0.5 + 0.3 * np.sin(2*x) - 0.5
f3 = lambda s: s * np.log(np.exp(f1/s) + np.exp(f2/s))
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
ax.plot(x, f1, linewidth=2.0, color="#1A6D91", label='f(x)', alpha=0.8)
ax.plot(x, f2, linewidth=2.0, color="#0E8A27", label='g(x)', alpha=0.8)
ax.legend(frameon=True, loc='lower left')
s1, s2 = 0.02, 0.8
for s in np.linspace(s1, s2, 10):
    ax.plot(x, f3(s), linewidth=1.0, color="#CB1515", alpha=1 - s)
ax.set_title(f'Smoothed maximum of two functions, sigma from {s2:0.2f} down to {s1:0.2f}', fontsize=14, pad=20)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_xlim(-2, 4); ax.set_ylim(-1, 4)
plt.show()
```

Note that the plot above is computed with the naive `np.exp(f1/s)`, which is fine at
$\sigma = 0.02$ for functions of order one and starts overflowing well before
$\sigma = 10^{-3}$. The de-maxed `logsum` written above handles the whole range — a good
first exercise in the practical.

A jump is smoothed by the logit probability rather than the logsum. If $f(x)$ is smooth
except for a jump of size $\bar f$ at $x_0$,

$$
\tilde f(x) = f(x) + \bar{f} \left[ \frac{\exp\tfrac{x-x_0}{\sigma}}{1 + \exp\tfrac{x-x_0}{\sigma}} - I\{x>x_0\} \right]
\to f(x) \;\text{ as }\;\sigma \to 0,
$$

where $\bar{f} = \lim_{x\to x_0^+} f(x) - \lim_{x\to x_0^-} f(x)$. Convergence here is
only pointwise, not uniform: at $x_0$ itself the smoothed function is off by
$\bar f/2$ no matter how small $\sigma$ is.

```{code-cell} python3
:tags: [hide-input]

x0 = 0.25
x = np.linspace(-1, 2, 10000)
g1 = lambda x: -0.4 * x**3 + 1.2 * x**2 + 0.5 * x + 0.5 + 0.3 * np.sin(2*x) - 0.5
g2 = lambda x: 2 * np.exp(-0.5 * (x - 0.5)**2) + 0.3 * x**2 - 1
fd = (x < x0)*g1(x) + (x > x0)*g2(x)
ix = np.max(np.where(x <= x0))
gap = g2(x[ix]) - g1(x[ix])
fd[ix] = np.nan                                   # to show the discontinuity
smooth = lambda x, s: fd + gap*(1/(1 + np.exp(-(x - x0)/s)) - (x > x0))
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
ax.plot(x, fd, linewidth=2.0, color="#2355A1", label='f(x)')
ax.legend(frameon=True, loc='lower left')
s1, s2 = 0.005, 0.1
for s in np.linspace(s1, s2, 10):
    ax.plot(x, smooth(x, s), linewidth=1.0, color="#CB1515", alpha=1 - 8*s)
ax.set_title(f'Smoothed discontinuity, sigma from {s2:0.3f} down to {s1:0.3f}', fontsize=14, pad=20)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_xlim(-1, 2); ax.set_ylim(-1, 2)
plt.show()
```

# The model we build today

The point of the practical is not the logit formula itself — it is one line — but the
*shape of the code around it*. Every structural project in this course, and your own
project later in the semester, has the same six parts:

1. a **model object** holding the parameters, with the "model parts" (utility, transitions)
   attached to it
2. a **solver** that maps parameters into the model solution
3. a **simulator** that generates data from a solved model
4. a **graphical module** that shows what the model and the data look like
5. an **estimator** that recovers parameters from data (from the NFXP class onwards)
6. a **counterfactual simulator** that changes something and re-solves (later still)

Keep these separate from the *run scripts* that call them. A run script is throwaway; the
model, solver and simulator are what you reuse for the whole semester.

## Specification

A decision maker in state $x \in \{1,2\}$ chooses one of $|D|$ alternatives, each
described by a vector $Y^d$ of $K$ attributes. We consider two utility specifications:

1. **Linear:** $u(d,x) = Y^d \beta^x$
2. **Log:** $u(d,x) = \ln(Y^d)\, \beta^x$

with state-dependent coefficients tied together as $\beta^1 = \beta$ and
$\beta^2 = \beta/2$, so the vector of structural parameters is $\beta \in \mathbb{R}^K$
plus the scale $\sigma$. Choice probabilities are logit with scale $\sigma$, as derived
above.

The state enters only through the coefficients here, which is deliberately minimal —
enough to have something to compare across states, and few enough parameters that the
dashboard is readable.

## The model object

Start with what holds the parameters. Two details are worth the extra lines: `__str__`,
so that printing a model tells you what it is, and `__setattr__`, so that an
inconsistent model cannot be created in the first place. Shape bugs found at assignment
time cost minutes; the same bug found inside a likelihood costs an afternoon.

```{code-cell} python3

class model:
    '''Static random utility model with EV1 taste shocks'''

    def __init__(self,
                 label='noname',
                 nalt=5,                    # number of alternatives
                 nattr=3,                   # number of attributes of each alternative
                 attr=None,                 # attribute values Y, (nalt, nattr)
                 param=None,                # structural parameters beta, (nattr,)
                 st=np.array([1, 2]),       # states of the decision maker
                 util_type='linear',        # 'linear' or 'log'
                 sigma=1.0):                # scale of the taste shocks
        self.label = label
        self.nalt = nalt                    # dimensions first: __setattr__ checks against them
        self.nattr = nattr
        self.st = st
        self.util_type = util_type
        self.sigma = sigma
        self.attr = np.ones((nalt, nattr)) if attr is None else attr
        self.param = np.zeros(nattr) if param is None else param

    def __str__(self):
        '''Human readable representation of the model'''
        return (f'Model "{self.label}" [{self.util_type}]: {self.nalt} alternatives, '
                f'{self.nattr} attributes, {self.st.size} states, sigma={self.sigma:g}\n'
                f'  beta = {np.array2string(self.param, precision=3, separator=", ")}')

    def __setattr__(self, name, value):
        '''Check consistency of the model attributes on assignment'''
        if name == 'attr':
            value = np.asarray(value, dtype=float)
            assert value.shape == (self.nalt, self.nattr), \
                f'attr must have shape ({self.nalt}, {self.nattr}), got {value.shape}'
        if name == 'param':
            value = np.asarray(value, dtype=float)
            assert value.size == self.nattr, \
                f'param must have {self.nattr} elements, got {value.size}'
        if name == 'sigma':
            assert value > 0, 'sigma must be strictly positive'
        super().__setattr__(name, value)

    # ---- model parts ----

    def beta_coef(self):
        '''State-specific coefficients, (nattr, nstates)'''
        beta_st = np.empty((self.nattr, self.st.size))
        beta_st[:, 0] = self.param
        beta_st[:, 1] = self.param/2
        return beta_st

    def utility(self):
        '''Deterministic component of utility, (nalt, nstates)'''
        if self.util_type == 'linear':
            return self.attr @ self.beta_coef()
        elif self.util_type == 'log':
            return np.log(self.attr) @ self.beta_coef()
        raise ValueError(f'Unknown utility specification: {self.util_type}')

    def save_solution(self, chpr):
        '''Store the computed choice probabilities in the model object'''
        self.solution = chpr
```

:::{div}
:class: discussion

- `__setattr__` is called for *every* assignment, including the ones inside `__init__`.
  Why must `self.nalt` be set before `self.attr`?
- What would break if `beta_coef` were computed once in `__init__` and stored?
- Where would the transition matrix live in this design, once the model becomes dynamic?
:::

## The solver

With `logit` already written and tested, the solver is three lines. Notice that it both
returns the solution and stores it on the model — convenient for plotting, and the reason
the simulator can assert that the model has been solved.

```{code-cell} python3

def model_solve(m):
    '''Solve the model: choice probabilities, alternatives in rows, states in columns'''
    chpr = logit(m.utility(), sigma=m.sigma, axis=0)     # de-maxed, see above
    assert np.allclose(chpr.sum(axis=0), 1.0), 'choice probabilities do not sum to one'
    m.save_solution(chpr)
    return chpr

def model_logsolve(m):
    '''Log choice probabilities, for the likelihood later in the course'''
    return loglogit(m.utility(), sigma=m.sigma, axis=0)
```

## The simulator

Simulating a discrete choice is the inverse-CDF method: form the cumulative
probabilities along the alternatives, draw $u \sim U[0,1]$, and find the interval that
contains it. `np.searchsorted` does the search in $O(\log |D|)$ per draw and is
vectorized over draws, which matters when we simulate a hundred thousand agents inside
an MSM criterion later.

```{code-cell} python3

import pandas as pd

def model_simulate(m, chpr=None, nobs=10, rng=None):
    '''Simulate nobs decision makers from a solved model, return a pandas DataFrame'''
    assert chpr is not None or hasattr(m, 'solution'), \
        'solve the model first, or pass choice probabilities to the simulator'
    chpr = m.solution if chpr is None else chpr
    rng = np.random.default_rng() if rng is None else rng
    var_names = ['id', 'st', 'choice'] + [f'attr{i}' for i in range(1, m.nattr + 1)]
    data = pd.DataFrame(np.nan, index=range(nobs), columns=var_names)
    data['id'] = 1 + np.arange(nobs)
    data['st'] = rng.integers(m.st.size, size=nobs)          # states drawn uniformly
    cumulative = np.cumsum(chpr, axis=0)                     # cdf over alternatives
    u = rng.random(nobs)                                     # uniform draws
    for st in range(m.st.size):
        mask = data['st'] == st
        # inverse cdf: index of the first cumulative probability exceeding u
        data.loc[mask, 'choice'] = np.searchsorted(cumulative[:, st], u[mask], side='right')
        # attach the attributes of the chosen alternative
        data.loc[mask, var_names[3:]] = m.attr[data.loc[mask, 'choice'].astype(int), :]
    dtypes = [np.int32, np.int32, np.int32] + [np.float64]*m.nattr
    return data.astype(dict(zip(var_names, dtypes)))
```

:::{div}
:class: discussion

- Why `side='right'` and not `side='left'`? What happens to a zero-probability
  alternative under each?
- The last cumulative probability is $1$ only up to floating point. Can `searchsorted`
  return an out-of-range index, and what would that do?
- We draw one uniform per agent. How many draws would the "obvious" alternative —
  simulating the $\epsilon(d)$ and taking the argmax — need, and would it give the same
  answer?
:::

```{code-cell} python3

rng = np.random.default_rng(2026)
nalt, nattr = 5, 3
attr = rng.uniform(1, 10, size=(nalt, nattr))
m1 = model(nalt=nalt, nattr=nattr, attr=attr, param=np.array([0.1, 0.35, 0.2]), label='model 1')
print(m1)
model_solve(m1)
simdata = model_simulate(m1, nobs=2500, rng=rng)
simdata.head()
```

## The graphical module

Three plotting functions, each of which can either make its own figure or draw into axes
handed to it. That second option is what makes the dashboard possible without
duplicating a line of plotting code.

```{code-cell} python3

import matplotlib

def plot_attributes(m, ax=None):
    '''Bar plot of the attributes of each alternative'''
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    width, x = 0.25, np.arange(m.nalt)
    for i in range(m.nattr):
        ax.bar(x + width*i, m.attr[:, i], width, label=f'attr{i+1}')
    ax.set_ylim(0, ax.get_ylim()[1]*1.15)
    ax.set_title(f'Attributes — {m.label}', fontsize=12)
    ax.set_xlabel('Alternatives')
    ax.legend(loc='upper left', ncols=m.nattr, fontsize=8)

def plot_choice_probabilities(m, ax=None):
    '''Bar plot of the model solution, one panel per state'''
    assert hasattr(m, 'solution'), 'solve the model before plotting its solution'
    if ax is None:
        fig, ax = plt.subplots(1, m.st.size, figsize=(10, 4))
        fig.suptitle(f'Choice probabilities — {m.label}', fontsize=13)
    ymax = 0
    for i in range(min(len(ax), m.st.size)):
        ax[i].bar(range(m.nalt), m.solution[:, i], color='xkcd:fresh green')
        ax[i].set_title(f'State {i}', fontsize=11)
        ax[i].set_xlabel('Alternatives')
        ax[i].set_ylabel('Probability')
        ymax = max(ymax, ax[i].get_ylim()[1])
    for i in range(min(len(ax), m.st.size)):
        ax[i].set_ylim(0, ymax)          # common vertical scale, or the panels lie

def plot_data(m, data, ax=None):
    '''Histograms of simulated choices: pooled, then one panel per state'''
    if ax is None:
        fig, ax = plt.subplots(1, m.st.size + 1, figsize=(12, 4))
        fig.suptitle(f'Frequencies of simulated choices — {m.label}', fontsize=13)
    bins = np.arange(-.5, m.nalt + .5, 1)
    nax, ymax = min(len(ax), m.st.size + 1), 0
    for i in range(nax):
        if i == 0:
            ax[i].hist(data['choice'], bins=bins, edgecolor='white', color='xkcd:light blue')
            ax[i].set_title(f'Pooled, obs={data.shape[0]}', fontsize=11)
        else:
            mask = data['st'] == i - 1
            ax[i].hist(data.loc[mask, 'choice'], bins=bins, edgecolor='white', color='xkcd:pale orange')
            ax[i].set_title(f'State {i-1}, obs={mask.sum()}', fontsize=11)
        ax[i].set_xlabel('Alternatives')
        ax[i].set_ylabel('Frequency')
        ymax = max(ymax, ax[i].get_ylim()[1])
    for i in range(nax):
        ax[i].set_ylim(0, ymax)

def plot_dashboard(m, data):
    '''Attributes and solution on top, simulated data below'''
    ncols = m.st.size + 1
    fig, axs = plt.subplots(2, ncols, figsize=(12, 7), layout='constrained')
    fig.suptitle(f'Dashboard — {m.label}', fontsize=14)
    ax = list(axs.flatten())
    plot_attributes(m, ax=ax[0])
    plot_choice_probabilities(m, ax=ax[1:ncols])
    plot_data(m, data=data, ax=ax[ncols:2*ncols])
```

```{code-cell} python3
:tags: [hide-input]

plot_dashboard(m1, data=simdata)
plt.show()
```

The dashboard is the whole point of building the graphical module: one function call
shows the primitives, the solution and the data side by side, so that changing a
parameter and re-running gives an immediate read on what that parameter does. This is
how you develop intuition about identification — long before writing an estimator, you
should be able to say which feature of the data each parameter moves.

## Playing with the model

```{code-cell} python3
:tags: [hide-input]

def compare(models, data=None):
    '''Solve several models and put their choice probabilities side by side'''
    fig, axs = plt.subplots(len(models), 2, figsize=(10, 3.2*len(models)), layout='constrained')
    axs = np.atleast_2d(axs)
    for k, m in enumerate(models):
        model_solve(m)
        plot_choice_probabilities(m, ax=list(axs[k]))
        axs[k][0].set_ylabel(f'{m.label}\nProbability')
    plt.show()
```

**The role of $\beta$.** One coefficient moved from $0.1$ to $0.9$ reshuffles the whole
profile of choice probabilities, and does so differently in the two states because
$\beta^2 = \beta/2$ shrinks every effect by half.

```{code-cell} python3

mA = model(nalt=nalt, nattr=nattr, attr=attr, param=np.array([0.1, 0.35, 0.2]), label='beta1 = 0.1')
mB = model(nalt=nalt, nattr=nattr, attr=attr, param=np.array([0.9, 0.35, 0.2]), label='beta1 = 0.9')
compare([mA, mB])
```

**The role of $\sigma$.** Same preferences, five times the noise: everything is pulled
towards the uniform distribution, exactly as the limit $\sigma \to \infty$ predicts.
This is why $\sigma$ and the scale of $\beta$ cannot both be identified in the linear
model — multiplying $\beta$ and $\sigma$ by the same positive constant leaves every
choice probability unchanged.

```{code-cell} python3

mC = model(nalt=nalt, nattr=nattr, attr=attr, param=mA.param, sigma=5.0, label='sigma = 5')
compare([mA, mC])
```

````{tip} Solution: can all choices concentrate on one alternative?
:class: dropdown

Yes, and in two different ways — which is the point. Either make one attribute dominate
by raising its coefficient, or shrink $\sigma$ towards zero so the model approaches
deterministic choice. Both produce a degenerate solution, and they are observationally
equivalent in the data.

```python
mD = model(nalt=nalt, nattr=nattr, attr=attr, param=np.array([5.1, 0.35, 0.2]), label='large beta1')
mE = model(nalt=nalt, nattr=nattr, attr=attr, param=mA.param, sigma=1e-3, label='tiny sigma')
compare([mD, mE])
```

Note that `mE` is only computable because `logit` de-maxes: with the naive formula
$\sigma=10^{-3}$ returns `nan`, as we saw above.
````

**Predicted versus simulated behavior.** The solution is a probability; the data is a
finite sample from it. With 100 agents the histogram is a rough sketch of the CCPs, with
100,000 it is the CCPs. Estimation is this relation run backwards, and the sampling noise
you see here is what standard errors measure.

```{code-cell} python3
:tags: [hide-input]

model_solve(mA)
small = model_simulate(mA, nobs=100, rng=rng)
large = model_simulate(mA, nobs=100_000, rng=rng)
for d, ttl in [(small, '100 observations'), (large, '100,000 observations')]:
    mA.label = ttl
    plot_data(mA, data=d)
    plt.show()
mA.label = 'beta1 = 0.1'
```

**Linear versus log utility.** The same $\beta$ means something entirely different once
the attributes are logged: differences between alternatives are compressed, so the
choice probabilities are flatter, and $\sigma$ is no longer a pure normalization.

```{code-cell} python3

mF = model(nalt=nalt, nattr=nattr, attr=attr, param=mA.param, util_type='log', label='log utility')
compare([mA, mF])
```

(task6.1)=
````{warning} Practical task 6.1: a static logit model, end to end

We write this code together in class, starting from the pre-code skeleton
`session06-sep10/logit_pre.ipynb`. **Pull the code repository before the class:**

```bash
cd sb-dse-code && git pull
```

Nothing here is collected — finish what is left over at home, and keep the notebook,
because the same skeleton is reused for the dynamic models later in the course.

**Numerical part**

1. Write `logit`, `logsum` and `loglogit` with de-maxing, and verify against
   `scipy.special.softmax`, `logsumexp` and `log_softmax`.
2. Find, by experiment, the utility level at which the naive formula fails and check it
   against `np.log(np.finfo(float).max)`.
3. Show that your `logsum` reproduces the mean of the simulated maximum of EV1 draws.
4. Redraw the smoothed-maximum figure using your de-maxed `logsum`, and push $\sigma$
   down to $10^{-6}$ — where the naive version cannot go.

**Coding part**

5. Build the `model` class with consistency checks in `__setattr__` and a readable
   `__str__`.
6. Write `model_solve` using your de-maxed logit, with the sum-to-one assertion.
7. Write `model_simulate` for $N$ decision makers, with a seeded generator so the run is
   reproducible.
8. Write the plotting routines and assemble the dashboard.

**Model understanding part** — answer with the dashboard, not with algebra

9. What does each $\beta_k$ do to the choice probabilities, and why does it act
   differently in the two states?
10. What does $\sigma$ do? Can you tell $\beta$ apart from $\sigma$ using data on choices
    alone?
11. Can you find parameters that concentrate all simulated choices on one alternative?
    In how many different ways?
12. Can you mimic the behavior of a DM in state $x=1$ using state $x=2$ and a different
    $\beta$? What does that tell you about identification?
13. Do your answers to 9–12 change in the log utility specification?

Use of AI assistance is allowed and encouraged, subject to the
[course AI policy](https://dse.iskh.me/#ai-policy) — but you must be able to explain
every line, including why the max is subtracted.
````

(6_logit_references)=
````{note} References and additional resources

- 📖 {cite:t}`train2009DiscreteChoiceMethods` "Discrete Choice Methods with Simulation",
  chapters 2–3 for logit and IIA, chapter 4 for GEV
  [online book](https://eml.berkeley.edu/books/choice2.html)
- 📖 {cite:t}`mcfadden1974ConditionalLogitAnalysisa` "Conditional logit analysis of
  qualitative choice behavior" — the original econometric treatment
- 📖 {cite:t}`mcfadden1981econometric` "Econometric models of probabilistic choice" —
  social surplus, GEV and the Williams–Daly–Zachary theorem
- 📖 {cite:t}`luce1959IndividualChoiceBehavior` "Individual choice behavior" — the choice
  axiom, and where the logit formula came from before random utility
- 📖 {cite:t}`marschak1960` "Binary choice constraints and random utility indicators"
- 📄 [SciPy documentation for `scipy.special.logsumexp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html)
  — the standard implementation of the de-maxing trick
- 📄 [What every computer scientist should know about floating-point arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
  — Goldberg (1991), the classic reference on overflow, underflow and cancellation
- 📖 {cite:t}`egm` "The endogenous grid method for discrete-continuous dynamic choice
  models" — where the smoothing of kinks and the upper envelope come back
````
