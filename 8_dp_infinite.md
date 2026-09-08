---
title: "📖 Programming practice: infinite horizon"
short_title: 📖 Infinite horizon
subtitle: Class 8 — Thursday, September 17
exports:
  - format: typst
    output: exports/8_dp_infinite.pdf
downloads:
  - file: 8_dp_infinite.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

Drop the terminal period and backwards induction has nothing to start from: the Bellman
equation becomes an equation in function space, solved by finding a fixed point. Why
that point exists and is unique, and the two algorithms that find it.

````{hint} Running the code for this lecture
:class: dropdown

The code for this class is in the folder `session08-sep17/` of the course **code
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

(dptheory)=
## Contraction mappings and fixed points

Solution of a dynamic model in infinite horizon is the solution to the **Bellman euqation** which is an actual equation in function space

We rely on contraction properties of the Bellman operator to find the fixed point that in order to solve infinite horizon dynamic models

````{attention} Definition

Let
- $ (S,\rho) $ be a complete metric space
- $ T: S \rightarrow S $ denote an operator mapping $ S $ to itself

$ T $ is called a *contraction* on $ S $ with modulus $ \lambda $ if $ 0 \le \lambda < 1 $ and

$$
\rho(Tx,Ty) \le \lambda \rho(x,y) \; \forall x,y \in S
$$

````

*Contraction mapping brings points in its domain “closer” to each other!*

````{tip} Example: the value of an annuity

$$\stackrel{\nearrow}{V} \quad
\stackrel{\searrow}{c} \quad
\stackrel{\searrow}{c} \quad
\stackrel{\searrow}{c} \quad
\dots$$

- interest rate $r$
- What is the value of the annuity $V$?

$$\stackrel{\nearrow}{V} \quad
\stackrel{\searrow}{c} \quad
\stackrel{\searrow}{c} \quad
\stackrel{\searrow}{c} \quad
\dots$$

$$V=\quad
\frac{c}{(1+r)^0} + \quad
\frac{c}{(1+r)^1} + \quad
\frac{c}{(1+r)^2} + \quad
\frac{c}{(1+r)^3} + \quad
\dots$$

$$\beta = \frac{1}{1+r}$$

$$V=\quad
c + \quad
c \beta + \quad
c \beta^2 + \quad
c \beta^3 + \quad
\dots
=
\sum_{t=0}^{\infty} \beta^t c$$

Assuming $\beta<1$

$$V = \sum_{t=0}^{\infty} \beta^t c = \frac{c}{1-\beta}$$


But we can also reformulate recursively (as \"Bellman equation\" without choice)

$$V = c + \beta ( c + \beta c + \beta^2 c + \dots ) = c + \beta V$$

$$T(V) = c + \beta V$$

$$|T(V_1) - T(V_2)| = |(c + \beta V_1) - (c + \beta V_2)| = \beta | V_1 -  V_2 |$$

- contraction mapping under Euclidean norm!
- modulus $\beta$

````

**Successive approximations** to find the value of annuity

1.  Start with a guess $V_0$
2.  Insert into the \"Bellman equation\"

$$V_{i+1} = c + \beta V_i$$

3.  Repeat until convergence

$$||V_{i}-V_{i-1}|| \le \varepsilon \text{ (small number)}$$


```{code-cell} python3
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

class annuity():

    def __init__(self,c=1,beta=.9):
        self.c = c           # Annual payment
        self.beta = beta     # Discount factor
        self.analytic = c/(1-beta)  # compute analytic solution right away

    def bellman(self,V):
        '''Bellman equation'''
        return self.c + self.beta*V

    def solve(self, maxiter = 1000, tol=1e-4, verbose=False):
        '''Solves the model using successive approximations'''
        if verbose: print('{:<4} {:>15} {:>15}'.format('Iter','Value','Error'))
        V0=0
        for i in range(maxiter):
            V1=self.bellman(V0)
            if verbose: print('{:<4d} {:>15.8f} {:>15.8f}'.format(i,V1,V1-self.analytic))
            if abs(V1-V0) < tol:
                break
            V0=V1
        else:  # when i went up to maxiter
            print('No convergence: maximum number of iterations achieved!')
        return V1
```

```{code-cell} python3
a = annuity(c=10,beta=0.954)
print('Analytic solution is',a.analytic)
print('Numeric solution is ',a.solve())
```

```{code-cell} python3
a.solve(verbose=True)
```


### Solving infinite horizon dynamic models

1.  Value function iterations = successive approximations on the Bellman operator $\rightarrow$ to solve for
    the fixed point of Bellman operator
2.  Policy iteration method = Howard\'s policy improvement algorithm,
    iterative solution for the fixed point of Bellman operator
3.  Newton-Kantorovich method = Newton solver for the fixed point of
    Bellman operator, mathematically identical to Howard\'s iterations

### Convergence of infinite horizon solution methods

- In infinite horizon all solution methods continue until convergence.
- How can we be sure that the algorithm would terminate?

The answer is given by the theory of contraction mappings:

- Bellman operator is generally a contraction mapping
- **Banach theorem** guarantees uniqueness of the fixed point, and
- Successive approximation solver is globally convergent (works with
    any starting point)

````{attention} Definition

**Banach contraction mapping theorem (fixed point theorem).**
Let $(S,\rho)$ be a complete metric space with a contraction mapping
$T: S \rightarrow S$. Then

1.  $T$ admits a unique fixed-point $V^{\star} \in S$, i.e.
    $T(V^{\star}) = V^{\star}$.
2.  $V^{\star}$ can be found by repeated application of the operator
    $T$, i.e. $T^n(V) \rightarrow V^{\star}$ as $n\rightarrow \infty$.

````

*In other words, the fixed point can be found by successive
approximations from any starting point* $\rightarrow$ *VFI method
follows*

### What about Bellman operator?

$$T(V)(\text{state}) = \max_{\text{decisions}} \big[ U(\text{state},\text{decision}) + \beta \mathbb{E}\big\{ V(\text{next state})  \big| \text{state},\text{decision} \big\} \big]$$

- Bellman operator $T: U \rightarrow U$ from functional space $U$ to
    itself
- metric space $(U,d_{\infty})$ with uniform/infinity/sup norm (max
    abs distance between functions over their domain)


````{attention} Definition

**Blackwell sufficient conditions for a contraction.**
Let $X \subseteq \mathbb{R}^n$ and $B(x)$ be the space of bounded
functions $f: X \rightarrow \mathbb{R}$ defined on $X$. Suppose that
$T: B(X) \rightarrow B(X)$ is an operator satisfying the following
conditions:

1.  (monotonicity) For any $f,g \in B(X)$ and $f(x) \le g(x)$ for all
    $x\in X$ implies $T(f)(x) \le T(g)(x)$ for all $x\in X$,
2.  (discounting) There exists $\beta \in (0,1)$ such that

$$T(f+a)(x) \le T(f)(x) + \beta a, \text{ for all } f\in B(X), a \ge 0, x\in X,$$

Then $T$ is a contraction mapping with modulus $\beta$.

````

- Monotonicity of Bellman equation follows trivially due to
    maximization in $T(V)(x)$
- Discounting: satisfied by elementary argument when $\beta<1$

Under additional boundedness conditions, **Bellman operator is a
contraction mapping** by Blackwell sufficient conditions

$\Rightarrow$

- unique solution
- VFI algorithm is globally convergent
- does not depend on the numerical implementation of the Bellman
    operator


### Why do we need other solution algorithms?

Although VFI is guaranteed to find the solution, it may be very
inefficient when modulus of contraction (discount factor $\beta$) is
close to one.

- Newton-based method converge quadratically, but are not globally convergent, have to be initialized at their domain of attraction
- Polyalgorithm would be a good idea, see NFXP in the next lecture


## Inventory dynamics problem with stochastic demand in infinite time

The model is given by the Bellman equation

$$
\begin{array}{rcl}
V(x_t,d_t) &=& \max_{q_t \ge 0} \Big\{ \pi_t + \beta \mathbb{E}\Big[ V\big(x_{t+1} , d_{t+1} \big) \Big| x_t,d_t,q_t \Big] \Big\} \\
&=& \max_{q_t \ge 0} \Big\{ s_t p - k_t r - c \mathbb{1}\{q_t>0\}
+ \beta \mathbb{E}\Big[ V\big( k_t, d_{t+1} \big) \Big] \Big\}
\end{array}
$$

$$
\begin{array}{rcl}
s_t &=& \min\{x_t,d_t\} \\
k_t &=& \max\{x_t-d_t,0\} + q_t
\end{array}
$$

### Dropping the time subscripts

Because we’ll be solving the problem in infinite horizon, the time subscripts can be dropped, and we can just have current period variables $ x,d,q,s,k $, and next period variables denoted by prime, i.e. $ x' $

The Bellman equation is then

$$
\begin{array}{rcl}
V(x,d) &=& \max_{q \ge 0} \Big\{ \pi + \beta \mathbb{E}\Big[ V\big(x', d' \big) \Big| x,d,q \Big] \Big\} \\
&=& \max_{q \ge 0} \Big\{ s\cdot p - k\cdot  r - c \mathbb{1}\{q>0\}
+ \beta \mathbb{E}\Big[ V\big( k, d' \big) \Big] \Big\}
\end{array}
$$

$$
\begin{array}{rcl}
s &=& \min\{x,d\} \\
k &=& \max\{x-d,0\} + q
\end{array}
$$

### Bellman equation in expected value function space

Note that similar to the bus engine replacement model, the inventory model features random variable which distribution does not depend on the previous period variables (it is *idiosyncratic*).

In this case it is possible to reduce the dimensionality of the fixed point problem by rewriting the Bellman operator in expected value function terms.

$$
EV(x') =  \mathbb{E}\Big[ V\big(x', d' \big) \Big| x,d,q \Big] =  \mathbb{E}\Big[ V\big(x', d' \big) \Big],
$$

where the expectation is taken over the distribution of the next period demand $ d' $.
The conditioning on $ x,d,q $ can be dropped exactly because $ d' $ is idiosyncratic.

We can then write the Bellman equation as

$$
V(x,d) = \max_{q \ge 0} \Big\{ s\cdot p - k\cdot  r - c \mathbb{1}\{q>0\}
+ \beta EV(k) \Big\}
$$

$$
V(x,d) = \max_{q \ge 0} \Big\{ p \min\{x,d\} - r ( \max\{x-d,0\} + q ) - c \mathbb{1}\{q>0\}
+ \beta EV(\max\{x-d,0\} + q) \Big\}
$$

Taking the expectation with respect to $ d $ on both sides, we get

$$
EV(x) = \mathbb{E}\Big[ \max_{q \ge 0} \Big\{ p \min\{x,d\} - r ( \max\{x-d,0\} + q ) - c \mathbb{1}\{q>0\}
+ \beta EV(\max\{x-d,0\} + q) \Big\} \Big]
$$

By assumption the inventory is discrete, and so it is natural to assume that the demand is also represented as a discrete random variable.  Then the expectation can be written as a sum weighted with
the corresponding probabilities $ pr(d) $, as

$$
EV(x) = \sum_{d} \Big[ \max_{q \ge 0} \Big\{ p \min\{x,d\} - r ( \max\{x-d,0\} + q ) - c \mathbb{1}\{q>0\}
+ \beta EV(\max\{x-d,0\} + q) \Big\} \Big] pr(d)
$$

This is functional equation in $ EV $ which is also a contraction mapping!

Demand is the truncated geometric of Tuesday, $ pr_i = (1-\lambda)^i \lambda $ on
$ i \in \{0,1,\dots,N\} $, with the last probability corrected for truncation.

### Post-trade stock

The order is decided after trading, so it depends on what is left rather than on $ x $
and $ d $ separately. Let $ y = \max\{x-d,0\} = x - \min\{x,d\} $ denote the
**post-trade stock**:

$$
\begin{array}{rcl}
q^\star(x,d) &=& \arg\max_{q \ge 0} \Big\{ p \min\{x,d\} - (y + q)r  - c \mathbb{1}\{q>0\}
+ \beta EV(y+q) \Big\}\\
&=& \arg\max_{q \ge 0} \Big\{ - q r - c \mathbb{1}\{q>0\}
+ \beta EV(y+q) \Big\} = q^\star(y)
\end{array}
$$

Everything that does not involve $ q $ drops out of the $ \arg\max $, so the policy is a
function of *one* variable. The solution of the model is then the pair

$$
EV(x) = \sum_{d} \Big[ p \min\{x,d\} - yr + \max_{q \ge 0} \Big\{ -qr -c \mathbb{1}\{q>0\}
+ \beta EV(y+q) \Big\} \Big] pr(d), \qquad
q^\star(y) = \arg\max_{q \ge 0} \Big\{ -qr - c \mathbb{1}\{q>0\} + \beta EV(y+q) \Big\}
$$

## Policy iterations

Also known as **Howard policy improvement**. It applies to infinite horizon problems in
discrete time, and is particularly well suited to finite state spaces. It is
mathematically equivalent to the Newton–Raphson method for the fixed point of the
Bellman operator, which returns in the next class as Newton–Kantorovich iterations.

The idea is to break the search for the fixed point into two steps and alternate:

1. **Policy evaluation** — compute the value of a *fixed* policy, which is the Bellman
   equation without the max operation:

   $$
   V(\text{state}) = U(\text{state},\text{decision}) + \beta \mathbb{E}\big\{ V(\text{next state}) \big| \text{state},\text{decision} \big\}
   $$

   - still a functional equation, but a simpler one
   - with discrete states it becomes a system of equations, and where the expectation is
     a matrix multiplication, a *linear* one

2. **Policy improvement** — for the value function just computed, apply the Bellman
   operator and keep the $ \arg\max $. This is a single VFI step

Repeat until the policy stops changing. Convergence is much faster than VFI — but each
iteration costs more, so an efficient policy evaluation step is what justifies the
method.

### Policy evaluation as a linear system

For the inventory model the first step is linear, and worth doing explicitly because the
same construction returns for the bus engine model.

- $ EV $ is a vector with $ N+1 $ elements, one per point of the state space
- $ EV(y+q(y)) $ is a *re-indexing* of that vector, so it can be written as $ M \cdot EV $
  with $ M $ a matrix of zeros and ones
- $ C(d) = p \min\{x,d\} - (y+q(y))r - c \mathbb{1}\{q(y)>0\} $ collects everything that
  does not involve $ EV $

$$
EV = \sum_{d} \Big[ C(d) + \beta M(d) \cdot EV \Big] pr(d)
= \bar{C} + \beta \bar{M} \cdot EV,
\qquad \bar{M} = \sum_{d} pr(d) M(d)
$$

which is the linear system $ (I-\beta\bar{M}) \cdot EV = \bar{C} $, solved in one call to
`np.linalg.solve`.

:::{div}
:class: discussion

- Policy iterations converge in far fewer iterations than VFI. Does that make them
  faster?
- What does $ \bar{M} $ mean in probabilistic terms, and why is $ I - \beta\bar{M} $
  invertible?
- VFI starts from a value function, policy iterations from a policy. Does the starting
  point matter for where either of them ends up?
:::

(task8.1)=
````{warning} Practical task 8.1: solving the infinite horizon inventory model

We code this together in class, starting from `session08-sep17/inventory_pre.ipynb`.
**Pull the code repository before the class:**

```bash
cd sb-dse-code && git pull
```

Nothing is collected.

1. The Bellman operator in expected value function space, and the policy as a function
   of the post-trade stock $ y $
2. A VFI solver on top of it, with a callback that plots the convergence path
3. Policy evaluation as the linear system above, and the policy iterations solver
4. The two solvers compared on the same model — iterations, run time, and the same
   answer to within tolerance

````


(8_dp_references)=
````{note} References and additional resources

- 📖 {cite:t}`adda2023DynamicEconomicsQuantitative` "Dynamic Economics: Quantitative Methods and Applications", chapters 2 and 3
- 📖 {cite:t}`Rust2016` "Dynamic programming", The New Palgrave Dictionary of Economics
- 📖 {cite:t}`sargent2025DynamicProgrammingFinite` "Dynamic Programming: Finite States" — contraction mappings and the two solvers, at length
- 📖 {cite:t}`maDynamicProgrammingDeconstructed2021` "Dynamic programming deconstructed: Transformations of the Bellman equation and computational efficiency" — why the expected value function representation is the cheaper fixed point
- Wiki: Banach fixed-point theorem https://en.wikipedia.org/wiki/Banach_fixed-point_theorem

````
