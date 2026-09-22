---
title: "📖 Practice: infinite horizon"
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


### Main solution algorithms for infinite horizon dynamic models

1.  Value function iterations = successive approximations on the Bellman operator $\rightarrow$ to solve for
    the fixed point of Bellman operator
2.  Policy iterations = Howard\'s policy improvement algorithm,
    iterative solution for the fixed point of Bellman operator
3.  Newton-Kantorovich iterations = Newton solver for the fixed point of
    Bellman operator, mathematically identical to Howard\'s iterations

## Convergence of infinite horizon solution methods

- In infinite horizon all solution methods continue until convergence.
- How can we be sure that the algorithm would terminate?

The answer is given by the theory of contraction mappings:

- Bellman operator is generally a contraction mapping!
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

In other words, the fixed point can be found by successive
approximations from any starting point!

This immediately gives rise to the following solution algorithm

````{attention} Value function iterations (VFI) algorithm

1. Start with any initial guess $V_0$ for the value function
2. Apply the Bellman operator to get a new guess $V_1 = T(V_0)$
3. Repeat until convergence, i.e. $||V_{i}-V_{i-1}|| \le \varepsilon$ for some small $\varepsilon$ 

Banach contraction mapping theorem guarantees convergence.

````


### So is Bellman operator a contraction?

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

1.  (monotonicity) For any $f,g \in B(X)$ 

$$f(x) \le g(x) \implies T(f)(x) \le T(g)(x) \; \forall x\in X,$$
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


:::{div}
:class: discussion

- Why do we need other solution algorithms besides VFI?
- What happens when the modulus of contraction is close to one?

:::

- Newton-based method converge quadratically if started from the basin of attraction
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

### Trick 1: Bellman equation in expected value function space

Idiosyncratic random shocks lead to the possibility of rewriting the Bellman equation in expected value function space, which reduces the dimensionality of the problem!

This is the EV trick in dynamic programming.

Define a new function, the **expected value function** $ EV(x) $ as

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
$$
$$
+ \beta EV(\max\{x-d,0\} + q) \Big\}
$$

Taking the expectation with respect to $ d $ on both sides, we get

$$
EV(x) = \mathbb{E}\Big[ \max_{q \ge 0} \Big\{ p \min\{x,d\} - r ( \max\{x-d,0\} + q ) - c \mathbb{1}\{q>0\}
$$
$$
+ \beta EV(\max\{x-d,0\} + q) \Big\} \Big]
$$

Compute the expectation by summing over all possible values of $d$ weighted by their probabilities $pr(d)$, as

$$
EV(x) = \sum_{d} \Big[ \max_{q \ge 0} \Big\{ p \min\{x,d\} - r ( \max\{x-d,0\} + q ) - c \mathbb{1}\{q>0\}
$$
$$
+ \beta EV(\max\{x-d,0\} + q) \Big\} \Big] pr(d)
$$

This is functional equation in $ EV $ which is also a contraction mapping!

Let as before demand have the truncated geometric distribution $ pr_i = (1-\lambda)^i \lambda $ on
$ i \in \{0,1,\dots,N\} $, with the last probability corrected for truncation.




### Trick 2: Post-trade stock

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
+ \beta EV(y+q) \Big\} \Big] pr(d)
$$
$$
q^\star(y) = \arg\max_{q \ge 0} \Big\{ -qr - c \mathbb{1}\{q>0\} + \beta EV(y+q) \Big\}
$$


(task8.1)=
````{warning} Practical task 8.1: solving the infinite horizon inventory model

We code this together in class, starting from `session08-sep17/inventory_pre.ipynb`.
**Pull the code repository before the class:**

```bash
cd sb-dse-code && git pull
```

1. The Bellman operator in expected value function space, and the policy as a function
   of the post-trade stock $ y $
2. A VFI solver on top of it, with a callback that plots the convergence path
3. What happens to the solution when parameters change, in particular the discount factor $ \beta $ is close to one?

````


(8_dp_references)=
````{note} References and additional resources

- 📖 {cite:t}`sargent2025DynamicProgrammingFinite` "Dynamic Programming: Finite States"
- Online version of the same book on [dp.quantecon.org](https://dp.quantecon.org)
- 📖 {cite:t}`adda2023DynamicEconomicsQuantitative` "Dynamic Economics: Quantitative Methods and Applications", chapters 2 and 3
- 📖 {cite:t}`Rust2016` "Dynamic programming", The New Palgrave Dictionary of Economics
- 📖 {cite:t}`sargent2025DynamicProgrammingFinite` "Dynamic Programming: Finite States" — contraction mappings and the two solvers, at length
- 📖 {cite:t}`maDynamicProgrammingDeconstructed2021` "Dynamic programming deconstructed: Transformations of the Bellman equation and computational efficiency" — why the expected value function representation is the cheaper fixed point
- Wiki: Banach fixed-point theorem https://en.wikipedia.org/wiki/Banach_fixed-point_theorem

````
