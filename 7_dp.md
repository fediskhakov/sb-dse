---
title: 📖 Dynamic programming
short_title: 📖 Dynamic programming
subtitle: Class 7 — Tuesday, September 15
exports:
  - format: typst
    output: exports/7_dp.pdf
downloads:
  - file: 7_dp.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

Dynamic programming is the recursive way of writing a sequential decision problem, and
every model later in the course is a variation on it: the Bellman equation, the
classification of DP models, and one inventory model solved by backwards induction.

````{hint} Running the code for this lecture
:class: dropdown

Every code example below is also a runnable notebook in the course **code repository**,
in the folder `session07-sep15/`.

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

## What is dynamic programming?

**"DP is recursive method for solving sequential decision problems"**

📖 Rust 2006, *New Palgrave Dictionary of Economics*

In computer science the meaning of the term is broader: **DP is a
general algorithm design technique for solving problems with overlapping
sub-problems.**

### Dynamic programming in economics

DP provides a framework to study decision making over time and under
uncertainty and can accommodate learning, strategic interactions between agents (game theory) and market interactions (equilibrium theory)

Many important problems and economic models are analyzed and solved
using dynamic programming:

- Dynamic models of labor supply
- Job search
- Human capital accumulation
- Health process, insurance and long term care
- Consumption/savings choices
- Durable consumption
- Growth models
- Heterogeneous agents models
- Overlapping generation models

## Dynamic discrete choice (DDC)

Extend the discrete choice framework to repeated discrete choice over time

- Given **state** $s_t \in S$ an agent takes a **decision** $d_t \in D(s)$ that determines **current utility** $u_t(s_t,d_t)$ as well as the future states of the world.
- The agent forms (subjective) **beliefs** about the uncertain *next period's state* $s_{t+1}$ that evolve according to a Markov transition probability $p(s_{t+1}|s_t, d_t)$. 
- The agent's problem is to choose a **optimal decision rule** 
$\mathbf{\delta} = \{\delta_0,...,\delta_T\}$, where $d_t=\delta_t(s_t)$, that solves

$$
\max_{\mathbf{\delta}=
\{\delta_0..... \delta_{T}\}} \mathbb{E}{_{\mathbf{\delta}}} \big[ \sum_{t=0}^T \beta^t u_t(s_t,d_t) | s_0=s \big]
$$

where $\mathbb{E}_{\delta}$ denotes expectation with respect to the controlled stochastic process $\{s_t,d_t\}$ induced by the decision rule $\delta$.
- The difficulty is that we are **looking for a set of functions** $\mathbf{\delta} = \{\delta_0,...,\delta_T\}$, not just for a set of numbers $\mathbf{d} = \{d_0,...,d_T\}$ 
- DP simplifies the DDC problem, allowing us to find $\mathbf{\delta} = \{\delta_0,...,\delta_T\}$ using a recursive procedure. 


### Bellman's Principle of Optimality

> An optimal policy has a property that whatever the initial state and
initial decision are, the remaining decisions must constitute an optimal
policy with regard to the state resulting from the first decision.

📖 Bellman, 1957 "Dynamic Programming"

### Breaking the problem into sequence of small problems

- Thus, the sequential decision problem is broken into *initial decision* problem and the *future decisions* problem
- The solution can be computed through **backward induction**, i.e. solving a sequential decision problem from the later periods
- Embodiment of the recursive way of modeling sequential decisions is **Bellman equation**

## Bellman equation

````{attention} Definition

Bellman equation is the complete recursive description of the dynamic optimization problem in discrete time:

$$V(\text{state}) = \max_{\text{decisions}} \big[ U(\text{state},\text{decision}) + \beta \mathbb{E}\big\{ V(\text{next state})  \big| \text{state},\text{decision} \big\} \big]$$

````

- **State variables** — vector of variables that describe all relevant
  information about the modeled decision process, $x_t$
- **Decision variables** — vector of variables describing the choices, $d_t$
- **Instantaneous payoff** — utility function, $u(x_t,d_t)$, with
  time separable discounted utility
- (*next state*) is the *stochastic* next period state resulting from
current state and decision
- expectation $\mathbb{E}\{\cdot\}$ is taken over the distribution of
the next period state conditional on current state and decision
- **Motion rules** — agent’s beliefs of how state variable evolve
  through time, conditional on choices, $x_{t+1} \sim F(x_t,d_t)$  
- $\beta$ is a discount factor to measure future rewards in terms of
current ones

Solution is given by: 

- **Value function** — maximum attainable utility $V(x_t)$  
- **Policy function** — mapping from state space to action space that
  returns the optimal choice, $d^{\star}(x_t)$  

The optimal choices are revealed along the solution of the Bellman
equation as decisions which solve the maximization problem in the right hand side ($\arg\max$ of the maximization in the Bellman equation) 

## Classification of DP models

There are many problems where DP methods are applicable, therefore it is important to quickly classify *your problem* to make others understand exactly what follows

### Whether time is continuous or discrete

1. Discrete time

    - time periods $t$, $t+1$
    - dynamics given by difference equations

2. Continuous time

    - all entities in the model are functions of time
    - dynamics given by differential equation
    - so, math is very different
    - continuous time for cleaner theoretical models, sometimes also solved numerically
    - *not part of this course*

3. Continuous time with discrete events

    - continuous time dynamics, but discrete events (e.g. Poisson process) that change the state of the system
    - appropriate in many settings such as search
    - significant simplification is certain settings
    - 📖 [DSE2024 lecture by Peter Arcidiacono](https://github.com/dseconf/DSE2024/tree/main/03_Arcidiacono) on the topic
    - *also not part of this course*

### Whether horizon is finite or infinite

1. Finite horizon

    - there is terminal period $T$
    - special form of Bellman equation in period $T$
    - in other words, as if $V(\text{at } T +1 ) = \mathbb{0}$
    - value function and policy function are time dependent
    - solved by backwards induction with $T$ number of steps

$$V(\text{state}) = \max_{\text{decisions}} \big[ U(\text{state},\text{decision}) \big] \text{ at terminal period } T$$

2. Infinite horizon

    - time subscripts are dropped, primes for next period values instead
    - solution is given by fixed point of the Bellman operator
    - have to actually solve a *functional equation*

*Most problems can be specified and solved in both finite or infinite horizon*

### Whether choice space is discrete, continuous, mixed discrete-continuous or discretized

1. Problems with discrete choice

    - Rust model of bus engine replacement [next week](9_zurcher.md)
    - Inventory management of discrete goods (later today)
    - Thousands other problems
    - 📖 {cite:t}`aguirregabiriaDynamicDiscreteChoice2010` "Dynamic discrete choice structural models: A survey"

1. Problems with continuous choice

    - cake eating and consumption-savings models (later in the course)
    - discretized?
    - treated as continuous?
    - require interpolating of value function in Bellman equation!

1. Problems with discrete and continuous choice

    - much more complicated: kinks in value functions, discontinuous
        policy function
    - require global optimization in Bellman equation which is not easy
    - one way out is to discretize choices at the cost of reduced accuracy
    - 📖 {cite:t}`egm` DCEGM paper

### What about state space?

- When choice is discrete, typically state space is also finite
- Even when state variables are continuous, by discretization it is
    *converted* to discrete

1. This is true in general when using numerical solvers: state space is discretized within some reasonable bounds

    - choice of upper bounds, number and placement of grid points influence the (accuracy of the) solution

2. Another approach to represent state space is to *project* value and/or policy function onto space of orthogonal polynomials (Chebyshev polynomials in particular)

    - only works well when value function is sufficiently smooth

### Whether model includes stochastic processes

1. Deterministic models

    - No random elements, all motion rules deterministic
    - No need for expectation operator in Bellman equation

$$V(\text{state}) = \max_{\text{decisions}} \big[ U(\text{state},\text{decision}) + \beta \cdot V(\text{next state}) \big]$$

2. Stochastic models with idiosyncratic shocks

    - expectation does not have to be conditioned on current period shocks
    - which opens a cheaper way of writing the fixed point, and we use it on Thursday

3. General form stochastic models

    - expectation in Bellman equation has to be computed with quadrature or Monte Carlo integration


### Whether choice is part of the problem at all

- Some problems solved by the DP methods do not have explicit choices and maximization

 ````{tip} Example: tiling with dominoes

Given a $3 \times n$ board, find **the number of ways** to fill it with $2 \times 1$
dominoes. There is nothing to choose and nothing to maximize, yet the problem has
overlapping sub-problems and is solved by exactly the recursion DP is built on.

These are the three possible ways to fill up a $3 \times 2$ board, and one of the
many ways to tile a $3 \times 8$ board:

```{image} _static/img/tile1.jpg
:height: 100px
:align: center
```

```{image} _static/img/tile2.jpg
:height: 100px
:align: center
```

**Breaking the big problem into sub-problems.** Observe that at any stage of filling
up the board from the left, the last column can be in one of three configurations:
completely filled, denoted $A_n$, top corner empty, $B_n$, or bottom corner empty,
$C_n$, where $n$ counts the columns covered so far.

```{image} _static/img/tile3.jpg
:height: 180px
:align: center
```

Any other configuration of the last column is impossible to reach with $2 \times 1$
dominoes:

```{image} _static/img/tile4.jpg
:height: 150px
:align: center
```

**Defining the recursion.** A completely filled $3 \times n$ board ends either with
three horizontal dominoes on top of a filled $3 \times (n-2)$ board, or with a corner
configuration on $n-1$ columns completed by one more domino:

```{image} _static/img/tile5.jpg
:height: 120px
:align: center
```

A board with the top corner empty ends either with one vertical domino on top of a
filled $3 \times (n-1)$ board, or with two horizontal dominoes on top of a $B_{n-2}$
configuration:

```{image} _static/img/tile6.jpg
:height: 120px
:align: center
```

The case of $C_n$ is the mirror image of $B_n$, so $C_n = B_n$. Therefore for any $n$
we have

$$
\begin{aligned}
A_n &= A_{n-2} + 2 B_{n-1} \\
B_n &= A_{n-1} + B_{n-2}
\end{aligned}
$$

and the answer to the whole problem is given by $A_n$. The two sequences are computed
inductively from the initial conditions, which is the backward induction of the next
section run forward.

```{code-cell} python3
def WaysTileDominoes(n):
    '''Compute the number of ways to tile 3 x n area by 2x1 tiles'''
    A, B = [0] * (n + 1), [0] * (n + 1)
    A[0] = 1  # one way to tile 3x0
    A[1] = 0  # no way to tile 3x1
    B[0] = 0  # no way to tile 3x0 without a corner
    B[1] = 1  # one way to tile 3x1 without a corner
    for i in range(2, n+1):  # loop over 2,3,..,n
        A[i] = A[i-2] + 2 * B[i-1]
        B[i] = A[i-1] + B[i-2]
    return A[n]
```

```{code-cell} python3
for n in range(1, 20):
    print('There are', WaysTileDominoes(n), 'ways to tile the 3 by', n, 'board')
```
````


:::{div}
:class: discussion

For the last example:
- Does it make sense that the number of tilings zero for every odd $n$?
- What is the complexity of this algorithm in $n$?

For dynamic programming classification:
- Which of the five questions changes the *solution method* you have to write, and which only changes the code that evaluates the payoff?
- Where does the inventory model below sit in each of these classifications?
:::


## Example: Inventory management model

Consider the following problem in discrete time and finite horizon
$t=0,\dots,T$

The notation is:

- $x_t\ge 0$ is inventory at period $t$ measured in **discrete units**
- $d_t\ge 0$ is *potentially stochastic* demand at period $t$
- $q_t\ge 0$ is the order of new inventory
- $p$ is the profit per one unit of (supplied) good
- $c$ is the fixed cost of ordering any amount of new inventory
- $r$ is the cost of storing one unit of good

The sales in period $t$ are given by $s_t = \min\{x_t,d_t\}$.

Inventory to be stored till next period is given by
$k_t = \max\{x_t-d_t,0\} + q_t = x_{t+1}$.

The profit in period $t$ is given by

$$
\begin{array}{rcl}
\pi_t & = & p \cdot \text{sales}_t - r \cdot x_{t+1} - c \cdot (\text{order made in period }t) \\
& = & s_t p - k_t r - c \mathbb{1}\{q_t>0\} = \\
& = &  p \min\{x_t,d_t\} - r \big[ \max\{x_t-d_t,0\} + q_t \big] - c \mathbb{1}\{q_t>0\}
\end{array}
$$

Assuming all $q_t \ge 0$, let $\sigma =  \{q_t\}_{t=1,\dots,T}$ denote a feasible inventory policy.

The expected profit maximizing problem is given by

$${\max}_{\sigma} \mathbb{E}\Big[ \sum_{t=0}^{T} \beta^t \pi_t \Big],$$

where $\beta$ is discount factor.

### Bellman equation for the problem

:::{div}
:class: discussion

- What are states and decisions?
- What are the motion rules?
- What is the instantaneous payoff/utility?

When $d_t$ is stochastic the policy has to be a function of the period $t$ inventory $x_t$. What about deterministic $d_t$?

- What is the sequence of events (timing assumptions)?
    
- (beginning of period)
1. current inventory 
1. demand
1. order (choice)
1. stored inventory
- (end of period)

:::

So, both $x_t$ and $d_t$ are taken into account for the new order to be
made, forming the state space.


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


The expectation in the Bellman equation is taken over the distribution
of the next period demand $d_{t+1}$, which we assume is independent of
any other variables and across time (idiosyncratic), thus the
conditioning on $(x_t,d_t,s_t)$ can be meaningfully dropped.

Expectation can be written as an integral over the distribution of
demand $F(d)$, and since inventory is discrete it\'s natural to assume
demand is as well.

The integral then transforms into a sum over the possible value of
demand, weighted by their probabilities $pr(d)$

$$
\begin{array}{rcl}
V(x_t,d_t) 
&=& \max_{q_t \ge 0} \Big\{ s_t p - k_t r - c \mathbb{1}\{q_t>0\} 
+ \beta \int V\big( k_t, d \big) \partial F(d)  \Big\} \\
&=& \max_{q_t \ge 0} \Big\{ s_t p - k_t r - c \mathbb{1}\{q_t>0\} 
+ \beta \sum_d V\big( k_t, d \big) pr(d)  \Big\}
\end{array}
$$


## First deterministic demand

Let's focus first on a deterministic case: let $d$ be fixed and constant over time. How does the Bellman equation change?

In the deterministic case with fixed $d$, it can be simply dropped from the state space, and the Bellman equation can be simplified to

$$
V(x_t) = \max_{q_t \ge 0} \Big\{ p \min\{x_t,d\} - r \big[ \max\{x_t-d,0\} + q_t \big] - c \mathbb{1}\{q_t>0\}
$$
$$
+ \beta V\big( \max\{x_t-d,0\} + q_t \big) \Big\}
$$

:::{div}
:class: discussion

How does this convert to code? What is in rows, what is in columns?
:::


```{code-cell} python3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = [12, 8]

class inventory_model:
    '''Small class to hold model fundamentals and its solution'''

    def __init__(self,label='noname',
                 max_inventory=10,  # upper bound on the state space
                 c = 3.2,           # fixed cost of order
                 p = 2.5,           # profit per unit of good
                 r = 0.5,           # storage cost per unit of good
                 β = 0.95,          # discount factor
                 demand = 4         # fixed demand
                 ):
        '''Create model with default parameters'''
        self.label=label # label for the model instance
        self.c, self.p, self.r, self.β = c, p, r, β
        self.demand = demand
        # created dependent attributes (it would be better to have them updated when underlying parameters change)
        self.n = max_inventory+1    # number of inventory levels
        self.upper = max_inventory  # upper boundary on inventory
        self.x = np.arange(self.n)  # all possible values of inventory (state space)

    def __repr__(self):
        '''String representation of the model'''
        return 'Inventory model labeled "{}"\nParamters (c,p,r,β) = ({},{},{},{})\nDemand={}\nUpper bound on inventory {}' \
               .format (self.label,self.c,self.p,self.r,self.β,self.demand,self.upper)

    def sales(self,x,d):
        '''Sales in given period'''
        return np.minimum(x,d)

    def next_x(self,x,d,q):
        '''Inventory to be stored, becomes next period state'''
        return x - self.sales(x,d) + q

    def profit(self,x,d,q):
        '''Profit in given period'''
        return self.p * self.sales(x,d) - self.r * self.next_x(x,d,q) - self.c * (q>0)

```
```{code-cell} python3
model=inventory_model(label='test')
print(model)

q=np.zeros(model.n)
print('Current profits with zero orders\n',model.profit(model.x,model.demand,q))

```

```{code-cell} python3
# illustration of broadcasting in the inventory model
q=model.x[:,np.newaxis]  # column vector
print('Current inventory\n',model.x)
print('Current sales\n',model.sales(model.x,model.demand))
print('Current orders\n',q)
print('Next period inventory\n',model.next_x(model.x,model.demand,q))
print('Current profits\n',model.profit(model.x,model.demand,q))

```



## Backwards induction

**Backwards induction algorithm** is used to solve finite horizon models

```
Solver for the finite horizon dynamic programming problems

1. Start at t=T
1. Solve Bellman equation at t, record optimal choice  
1. Decrease t unless t=1, and return to previous step.  


As result, for all t=1,..,T have found the optimal choice (as a function of state)
```

First, we need to code up the Bellman equation

```{code-cell} python3
:tags: [hide-input]

def bellman(m,v0):
    '''Bellman equation for inventory model
       Inputs: model object
               next period value function
    '''
    # create the grid of choices (same as x), column-vector
    q = m.x[:,np.newaxis]
    # compute current period profit (relying on numpy broadcasting to get the matrix with choices in rows)
    p = m.profit(m.x,m.demand,q)
    # indexes for next period value with extrapolation using last value
    i = np.minimum(m.next_x(m.x,m.demand,q),m.upper)
    # compute the Bellman maximand
    vm = p + m.β*v0[i]
    # find max and argmax
    v1 = np.amax(vm,axis=0)   # maximum in every column
    q1 = np.argmax(vm,axis=0) # arg-maximum in every column = order volume
    return v1, q1
```

```{code-cell} python3
v = np.zeros(model.n)
for i in range(3):
    v,q = bellman(model,v)
    print('Value =',v,'Policy =',q,sep='\n',end='\n\n')

```

```{code-cell} python3
:tags: [hide-input]

def solver_backwards_induction(m,T=10,verbose=False):
    '''Backwards induction solver for the finite horizon case'''
    # solution is time dependent
    m.value  = np.zeros((m.n,T))
    m.policy = np.zeros((m.n,T))
    # main DP loop (from T to 1)
    for t in range(T,0,-1):
        if verbose:
            print('Time period %d\n'%t)
        j = t-1 # index of value and policy functions for period t
        if t==T:
            # terminal period: ordering zero is optimal
            m.value[:,j] = m.profit(m.x,m.demand,np.zeros(m.n))
            m.policy[:,j] = np.zeros(m.n)
        else:
            # all other periods
            m.value[:,j], m.policy[:,j] = bellman(m,m.value[:,j+1]) # next period to Bellman
        if verbose:
            print(m.value,'\n')
    # return model with updated value and policy functions
    return m

model = inventory_model(label='illustration')
model=solver_backwards_induction(model,T=5,verbose=True)
print('Optimal policy:\n',model.policy)
```

```{code-cell} python3
:tags: [hide-input]

def plot_solution(model):
    plt.step(model.x,model.value)
    plt.legend([f'{i+1}' for i in range(model.value.shape[1])])
    plt.title('Value function')
    plt.show()
    plt.step(model.x,model.policy)
    plt.legend([f'{i+1}' for i in range(model.policy.shape[1])])
    plt.title('Policy function (optimal order sizes)')
    plt.show()
plot_solution(model)

```

```{code-cell} python3

mod = inventory_model(label='production',max_inventory=50)
mod.demand = 15
mod.c = 5
mod.p = 2.5
mod.r = 1.4
mod.β = 0.975
mod = solver_backwards_induction(mod,T=15)
plot_solution(mod)
```

## Back to stochastic demand

Fixing $d$ was a convenience for the first implementation. Putting the randomness back
changes nothing about the method — backwards induction still runs from $T$ down to $1$,
one Bellman operator per period — and only two things about the code.

**The state is a pair.** With $d$ random the manager knows the inventory *and* the
realized demand when ordering, so the value function is defined on the grid of
$(x,d)$ rather than on $x$ alone, and so is the policy $q^\star(x,d)$.

**The Bellman maximand carries an expectation.** Next period demand $d'$ is not known
when the order is placed, so the continuation value is averaged over it:

$$
V_t(x,d) = \max_{q \ge 0} \Big\{ p \min\{x,d\} - r \big[ \max\{x-d,0\} + q \big]
- c \mathbb{1}\{q>0\}
$$
$$
+ \beta \sum_{d'} V_{t+1}\big( \max\{x-d,0\} + q, d' \big) pr(d') \Big\}
$$

with the terminal period unchanged: order nothing, collect the last period's profit,

$$
V_T(x,d) = p \min\{x,d\} - r \max\{x-d,0\}
$$

**A distribution for demand.** Take it discrete, on the same grid as the inventory, so
that no interpolation is needed anywhere. 

Let demand follow a **truncated geometric**
distribution on $ \{0,1,\dots,N\} $,

$$
pr_i = (1-\lambda)^i \lambda, \qquad i = 0,1,\dots,N,
$$

with the last probability corrected so that the vector sums to one — the truncated tail
is piled onto $ pr_N $. The parameter $\lambda \in (0,1)$ controls how fast demand dies
off; the mean of the untruncated distribution is $(1-\lambda)/\lambda$.

```{code-cell} python3
:tags: [hide-input]

N = 25                                    # upper bound of the inventory grid
k = np.arange(N+1)                        # possible values of demand

def demand_pr(lam, n=N+1):
    '''Truncated geometric probabilities over the grid, last one corrected'''
    pr = (1-lam)**np.arange(n) * lam
    pr[-1] = 1 - pr[:-1].sum()            # the truncated tail piles onto the last point
    return pr

rng = np.random.default_rng(2026)
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
for lam,clr in [(0.15,'#1A6D91'), (0.25,'#0E8A27'), (0.50,'#CB1515')]:
    pr = demand_pr(lam)
    ax1.step(k, pr, where='mid', color=clr, lw=2,
             label=f'$\\lambda$={lam:.2f}, mean {np.dot(k,pr):.1f}, $pr_N$={pr[-1]:.3f}')
ax1.set_title('Truncated geometric demand')
ax1.set_xlabel('demand $d$'); ax1.set_ylabel('probability')
ax1.legend(frameon=False); ax1.grid(alpha=.3)

lam, T = 0.25, 40
pr = demand_pr(lam)
d = rng.choice(k, size=T, p=pr)           # one path of realized demand
ax2.step(np.arange(T), d, where='mid', color='#0E8A27', lw=1.5)
ax2.axhline(np.dot(k,pr), color='k', ls='--', lw=1, label='mean demand')
ax2.set_title(f'One realization over {T} periods, $\\lambda$={lam:.2f}')
ax2.set_xlabel('period $t$'); ax2.set_ylabel('demand $d_t$')
ax2.legend(frameon=False); ax2.grid(alpha=.3)
plt.show()
```

Two things to read off the left panel. The distribution is *decreasing*, so on this
specification small demands are the likely ones, and $\lambda$ moves the mean without
changing that shape. And the mass sitting on the last grid point is the truncation: at
$\lambda=0.15$ it is $0.017$, small but not nothing, which is the price of capping
demand at the top of the inventory grid. If a run of the model ever leans on that point,
the grid is too short.

The right panel is what the manager actually faces — one path of realized demand. There
is no smoothness to exploit and no trend to extrapolate: each period is an independent
draw, which is exactly the assumption that lets us write the expectation as a sum with
fixed weights.

:::{div}
:class: discussion

- The sum over $d'$ sits inside the maximization over $q$. Does it have to?
- The value function is now a matrix rather than a vector. Which axis should hold what,
  and where does the expectation go in a vectorized implementation?
- Does the manager's *order* depend on $x$ and $d$ separately, or only on some
  combination of them? Look at the timing of events again.
:::



## Origin of the term *Dynamic Programming*

> The 1950's were not good years for mathematical research. We had a very interesting gentleman in Washington named Wilson. He was Secretary of Defence, and he actually had a pathological fear and hatred of the word "research".
> I'm not using the term lightly; I'm using it precisely. His face would
suffuse, he would turn red, and he would get violent if people used the term, research, in his presence. You can imagine how he felt, then, about the term, mathematical.
> Hence, I felt I had to do something to shield Wilson and the Air Force from the fact that I was really doing mathematics inside the RAND Corporation.
>
> What title, what name, could I choose?
>
> In the first place, I was interested in planning, in decision-making, in thinking. But planning, is not a good word for various reasons. I decided therefore to use the word, "programming".
> I wanted to get across the idea that this was dynamic, this was
multistage, this was time-varying.
>
> I thought, let's kill two birds with one stone. Let's take a word which has an absolutely precise meaning, namely dynamic, in the classical physical sense.
> It also has a very interesting property as an adjective, and that is
it's impossible to use the word, dynamic, in the pejorative sense.
>
> Thus, I thought dynamic programming was a good name. It was something
not even a Congressman could object to. So I used it as an umbrella for my activities.
> 
> --- 📖 Bellman's autobiography "The Eye of the Hurricane"

(task7.1)=
````{danger} Homework: the inventory model with stochastic demand

This is a graded homework assignment.

Implement the stochastic version set up above: truncated geometric demand, the value
function on the $(x,d)$ grid, and backwards induction over the finite horizon. The
notebook `tasks/epsilon_inventory/` in the class repository carries the deterministic
code of this class and the task to complete.

```bash
git pull upstream main                                    # collect the task
cp -r tasks/epsilon_inventory solutions/epsilon_inventory # work on the copy
```

The code shown in this class is `session07-sep15/inventory.ipynb` in the code
repository, which is worth having beside you while you work:

```bash
cd sb-dse-code && git pull
```

Remember to follow the git workflow <https://dse.iskh.me/workflow/#submission> to submit
your solution.

````


(7_dp_references)=
````{note} References and additional resources

- 📖 {cite:t}`Rust2016` "Dynamic programming", The New Palgrave Dictionary of Economics
- 📖 {cite:t}`sargent2025DynamicProgrammingFinite` "Dynamic Programming: Finite States"
- Online version of the same book on [dp.quantecon.org](https://dp.quantecon.org)
- 📖 {cite:t}`adda2023DynamicEconomicsQuantitative`, "Dynamic Economics: Quantitative Methods and Applications", chapters 2 and 3
- 📖 {cite:t}`aguirregabiriaDynamicDiscreteChoice2010` "Dynamic discrete choice structural models: A survey"
- Wiki: Bellman equation https://en.wikipedia.org/wiki/Bellman_equation
- Computer science view on DP [link](https://www.techiedelight.com/introduction-dynamic-programming)
- "Knowing When to Stop" by Theodore Hill, American Scientist [link](https://www.americanscientist.org/article/knowing-when-to-stop)

````
