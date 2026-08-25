---
title: 📖 Root finding and optimization
short_title: 📖 Solvers
subtitle: Class 4 — Thursday, September 3
kernelspec:
  name: python3
  display_name: Python 3
---

Two classic algorithms for solving $f(x)=0$, and what they become when the equation
we are solving is the first order condition of a likelihood. Every estimator in this
course sits on top of one of these.

## Bisection method

The first of two very important classic algorithms for equation solving.

Solve equations of the form (we focus on the scalar case today)

$$f(x) = 0, \quad x \in [a,b] \subset \mathbb{R}, \; f(a)f(b)<0$$

The latter condition requires that the function $f(x)$ takes different signs at the
endpoints $a$ and $b$.

The algorithm is similar to binary search, but in **continuous space**.

```
Input: function f(x)
       brackets [a,b] such that f(a)f(b)<0
       convergence tolerance epsilon
       maximum number of iterations max_iter

Algorithm:
  step 0: ensure all conditions are satisfied
  step 1: compute the sign of the function at (a+b)/2
  step 2: replace a with (a+b)/2 if f(a)f((a+b)/2)>0, otherwise replace b with (a+b)/2
  step 3: repeat steps 1-2 until |a-b| < epsilon, or max_iter number of iterations is reached
  step 4: return (a+b)/2
```

```{code-cell} python3
:tags: [hide-input, remove-output]

def bisection(f,a=0,b=1,tol=1e-6,maxiter=100,callback=None):
  '''Bisection method for solving equation f(x)=0
  on the interval [a,b], with given tolerance and number of iterations.
  Callback function is invoked at each iteration if given.
  '''
  if f(a)*f(b)>0:
    raise ValueError('Function has the same sign at the bounds')
  for i in range(maxiter):
    err = abs(b-a)
    if err<tol: break
    x = (a+b)/2
    a,b = (x,b) if f(a)*f(x)>0 else (a,x)
    if callback != None: callback(err=err,x=x,iter=i)
  else:
    raise RuntimeError('Failed to converge in %d iterations'%maxiter)
  return x
```

```{code-cell} python3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9, 6]

f = lambda x: -4*x**3+5*x+1
a,b = -3,-.5  # upper and lower limits
xd = np.linspace(a,b,1000)  # x grid
plt.plot(xd,f(xd),c='red')  # plot the function
plt.plot([a,b],[0,0],c='black')  # plot zero line
ylim=[f(a),min(f(b),0)]
plt.plot([a,a],ylim,c='grey')  # plot lower bound
plt.plot([b,b],ylim,c='grey')  # plot upper bound
def plot_step(x,**kwargs):
    plot_step.counter += 1
    plt.plot([x,x],ylim,c='grey')
plot_step.counter = 0  # new public attribute
bisection(f,a,b,callback=plot_step)
print('Converged in %d steps'%plot_step.counter)
plt.show()
```

Bisection is **slow but bulletproof**: given a valid bracket it always converges, at
a linear rate, halving the interval each step. Remember this when we get to solving
Bellman equations — a robust but slow method makes an excellent fallback inside a
poly-algorithm.

(task2.3)=
````{admonition} Practical task 2.3: implementing the bisection method
:class: warning

Implement the bisection method yourself, with the signature above, including:

- validation of the bracket
- a callback invoked at each iteration
- a convergence criterion on both $|a-b|$ and $|f(x)|$

Verify the linear convergence rate numerically by plotting $\log|x_i - x^\star|$
against the iteration number.
````

## Newton–Raphson method

The second of the two classic methods for solving an equation $f(x)=0$; gradient based.

General form

$$f(x)=0$$

- Equation solving
- Finding a maximum/minimum based on the FOC, in which case $f(x)=Q'(x)$

### Derivation using a Taylor series expansion

$$
f(x) = \sum_{k=0}^{\infty} \frac{f^{(k)}(x_0)}{k!} (x-x_0)^k
$$

Take the first two terms, assume $f(x)=0$ is the solution, and let $x_0=x_i$ and
$x=x_{i+1}$

$$
0 = f(x) = f(x_i) + f'(x_i) (x_{i+1}-x_i) \quad \Rightarrow \quad x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}
$$

The main idea of the Newton–Raphson method is to iterate on this equation starting
from some $x_0$

$$
x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}, \; i=1,2,\ldots
$$

It is applicable to systems of equations, in which case $x\in\mathbb{R}^n$ and
$f: \mathbb{R}^n \to \mathbb{R}^n$, with the derivative replaced by the Jacobian.

```
Input: function f(x)
       gradient function f'(x)
Algorithm:
1. Start with some good initial value
2. Update x using the Newton step above
3. Iterate until convergence
```

```{code-cell} python3
:tags: [hide-input, remove-output]

def newton(fun,grad,x0,tol=1e-6,maxiter=100,callback=None):
    '''Newton method for solving equation f(x)=0
    with given tolerance and number of iterations.
    Callback function is invoked at each iteration if given.
    '''
    for i in range(maxiter):
        x1 = x0 - fun(x0)/grad(x0)
        err = abs(x1-x0)
        if callback != None: callback(err=err,x0=x0,x1=x1,iter=i)
        if err<tol: break
        x0 = x1
    else:
        raise RuntimeError('Failed to converge in %d iterations'%maxiter)
    return (x0+x1)/2
```

```{code-cell} python3
:tags: [hide-input]

f = lambda x: -4*x**3+5*x+1
g = lambda x: -12*x**2+5
a,b = -3,-.5  # upper and lower limits
xd = np.linspace(a,b,1000)  # x grid
def plot_step(x0,x1,iter,**kwargs):
    plot_step.counter += 1
    if iter<5:
        plt.plot(xd,f(xd),c='red')  # plot the function
        plt.plot([a,b],[0,0],c='black')  # plot zero line
        ylim = [min(f(b),0),f(a)]
        plt.plot([x0,x0],ylim,c='grey') # plot x0
        l = lambda z: g(x0)*(z - x1)
        plt.plot(xd,l(xd),c='green')  # plot the tangent line
        plt.ylim(bottom=10*f(b))
        plt.title('Iteration %d'%(iter+1))
        plt.show()
plot_step.counter = 0  # new public attribute
newton(f,g,x0=-2.5,callback=plot_step)
print('Converged in %d steps'%plot_step.counter)
```

Newton–Raphson is **fast but fragile**: quadratic convergence near the root, but it
can diverge, cycle, or run off to a different root if started badly or if
$f'(x)$ is near zero.

### Measuring the complexity of Newton and bisection methods

- What is the size of the input $n$?
- The desired precision of the solution!
- Thus, attention to the errors in the solution as the algorithm proceeds
- The rate of convergence is part of the computational complexity of these algorithms

**Computational complexity**: calculating a root of a function $f(x)$ **with
$n$-digit precision**, provided that a good initial approximation is known, is
$O\big(\log(n) F(n)\big)$, where $F(n)$ is the cost of calculating $f(x)/f'(x)$ with
$n$-digit precision.

(task2.4)=
````{admonition} Practical task 2.4: implementing Newton–Raphson
:class: warning

Implement the Newton–Raphson method with the signature above, and

1. verify quadratic convergence numerically on the cubic used above
2. find a starting value from which the method fails to converge, and explain why
3. combine the two solvers: a poly-algorithm that starts with bisection to get into
   the neighborhood of the root, then switches to Newton for the final digits
````

(task2.5)=
````{admonition} Practical task 2.5 [optional]: multivariate Newton
:class: warning

Extend your Newton solver to $f: \mathbb{R}^n \to \mathbb{R}^n$ using the Jacobian
and `numpy.linalg.solve`. Test it on a small nonlinear system of your choice.

We need exactly this in Part III, where the Bellman operator is solved by
Newton–Kantorovich iterations.
````

## From root finding to optimization

Estimation problems in this course are of the form

$$
\hat{\theta} = \arg\min_{\theta \in \Theta} \frac{1}{N}\sum_{i=1}^N q(w_i;\theta)
$$

where $q$ is the negative log-likelihood contribution, a moment condition, or some
other criterion. In nonlinear models the optimum rarely has a closed form and must
be found numerically — by solving the first order conditions, which is exactly the
root finding problem above.

The relevant algorithms — Newton–Raphson, BHHH, BFGS — all belong to the class of
quasi-Newton methods with the form

$$
\theta^{(g+1)}
=
\theta^{(g)}
-
\lambda
\left(\sum_i H_i(\theta^{(g)})\right)^{-1}
\sum_i s_i(\theta^{(g)})
$$

where

- $g$ is the index of the iteration
- $\lambda$ is the *step size*
- $\sum_i H_i(\theta^{(g)})$ is the sample sum of the Hessians of $q(w_i;\theta)$
  evaluated at $\theta^{(g)}$, reflecting the curvature of $Q_N=\sum_i q(w_i;\theta)$
- $\sum_i s_i(\theta^{(g)})$ is the sample sum of the scores evaluated at
  $\theta^{(g)}$, measuring the slope of $Q_N$

### Newton–Raphson for optimization

Second order Taylor expansion of the objective function around $\theta^{(g)}$:

$$
\sum_{i=1}^N q_i(\theta^{(g+1)})
\approx
\sum_{i=1}^N q_i(\theta^{(g)})
+
(\theta^{(g+1)}-\theta^{(g)})'
\sum_{i=1}^N s_i(\theta^{(g)})
+
\frac{1}{2}
(\theta^{(g+1)}-\theta^{(g)})'
\sum_{i=1}^N H_i(\theta^{(g)})
(\theta^{(g+1)}-\theta^{(g)})
$$

First order condition:

$$
\sum_{i=1}^N s_i(\theta^{(g)}) +
\sum_{i=1}^N H_i(\theta^{(g)})
(\theta^{(g+1)}-\theta^{(g)})
= 0
$$

$$
\theta^{(g+1)}
=
\theta^{(g)}
-
\left(\sum_{i=1}^N H_i(\theta^{(g)})\right)^{-1}
\sum_{i=1}^N s_i(\theta^{(g)})
$$

**Properties**

- Uses slope and curvature
- Moves downhill if the Hessian is positive definite

### Step size $\lambda$ and global convergence

In addition to the plain Newton–Raphson update, it is typical to include a step size
search. This ensures **global convergence** of the algorithm, at a slower rate.

1. Start with the full Newton step, $\lambda=1$
2. If the step decreases the objective, proceed with $\lambda=1$
3. If the step does not decrease the objective, reduce $\lambda$ to half its value
4. Repeat steps 2–3 until an improving step is found

The step sizes are therefore in the sequence $\lambda=1,1/2,1/4,1/8,\ldots$, so the
approach is often called **step-halving line search**.

This addition is computationally cheap because the Hessian and the scores do not have
to be recomputed for different $\lambda$ values — only the function values.

Newton–Raphson works best when the objective is close to quadratic and convex. Yet
the Hessian may fail to be positive definite far from the optimum, leading to uphill
moves. The algorithm can be improved by ensuring that the curvature matrix is always
positive definite — which is exactly what BHHH does.

### BHHH

BHHH (Berndt, Hall, Hall and Hausman) approximates the Hessian by the outer product
of the scores:

$$
\theta^{(g+1)}
=
\theta^{(g)}
-
\lambda
\left(\sum_i s_i s_i'\right)^{-1}
\sum_i s_i
$$

**Motivation**

- Information identity: $\E(s_i s_i') \to \E(H_i)$
- Derived for the ML estimator, but works more generally for M-estimators

**Advantages**

- Always positive definite
- No need to compute the Hessian — only first derivatives

**Disadvantages**

- The approximation is valid only
  - at the true parameters
  - for large $N$
  - for well-specified models

### BFGS and other quasi-Newton methods

BFGS builds up an approximation to the inverse Hessian iteratively, ensuring positive
definiteness. Other approaches to approximating the Hessian, such as the DFP
(Davidon–Fletcher–Powell) method, were developed for general optimization problems.

BHHH, being based on the statistical properties of the scores, is more specific to
econometric M-estimation problems — and for that reason it is often *not* found in
standard optimization packages.

:::{admonition} Where this comes back
:class: note

The NFXP estimator in Part III is a Newton-type optimizer with analytical derivatives
on the outside, and a Newton–Kantorovich solver for the Bellman equation on the
inside. Both loops are the algorithms of this class, so it pays to have them clear now.

The full treatment of M-estimation, asymptotics and the information identity is
reference reading in the course — see the source notes listed below.
:::

(2_solvers_references)=
````{admonition} References and additional resources
:class: note

- On the computational complexity of Newton's method
  [link](https://m.tau.ac.il/~tsirel/dump/Static/knowino.org/wiki/Newton's_method.html#Computational_complexity)

- "Improved convergence and complexity analysis of Newton's method for solving equations"
  [link](https://www.tandfonline.com/doi/abs/10.1080/00207160601173431)

- 📺 Oscar Veliz videos on the Newton method and its domains of attraction
  - [Newton's method](https://www.youtube.com/watch?v=E24zUEKqgwQ)
  - [Basins of attraction](https://www.youtube.com/watch?v=zyXRo8Qjj0A)
  - [Newton fractals](https://www.youtube.com/watch?v=MWD2A0Vg2V0)

- 📖 {cite:t}`adda2023DynamicEconomicsQuantitative` "Dynamic Economics: Quantitative
  Methods and Applications" — numerical methods appendix
````
