---
title: 📖 Root finding and optimization
short_title: 📖 Classic solvers
subtitle: Class 4 — Thursday, September 3
exports:
  - format: typst
    output: exports/4_solvers.pdf
downloads:
  - file: 4_solvers.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

Two classic algorithms for solving $f(x)=0$, either as an equation or system of equations, or as FOC of an optimization problem.
These will be referred to in this course a countless number of times.

# Bisection method

Consider first the one-dimensional equations of the form

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
    if callback != None: callback(err=err,x=x,iter=i,a=a,b=b)
    a,b = (x,b) if f(a)*f(x)>0 else (a,x)
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
ylim = [-5,25]  # vertical zoom: what matters is the sign, the function runs off the top
def plot_step(a,b,x,iter,**kwargs):
    plot_step.counter += 1
    if iter<7:
        lo,hi = (x,b) if f(a)*f(x)>0 else (a,x)  # the half that is kept
        plt.plot(xd,f(xd),c='red')  # plot the function
        plt.plot([xd[0],xd[-1]],[0,0],c='black')  # plot zero line
        plt.fill_between([a,b],*ylim,color='grey',alpha=0.10)  # current bracket
        plt.fill_between([lo,hi],*ylim,color='green',alpha=0.15)  # the half to keep
        plt.plot([a,a],ylim,c='grey')  # plot lower bound
        plt.plot([b,b],ylim,c='grey')  # plot upper bound
        plt.plot([x,x],ylim,c='green')  # plot the midpoint
        plt.scatter([a,b,x],[f(a),f(b),f(x)],c=['grey','grey','green'],zorder=3)
        plt.title('Iteration %d: [%1.4f,%1.4f], f(x)=%+1.4f, keep [%1.4f,%1.4f]'
                  %(iter+1,a,b,f(x),lo,hi))
        plt.xlim(xd[0],xd[-1])
        plt.ylim(ylim)
        plt.show()
plot_step.counter = 0  # new public attribute
bisection(f,a,b,callback=plot_step)
print('Converged in %d steps'%plot_step.counter)
```

Bisection is **slow but bulletproof**: given a valid bracket it always converges, at
a linear rate, halving the interval each step. 

Remember this when we get to solving complex models — a robust but slow method makes an excellent fallback in the complex problems.

# Newton–Raphson method

The second of the two classic solution methods; this one is gradient based.

General form

$$f(x)=0$$

- Equation solving
- Finding a maximum/minimum based on the FOC, in which case $f(x)=Q'(x)$
- System of equations $f(x)=0$, $x\in\mathbb{R}^n$, $f:\mathbb{R}^n\to\mathbb{R}^n$

## Derivation of Newton step 

Remember the Taylor series expansion

$$
f(x) = \sum_{k=0}^{\infty} \frac{f^{(k)}(x_0)}{k!} (x-x_0)^k
$$

Take the first two terms around the solution $f(x)=0$, and let $x_0=x_i$ and
$x=x_{i+1}$ (step towards the solution)

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




# Rates of convergence and complexity

:::{div}
:class: discussion

- Which of the two algorithms is intuitively faster?
- Fundamentally why?

:::


````{attention} Definition

Let $x^\star$ denote the solution, $f(x^\star)=0$, and write $e_k = |x_k - x^\star|$ for the error at iteration $k$. The sequence converges

- **linearly** with rate $C \in (0,1)$ if $e_{k+1} \leq C\, e_k$
- **superlinearly** if $e_{k+1} \leq C_k e_k$ with $C_k \rightarrow 0$
- **quadratically** if $e_{k+1} \leq C\, e_k^2$ for some $C>0$
````


Continue the example above with a different `callback` function to print the errors at each iteration.


```{code-cell} python3
:tags: [hide-input]

xstar = -1.0  # f(-1) = 4-5+1 = 0, the exact root in the bracket

def print_err(iter,err,**kwargs):
    '''Report the error at each iteration of either solver.
    Bisection passes the midpoint as x, Newton–Raphson the new point as x1.
    '''
    x = kwargs['x'] if 'x' in kwargs else kwargs['x1']
    print('%3d:  x = %+1.14f   |x-x*| = %1.3e   reported err = %1.3e'
          %(iter,x,abs(x-xstar),err))

print('Bisection on [-3,-0.5]')
bisection(f,-3,-.5,callback=print_err)
print('\nNewton–Raphson from x0=-2.5')
newton(f,g,x0=-2.5,callback=print_err)
```

Two things stand out: 
- bisection needs 22 iterations where Newton needs 7! 
- unlike Newton, the error of bisection is **not monotone**: it grows at every second step, because the
midpoint of a bracket can easily sit further from the root than the previous midpoint
did. What halves exactly, every single time, is the *bracket*, and that is what the
solver reports as `err`.

The same callback mechanism, now storing and plotting the errors instead of printing them, gives the picture:

```{code-cell} python3
:tags: [hide-input]

def collect(store):
    '''Make a callback that appends (true error, reported error) to the given list'''
    def cb(iter,err,**kwargs):
        x = kwargs['x'] if 'x' in kwargs else kwargs['x1']
        store.append((abs(x-xstar),err))
    return cb

eb, en = [], []
bisection(f,-3,-.5,callback=collect(eb))
newton(f,g,x0=-2.5,callback=collect(en))
eb, en = np.array(eb), np.array(en)

plt.semilogy(np.arange(1,len(eb)+1),eb[:,0],'o',c='blue',label=r'bisection $|x_k-x^\star|$')
plt.semilogy(np.arange(1,len(eb)+1),eb[:,1],'-',c='blue',alpha=.5,label='bisection bracket width')
plt.semilogy(np.arange(1,len(en)+1),en[:,0],'o-',c='red',label=r'Newton–Raphson $|x_k-x^\star|$')
plt.xlabel('iteration $k$')
plt.legend()
plt.grid(True,which='both',alpha=.3)
plt.show()
```

On a log scale linear convergence is a **straight line** — the error is multiplied by
the same factor every iteration. 

Quadratic convergence is a **cliff**: the number of
correct digits doubles, so the curve bends down without limit until it hits machine
precision.


```{hint} What the solver can actually measure

In an application $x^\star$ is unknown, so the true error of the left column above is
not available and the solver has to stop on something it can compute: the bracket
width for bisection, the step size $|x_{k+1}-x_k|$ for Newton.

For bisection that quantity is an honest upper bound on the error. For Newton it lags
by one iteration — look at the last row, where the reported error is $1.4\cdot10^{-7}$
while the true error is already $3.4\cdot10^{-14}$. Quadratic convergence makes the
step size a very conservative stopping criterion, so `tol` in a Newton solver buys far
more accuracy than it asks for.
```

:::{div}
:class: discussion

- Bisection took 22 iterations and Newton 7. Is that the right way to compare them?
- The Newton curve stops at $10^{-14}$ and cannot go lower. Why?
:::

## Algorithmic complexity

Let's take a loot at the algorithmic complexity of the two methods now.

:::{div}
:class: discussion

- What is the size of the problem $n$ for the solvers?
:::

Given the importance of the *the desired precision of the solution*, the rate of
convergence is therefore part of the computational complexity of the algorithm, and
the two questions to ask of each method are how many iterations it needs to reach a
given accuracy, and what one iteration costs.
 
Suppose we want to find a root $x^\star$ of $f(x^\star) = 0$ to accuracy $\varepsilon$.

### Bisection: one binary digit per iteration

Starting from an interval of width $L=b-a$, each iteration halves the interval, so
after $k$ steps the error is bounded by

$$
e_k \leq \frac{L}{2^k}
$$

To achieve $e_k \leq \varepsilon$ we need $L/2^k \leq \varepsilon$, that is

$$
k \geq \log_2\left(\frac{L}{\varepsilon}\right)
\quad \Longrightarrow \quad
k_{\text{bisection}} = O\left(\log\frac{1}{\varepsilon}\right)
$$

Each iteration gains approximately **one binary digit of accuracy**, whatever the
function looks like. That is the flip side of being bulletproof: the bound holds
always, and it is never beaten.

### Newton–Raphson: doubling the number of digits

Under certain regularity conditions the error of the Newton
iteration satisfies

$$
e_{k+1} \approx C e_k^2
$$

where constant $C$ depends on the $f(x)$ and $f'(x)$.

The number of correct digits approximately **doubles with each iteration**.

Schematically $e_k \sim C' e_0^{2^k}$, and to obtain $e_k \leq \varepsilon$ we need
$2^k = O\big(\log(1/\varepsilon)\big)$, hence

$$
k_{\text{Newton}} = O\left(\log\log\frac{1}{\varepsilon}\right)
$$

Counted in digits rather than in $\varepsilon$, the same statement reads: calculating a
root of $f(x)$ **with $p$-digit precision**, provided that a good initial approximation
is known, costs $O\big(\log(p) F(p)\big)$, where $F(p)$ is the cost of calculating
$f(x)/f'(x)$ with $p$-digit precision. Both the number of iterations and the price of
a single iteration grow with the precision demanded.

Note the words *locally* and *provided that a good initial approximation is known*.
Everything in this subsection is conditional on being close enough to the root!

### Multivariate Newton: the cost of one iteration

In $n$ dimensions, for $f:\mathbb{R}^n\rightarrow\mathbb{R}$, the Newton step is no
longer a division but the solution of a linear system

$$
H(x_k)\,\Delta x = -\nabla f(x_k)
$$

For a dense $n \times n$ Hessian that costs $O(n^3)$ per iteration, so once the
iterations are in the region of quadratic convergence the overall cost is roughly

$$
O\left(n^3\log\log\frac{1}{\varepsilon}\right)
$$

The iteration count barely depends on the accuracy we ask for; the dimension of the
problem is what hurts. This is why the structure of the Jacobian matters so much in
practice — sparsity, block structure, or an analytical inverse turn the $n^3$ into
something affordable.


| Method | Convergence | Iterations for accuracy $\varepsilon$ | Cost per iteration |
| :-- | :-- | :-- | :-- |
| **Bisection** | linear | $O\big(\log(1/\varepsilon)\big)$ | one evaluation of $f$ |
| **Newton–Raphson** | quadratic, locally | $O\big(\log\log(1/\varepsilon)\big)$ | one evaluation of $f$ and $f'$, up to $O\big(n^3)$ |

The key distinction between the two methods is therefore:

- **Bisection** — cheap iterations, linear convergence, no way to fail
- **Newton** — expensive iterations, quadratic local convergence, and no guarantee of
  getting there at all



:::{div}
:class: discussion

- Can we get the best of the both approaches? How?
:::



# When does Newton–Raphson fail?

Newton–Raphson is **fast but fragile**: quadratic convergence near the root, but what happens away from the root?


Five things can go wrong, in rough order of how often
they bite in economic applications:

1. **Multiple solutions** — the method converges, but to a root you did not want
2. **Divergence** — the iterations run away, or converge to a root far outside the
   region of interest: the *domain of attraction* problem
3. **Cycles** — the iterations enter a loop and never converge
4. **Function domain and differentiability** — a Newton step leaves the region where
   $f(x)$ or $f'(x)$ is defined
5. **Reduced performance** — quadratic convergence is lost in special cases


## Multiple solutions

The function of the first example has three roots.

Which one Newton returns is
decided by the starting value alone, and the mapping from starting values to roots is
neither monotone nor otherwise well behaved.

Starting Newton-Raphson from $x_0 \in \{-0.565,-0.580,-0.595\}$

```{code-cell} python3
:tags: [hide-input]

def newton_pic(f,g,x0,a=0,b=1,**kwargs):
    '''Illustrate the Newton method iterations on the interval [a,b]'''
    xd = np.linspace(a,b,1000)
    plt.plot(xd,f(xd),c='red')       # the function
    plt.plot([a,b],[0,0],c='black')  # zero line
    def plot_step(**kw):
        plot_step.counter += 1
        z0,z1 = kw['x0'],kw['x1']
        plt.plot([z0,z0],[0,f(z0)],c='green')  # from the axis up to the function
        plt.plot([z0,z1],[f(z0),0],c='green')  # tangent line down to the axis
    plot_step.counter = 0
    try:
        xs = newton(f,g,x0,callback=plot_step,**kwargs)
        plt.title('Started at %1.3f, converged to %1.5f in %d steps'%(x0,xs,plot_step.counter))
    except RuntimeError:
        plt.title('Started at %1.3f, failed to converge in %d iterations'%(x0,plot_step.counter))
    plt.xlim((a,b))
    plt.show()

f = lambda x: -4*x**3+5*x+1  # function
g = lambda x: -12*x**2+5     # derivative
for x0 in [-0.565,-0.58,-0.595]:
    newton_pic(f,g,x0,a=-3,b=1.5)

```

*Be aware* is all that can be said here: multiple roots are a property of the problem,
not of the algorithm. 

In structural work this is the situation where the same solver,
started from different points, produces different estimates.

```{hint} Multistarts

It is a very good practice to run your solver and estimator multiple times from different starting values to check the robustness of the results.  This technique is referred to as **multistarts**.

```

## Divergence and the domain of attraction

$\arctan(x)=0$ has a single root at $x=0$, and Newton still fails from a starting
point that is only slightly too far out: the tangent overshoots, and each step
overshoots by more than the last.

Starting values $x_0 \in \{1.25, 1.50\}$

```{code-cell} python3
:tags: [hide-input]

f = lambda x: np.arctan(x)
g = lambda x: 1/(1+x**2)
newton_pic(f,g,x0=1.25,a=-20,b=20)            # inside the domain of attraction
newton_pic(f,g,x0=1.5,a=-20,b=20,maxiter=8)   # outside it
```


## Cycles

The iterations can also enter a cycle, returning to the same point every second step.

In the example below I had to compute the appropriate starting value numerically (using Newton method to the equation "two Newton steps return me where I started"), but this can happen in practice due to analytical properties of the function being solved. Important to be aware of such possibility.

```{code-cell} python3
:tags: [hide-input]

f = lambda x: -4*x**3+5*x+1  # function
g = lambda x: -12*x**2+5     # derivative
h = lambda x: -24*x          # second derivative

ns = lambda x: x - f(x)/g(x)          # the Newton step itself
ds = lambda x: f(x)*h(x)/g(x)**2      # its derivative
f2 = lambda x: ns(ns(x)) - x          # two Newton steps return to the start
g2 = lambda x: ds(ns(x))*ds(x) - 1    # derivative of the above

x0 = newton(f2,g2,x0=-0.56,tol=1e-16)  # find the cycling starting point
print('To cycle start with x0 = %1.16f'%x0)

newton_pic(f,g,x0,a=-1.5,b=1.5,maxiter=15)
```

## Function domain and differentiability

Nothing in the Newton step knows where $f(x)$ is defined. Solving $\log(x)=0$ from
$x_0=2.9$, the very first step lands on a negative number and every value after that
is `nan`.

```{code-cell} python3
:tags: [hide-input]

f = lambda x: np.log(x)
g = lambda x: 1/x
newton_pic(f,g,x0=2.9,a=0.001,b=3)
```

This is the failure mode to expect in many structural models. 

The usual remedies are reparametrization (solve in $\exp x$ rather than
$x$) or specialized algorithms.

## Suboptimal performance

Quadratic convergence requires a simple root, $f'(x^\star) \ne 0$. At a root of
multiplicity $m$ the method still converges, but only linearly, at rate $1 - 1/m$.
For $f(x)=x^9$ that is $8/9$: a hundred iterations to reach the tolerance that the
first example reached in seven.

```{code-cell} python3
:tags: [hide-input]

f = lambda x: x**9   # a very special case: root of multiplicity 9
g = lambda x: 9*x**8
newton_pic(f,g,x0=1.0,a=-1.5,b=1.5)

def print_err(**kwargs):
    if kwargs['iter'] % 5 == 0:
        print('{:4d}:  x = {:17.14f}  err = {:8.6e}'.format(kwargs['iter'],kwargs['x1'],kwargs['err']))
newton(f,g,x0=1.0,callback=print_err)

```


## What this means for economics

- Many of the pathological cases above rarely occur in practice
- The ones that are real in economic applications are
  1. multiple solutions
  2. constraints on the domain of the function
  3. the domain of attraction
- It is therefore a good idea to **combine a slow robust solver** — to get into the
  domain of attraction — **with the Newton method** for fast convergence at the end
- Successive approximations followed by Newton iterations is the classic approach in
  the estimation of dynamic programming models, namely the NFXP estimator of
  {cite:t}`rustOptimalReplacementGMC1987`







# Multivariate Newton and optimization

Everything above is scalar, and everything above is about *solving an equation*. 

But the natural multivariate problem in economics is not a system of equations — it is the first order condition of an optimization problem.

Consider $F(x) \rightarrow \max$ with $x \in \mathbb{R}^n$. 
The first order conditions for the interior optimum are

$$
\nabla F(x) = 0
$$

which is a system of $n$ equations in $n$ unknowns. 
Let's use Newton-Raphson to solve it.

## Derivation from the two-dimensional Taylor expansion

The scalar Newton step came from a first order Taylor expansion of $f$ around $x_i$.
In the optimization problems the step comes from a **second order** expansion of the objective — one
order higher, because we are now looking for the gradient of the derivative in the FOC.

Take $n=2$ and write the step as $(h,k)$. Around the current point $(x_i,y_i)$,

$$
F(x_i+h,\,y_i+k) \approx
F + F_x h + F_y k +
\tfrac{1}{2}\big(F_{xx}h^2 + 2F_{xy}hk + F_{yy}k^2\big)
$$

where every derivative is evaluated at $(x_i,y_i)$. The right hand side is a quadratic
in $(h,k)$ — a paraboloid — and we can find *its* stationary point exactly. Its two
first order conditions are

$$
\frac{\partial}{\partial h}:\; F_x + F_{xx}h + F_{xy}k = 0,
\qquad
\frac{\partial}{\partial k}:\; F_y + F_{xy}h + F_{yy}k = 0
$$

which is a $2\times 2$ **linear** system for the step:

$$
\begin{pmatrix} F_{xx} & F_{xy} \\ F_{xy} & F_{yy} \end{pmatrix}
\begin{pmatrix} h \\ k \end{pmatrix}
=
-\begin{pmatrix} F_{x} \\ F_{y} \end{pmatrix}
$$

The matrix on the left is the **Hessian** $\nabla^2 F$, the vector on the right the
**gradient** $\nabla F$. Solving the system by hand,

$$
\begin{pmatrix} h \\ k \end{pmatrix}
=
-\frac{1}{F_{xx}F_{yy}-F_{xy}^2}
\begin{pmatrix} F_{yy} & -F_{xy} \\ -F_{xy} & F_{xx} \end{pmatrix}
\begin{pmatrix} F_{x} \\ F_{y} \end{pmatrix}
$$

and the iteration $x_{i+1} = x_i + (h,k)$ is the **multivariate Newton method**

$$
x_{i+1} = x_{i} - \big( \nabla^2 F(x_i) \big)^{-1} \nabla F(x_i)
$$

Three observations before any code.

- It is the scalar formula with $1/f'(x_i)$ replaced by $\big(\nabla^2F(x_i)\big)^{-1}$.
  The scalar case is $n=1$, nothing more.
- The determinant $F_{xx}F_{yy}-F_{xy}^2$ sits in the denominator. It is the
  two-dimensional counterpart of $f'(x_i)\approx 0$ in the scalar method: when the
  Hessian is close to singular the step explodes. Everything the failure-mode section
  said about $f'$ applies to the determinant here.
- The derivation used only the *first* order condition of the quadratic model.
  Nothing in it distinguishes a maximum from a minimum or a saddle point — a fact we
  will pay for below.

For a general system $G(x)=0$ with $G: \mathbb{R}^n \to \mathbb{R}^n$ the same
argument on a first order expansion of $G$ gives

$$
x_{i+1} = x_{i} - \big( \nabla G(x_i) \big)^{-1} G(x_i)
$$

where $\nabla G$ is the **Jacobian**, the $n\times n$ matrix of partial derivatives.
Optimization is the special case $G = \nabla F$, whose Jacobian is the Hessian
$\nabla G = \nabla^2 F$.

```{hint} Do not invert the matrix

Written with an inverse, implemented with a solve: `numpy.linalg.solve(H,G)` computes
the step from the linear system directly; `numpy.linalg.inv(H) @ G` forms the inverse
first, which costs more and loses accuracy. The inverse of the Hessian is worth
computing only when it is wanted for itself — as the asymptotic variance matrix of an
estimator, for instance.
```

## An objective function with many critical points

Consider the following objective function which is the sum of four Gaussian bumps, one of which is a banana-shaped ridge:

$$
F(x,y)=\sum_{i=1}^{4} a_i \exp\big(-q_i(x,y)\big),
\qquad
(a_1,a_2,a_3,a_4) = (1.00,\,0.94,\,0.78,\,1.20)
$$

with

$$
\begin{aligned}
q_1 &= \frac{(x-0.22)^2}{0.018}+\frac{(y-0.27)^2}{0.030}, &
q_2 &= \frac{(x-0.73)^2}{0.080}+\frac{(y-0.25)^2}{0.012}, \\
q_3 &= \frac{(x-0.55)^2}{0.055}+\frac{\big[y-0.63+2.5(x-0.55)^2\big]^2}{0.0025}, &
q_4 &= \frac{(x-0.55)^2}{0.005}+\frac{(y-0.48)^2}{0.010}
\end{aligned}
$$


```{code-cell} python3

def quad(x,y,c,d,A,B):
    '''Exponent q of a plain Gaussian bump, with its gradient and Hessian'''
    q   = (x-c)**2/A + (y-d)**2/B
    dq  = [2*(x-c)/A, 2*(y-d)/B]
    d2q = [[2/A+0*x, 0*x],[0*x, 2/B+0*x]]
    return q,dq,d2q

def quad_banana(x,y):
    '''Exponent q of the curved ridge, with its gradient and Hessian'''
    u = x - 0.55
    v = y - 0.63 + 2.5*u**2
    q   = u**2/0.055 + v**2/0.0025
    dq  = [2*u/0.055 + 4000*u*v, 800*v]
    d2q = [[2/0.055 + 4000*v + 20000*u**2, 4000*u],[4000*u, 800+0*x]]
    return q,dq,d2q

def bumps(x,y):
    '''The four terms of F: amplitude a, exponent q, and the derivatives of q'''
    return [(1.00,) + quad(x,y,0.22,0.27,0.018,0.030),
            (0.94,) + quad(x,y,0.73,0.25,0.080,0.012),
            (0.78,) + quad_banana(x,y),
            (1.20,) + quad(x,y,0.55,0.48,0.005,0.010)]

def F(x,y):
    return sum(a*np.exp(-q) for a,q,dq,d2q in bumps(x,y))
```

```{code-cell} python3
:tags: [hide-input]

plt.rcParams['figure.figsize'] = [9, 6]
def contour_plot(fun,levels=30,xlim=(0,1.2),ylim=(0,0.72),npoints=200,ax=None,clip=None):
    '''Contour plot of a function of two variables.
    With clip=p the levels are symmetric around zero and cut at the p-th percentile
    of |Z|, which keeps a few extreme values from swamping the picture.
    '''
    X,Y = np.meshgrid(np.linspace(*xlim,npoints),np.linspace(*ylim,npoints))
    Z = fun(X,Y)
    if clip is None:
        lv = np.linspace(Z.min(),Z.max(),levels)
        lv = np.concatenate([np.exp(np.linspace(np.log(Z.min()),np.log(0.2),levels//2)),np.linspace(0.3,Z.max(),levels//2)])
    else:
        c = np.percentile(np.abs(Z),clip)
        lv = np.linspace(-c,c,levels)
    if ax is None:
        fig, ax = plt.subplots()
    ax.contour(X,Y,Z,levels=lv)
    ax.set_aspect('equal','box')
    ax.set_xlim(*xlim)  # fix the window: paths drawn on top may leave it
    ax.set_ylim(*ylim)
    return ax

contour_plot(F)
plt.show()
```

Three peaks, a ridge running over the top, and long flat plains towards the corners
where all four exponentials have died out. A grid search over the square finds eleven
critical points: four maxima, five saddle points and two minima. 

The global maximum is $F=1.2107$ sits at $(0.5507,0.4789)$ at the central peak.

## Gradient and Hessian

Writing each term as $g_i = a_i \exp(-q_i)$ saves us from differentiating four
different functions. The chain rule gives, for each term,

$$
\nabla g_i = -g_i \nabla q_i,
\qquad
\nabla^2 g_i = g_i \big[ (\nabla q_i)(\nabla q_i)^{\top} - \nabla^2 q_i \big]
$$

so that

$$
\nabla F = -\sum_{i=1}^4 g_i \nabla q_i,
\qquad
\nabla^2 F = \sum_{i=1}^4 g_i \big[ (\nabla q_i)(\nabla q_i)^{\top} - \nabla^2 q_i \big]
$$

Only the derivatives of the exponents $q_i$ are left to compute. For a plain Gaussian
term with $q=\frac{(x-c)^2}{A}+\frac{(y-d)^2}{B}$,

$$
\nabla q =
\begin{pmatrix} 2(x-c)/A \\ 2(y-d)/B \end{pmatrix},
\qquad
\nabla^2 q =
\begin{pmatrix} 2/A & 0 \\ 0 & 2/B \end{pmatrix}
$$

and for the ridge, writing $u = x-0.55$ and $v = y-0.63+2.5u^2$,

$$
\nabla q_3 =
\begin{pmatrix} 2u/0.055 + 4000\,uv \\ 800\,v \end{pmatrix},
\qquad
\nabla^2 q_3 =
\begin{pmatrix} 2/0.055 + 4000\,v + 20000\,u^2 & 4000\,u \\ 4000\,u & 800 \end{pmatrix}
$$

The `bumps()` function above already returns $q_i$, $\nabla q_i$ and $\nabla^2 q_i$ for
all four terms, so the gradient and the Hessian of $F$ are two short loops:

```{code-cell} python3
def G(x,y):
    '''Gradient of F: grad(a exp(-q)) = -a exp(-q) grad(q)'''
    out = [0,0]
    for a,q,dq,d2q in bumps(x,y):
        g = a*np.exp(-q)
        for k in range(2):
            out[k] = out[k] - g*dq[k]
    return out

def H(x,y):
    '''Hessian of F: H(a exp(-q)) = a exp(-q) [grad(q) grad(q)' - H(q)]'''
    out = [[0,0],[0,0]]
    for a,q,dq,d2q in bumps(x,y):
        g = a*np.exp(-q)
        for k in range(2):
            for j in range(2):
                out[k][j] = out[k][j] + g*(dq[k]*dq[j] - d2q[k][j])
    return out
```

```{hint} Always check analytical derivatives numerically

Hand-derived gradients and Hessians are a classic source of silent bugs: the solver
still runs, it just converges to the wrong point or not at all. Compare against a
central finite difference before trusting them.
```

```{code-cell} python3
:tags: [hide-input]

eps = 1e-6
for x,y in [(0.30,0.70),(0.55,0.50),(0.80,0.20)]:
    Gn = [(F(x+eps,y)-F(x-eps,y))/(2*eps), (F(x,y+eps)-F(x,y-eps))/(2*eps)]
    Hn = [[(G(x+eps,y)[k]-G(x-eps,y)[k])/(2*eps) for k in range(2)],
          [(G(x,y+eps)[k]-G(x,y-eps)[k])/(2*eps) for k in range(2)]]
    print('at (%4.2f,%4.2f):  max |G-Gnum| = %.2e   max |H-Hnum| = %.2e'
          % (x,y,np.amax(np.abs(np.array(G(x,y))-np.array(Gn))),
                 np.amax(np.abs(np.array(H(x,y))-np.array(Hn).T))))
```

The two components of the gradient, and the four elements of the Hessian, are
themselves functions of $(x,y)$ — the Newton step is a *local* linear approximation
built from them at the current point. Note how much sharper their variation is than
that of $F$ itself, and that $H_{12}$ and $H_{21}$ are the same picture, as they must be.

```{code-cell} python3
:tags: [hide-input]

fig, axs = plt.subplots(1,2,figsize=(11,5))
for k in range(2):
    contour_plot(lambda x,y: G(x,y)[k],ax=axs[k],clip=95,levels=20)
    axs[k].set_title('$G_%d(x,y)$'%(k+1))
plt.show()

fig, axs = plt.subplots(1,2,figsize=(11,10))
k=0
for j in range(2):
    contour_plot(lambda x,y: H(x,y)[k][j],ax=axs[j],clip=95,levels=20)
    axs[j].set_title('$H_{%d%d}(x,y)$'%(k+1,j+1))
plt.show()

fig, axs = plt.subplots(1,2,figsize=(11,10))
k=1
for j in range(2):
    contour_plot(lambda x,y: H(x,y)[k][j],ax=axs[j],clip=95,levels=20)
    axs[j].set_title('$H_{%d%d}(x,y)$'%(k+1,j+1))
plt.show()
```

## Multivariate Newton solver

The scalar solver carries over with two changes: the division becomes a linear solve,
and the error is measured with a vector norm.

```{code-cell} python3
def newton2(fun,grad,x0,tol=1e-6,maxiter=100,callback=None):
    '''Newton method for solving a system of equations fun(x)=0,
    where x is a vector of 2 elements and grad is the Jacobian.
    Callback function is invoked at each iteration if given.
    '''
    # conversion to array function of array argument
    npfun  = lambda x: np.asarray(fun(x[0],x[1]))
    npgrad = lambda x: np.asarray(grad(x[0],x[1]))
    x0 = np.asarray(x0,dtype=float)
    for i in range(maxiter):
        x1 = x0 - np.linalg.solve(npgrad(x0),npfun(x0))  # matrix version
        err = np.amax(np.abs(x1-x0))  # vector sup norm
        if callback != None: callback(iter=i,err=err,x0=x0,x1=x1)
        if err<tol: break
        x0 = x1
    else:
        raise RuntimeError('Failed to converge in %d iterations'%maxiter)
    return x1
```

```{code-cell} python3
:tags: [hide-input, remove-output]

def newton_path(x0,**kwargs):
    '''Solve G(x)=0 from x0, recording the sequence of Newton steps'''
    path = [np.asarray(x0,dtype=float)]
    xs = newton2(G,H,x0,callback=lambda **kw: path.append(kw['x1']),**kwargs)
    return xs, np.array(path)

def plot_newton_path(x0,**kwargs):
    '''Plot the Newton iterations on top of the contours of F'''
    xs, path = newton_path(x0,**kwargs)
    ax = contour_plot(F)
    ax.plot(path[:,0],path[:,1],c='r',marker='.')  # the path
    ax.scatter(*path[0],c='r',marker='o')          # starting point
    ax.scatter(*path[-1],c='b',marker='*',s=150,zorder=3)  # solution
    ax.set_title('Started at (%1.2f,%1.2f), converged in %d steps'%(*path[0],len(path)-1))
    plt.show()
```

Started just below the highest peak, the quadratic approximation is a good model of
the objective and the solver walks straight up to it in four steps:

```{code-cell} python3
plot_newton_path([0.55,0.45])
```

Now move the starting point down by $0.05$, still on the flank of the same peak. The
first step jumps $0.2$ downward into the low ground between the two lower peaks, the
second lands where $F$ is a fifteenth of its value at the start, and from there the
method wanders back up and settles on a saddle point — not on the peak it started
from:

```{code-cell} python3
plot_newton_path([0.55,0.40])
```



To see how little it takes, draw 10 Newton paths from random starting points in a small area
in the center. Stars mark the paths that converged and crosses the point where a run gave up —
for the plain solver most of those are far outside the picture, after `maxiter=100` steps.

```{code-cell} python3
:tags: [hide-input]

def newton_trace(x0,solver=newton2,**kwargs):
    '''Newton path from x0, with the reason the solver stopped'''
    path = [np.asarray(x0,dtype=float)]
    try:
        solver(G,H,x0,callback=lambda **kw: path.append(kw['x1']),**kwargs)
        status = 'converged'
    except np.linalg.LinAlgError:
        status = 'singular Hessian'
    except RuntimeError as e:
        status = 'no ascent direction' if 'ascent' in str(e) else 'maxiter reached'
    return np.array(path), status

center, side = np.array([0.6,0.43]), 0.05  # the small square of starting points
starts = center + np.random.default_rng(44).uniform(-side/2,side/2,size=(10,2))
colors = plt.cm.autumn(np.linspace(0,0.85,len(starts)))  # slightly different colors

def plot_newton_paths(solver=newton2,starts=starts,colors=colors,**kwargs):
    '''Follow all the starting points at once, each path in its own color'''
    ax = contour_plot(F)
    for c,x0 in zip(colors,starts):
        path, status = newton_trace(x0,solver=solver,**kwargs)
        ax.plot(path[:,0],path[:,1],c=c,marker='.',lw=1.2,zorder=2)  # the path
        ax.scatter(*path[0],c=[c],marker='o',s=25,zorder=3)          # starting point
        if status == 'converged':
            ax.scatter(*path[-1],c=[c],marker='*',s=140,edgecolors='k',lw=.4,zorder=4)
        else:  # mark where the solver gave up
            ax.scatter(*path[-1],c=[c],marker='X',s=80,edgecolors='k',lw=.4,zorder=4)
    ax.scatter([],[],c='grey',marker='o',s=25,label='start')            # legend only
    ax.scatter([],[],c='grey',marker='*',s=140,edgecolors='k',lw=.4,label='converged')
    ax.scatter([],[],c='grey',marker='X',s=80,edgecolors='k',lw=.4,label='gave up')
    ax.legend(loc='upper left',fontsize=8,framealpha=0.8)
    plt.show()

plot_newton_paths()
```

The starting points are withing a square of $(0.6,0.43)$ with side $0.05$!
Only a few paths converge, to different critical points; many leave the picture, several crossing directly over a peak on the way out without stopping at it.

## Newton finds critical points, not maxima

The solver was given the first order condition $\nabla F(x)=0$, and that is all it
knows. Any point satisfying it — a maximum, a minimum or a saddle point — is a
solution, and which one is reached depends entirely on the starting value. The
classification is ours to make afterwards, from the eigenvalues of the Hessian.

```{code-cell} python3
:tags: [hide-input]

for x0 in [(0.55,0.45),(0.55,0.40),(0.20,0.30),(0.75,0.25),
           (0.70,0.50),(0.55,0.65),(0.50,0.50),(0.10,0.80)]:
    try:
        xs, path = newton_path(x0)
        ev = np.linalg.eigvalsh(np.asarray(H(*xs)))  # Hessian at the solution
        kind = 'maximum' if (ev<0).all() else 'minimum' if (ev>0).all() else 'saddle'
        print('from (%4.2f,%4.2f) --> (%6.4f,%6.4f)  F = %5.3f  %-7s  in %2d steps'
              % (*x0,*xs,F(*xs),kind,len(path)-1))
    except RuntimeError as e:
        print('from (%4.2f,%4.2f) --> %s'%(*x0,e))
```

:::{div}
:class: discussion

- Two of the starting points above are 0.05 apart and end up at different critical
  points, one of them not even a maximum. What does that say about the domain of
  attraction of each solution?
- How would you check, in code, that the point returned is the maximum you wanted?
- What does a nearly singular Hessian do to the length of the step?
:::

The lesson is the one from the scalar case, only sharper in higher dimensions: the
failure modes do not go away, the domains of attraction are harder to picture, and the
poly-algorithm remedy — a slow robust method first, Newton for the last digits — is
correspondingly more valuable.

(line_search)=
# Line search and step size

Look again at the fan of paths above. What goes wrong is rarely the *direction* of the
Newton step — near the current point the local quadratic approximation is a reasonable model, and it
points the right way. What goes wrong is the *length*. The Newton step jumps all the
way to the stationary point of the quadratic approximation, and when it is a poor
description of $F$ further out, it lands somewhere unrelated to the problem.

The remedy is to keep the direction and shorten the step. Write the full Newton step as

$$
\Delta_i = \big(\nabla^2 F(x_i)\big)^{-1} \nabla F(x_i)
$$

and take

$$
x_{i+1} = x_i - \lambda\,\Delta_i, \qquad \lambda \in (0,1]
$$

where $\lambda$ is the **step size**. With $\lambda=1$ this is the method we already
have; with smaller $\lambda$ it is a **damped** Newton step.

## Choosing $\lambda$ by step halving

The step size is chosen by trial: take the full step, check whether it improved
matters, and if not, halve it.

1. Start with the full Newton step, $\lambda=1$
2. If the step improves on the current point, accept it and go on to the next iteration
3. If it does not, replace $\lambda$ by $\lambda/2$
4. Repeat steps 2–3 until an improving step is found

The step sizes tried are therefore $\lambda=1,\tfrac12,\tfrac14,\tfrac18,\ldots$, so
this is called a **step-halving line search**. It is computationally cheap: the
gradient and the Hessian are computed once per iteration and reused for every trial
$\lambda$ — only the cheap part, the function value, is recomputed.

We are maximizing, so what counts as an improvement is the **criterion function
itself**: a trial step is accepted when $F$ goes up. This turns the solver into a
genuine maximizer — it can no longer converge to a minimum — but it only works if
$-\Delta_i$ is an ascent direction, which requires the Hessian to be negative
definite. Far from the optimum it need not be, and then no $\lambda$ improves $F$
and the line search fails. Forcing the curvature matrix to be definite is exactly
what the **quasi-Newton methods** listed at the end of these notes achieve.

```{code-cell} python3
def newton2_ascent(fun,grad,x0,obj=F,tol=1e-6,maxiter=100,maxhalve=25,callback=None):
    '''Newton method with a step-halving line search on the criterion obj.
    A step is accepted only when obj increases, so the iterations always climb.
    '''
    npfun  = lambda x: np.asarray(fun(x[0],x[1]))
    npgrad = lambda x: np.asarray(grad(x[0],x[1]))
    x0 = np.asarray(x0,dtype=float)
    obj0 = obj(*x0)  # criterion at the current point
    for i in range(maxiter):
        step = np.linalg.solve(npgrad(x0),npfun(x0))  # the full Newton step
        lam = 1.0
        for j in range(maxhalve):  # step-halving line search
            x1 = x0 - lam*step
            obj1 = obj(*x1)
            if obj1 > obj0: break  # uphill, accept this lambda
            lam = lam/2
        else:
            raise RuntimeError('No ascent direction at iteration %d'%i)
        err = np.amax(np.abs(x1-x0))
        if callback != None: callback(iter=i,err=err,x0=x0,x1=x1,lam=lam)
        if err<tol: break
        x0, obj0 = x1, obj1
    else:
        raise RuntimeError('Failed to converge in %d iterations'%maxiter)
    return x1
```

## The same starting points, with a line search

Exactly the picture from above — the same ten starting points in the same small square,
the same colors — but solved with the line search component.

Also a wider choice of starting points, line search can handle a few further out.

```{code-cell} python3
:tags: [hide-input]

plot_newton_paths(newton2_ascent)

center, side = np.array([0.5,0.4]), 0.5  # the small square of starting points
starts = center + np.random.default_rng(5).uniform(-side/2,side/2,size=(25,2))
colors = plt.cm.autumn(np.linspace(0,0.85,len(starts)))

plot_newton_paths(solver=newton2_ascent,starts=starts,colors=colors)
```

A completely different picture! 

Nothing runs away: every path either climbs to a peak or stops, at a cross, where step halving could not find an uphill step. 

The line search never returns a minimum or a saddle point — it cannot, because every
accepted step increases $F$ and neither is approached from below.

The line search is also the only one of the two that reaches the **global** maximum $F=1.2084$ from this cluster of starting points, and it does so in fewer iterations, having halved the step on just
three of them.

The price is multiple failures. They are not failures to converge: the solver stops
because no $\lambda$ improved $F$, which happens whenever the Newton direction is not
an ascent direction.

That is a property of the direction, not of the step length, and
shortening the step cannot cure it. The $-\big(\nabla^2F\big)^{-1}\nabla F$ points uphill
only when the Hessian is negative definite, which far from a peak it need not be.

:::{div}
:class: discussion

- Plain Newton converges more often, but half of what it returns is not a maximum. Is a method that more often gives up (fails to find suitable step size) better or worse?

- Would step halving have rescued any of the failures of the one-dimensional Newton?

- The full step is tried first at every iteration, and is only halved when Newton is going off. What implications does this have for the convergence properties of the method?
:::

```{note} Line search in practice

Halving until the criterion improves is the simplest rule that works. Production codes
ask for a bit more — that the improvement be proportional to the step taken (the
*Armijo* condition) and that the step not be too short (the *Wolfe* conditions) — which
is what buys the formal global convergence theorems. `scipy.optimize` implements these,
and its `line_search` is available separately from the solvers that use it.

When the problem is to solve $G(x)=0$ rather than to maximize something, the same
search is run on the residual $\|G(x)\|$ instead of on $F$. That version converges from
more starting points, but it has no reason to prefer a maximum over a minimum or a saddle point in this case.

The other way to control the step is to bound its length in advance rather than shrink
it afterwards, and solve the quadratic model within that bound. Those are **trust
region** methods, the main alternative to line search.

```


# BFGS, BHHH and other quasi-Newton methods

The line search of the previous section can only work if the direction it is handed
points uphill, and $-\big(\nabla^2 F\big)^{-1}\nabla F$ does so only where the Hessian
is negative definite. **Quasi-Newton** methods remove that condition by never using the
true Hessian at all: they carry an approximation to it, built from the gradients already
computed, and update that approximation in a way that preserves its definiteness at
every iteration. The direction is then an improving one by construction, and the line
search always has a step to find.

BFGS (Broyden–Fletcher–Goldfarb–Shanno) builds up an approximation to the inverse Hessian iteratively, ensuring correct 
definiteness (negative for maximization and positive for minimization problems). 
Other approaches to approximating the Hessian, such as the DFP
(Davidon–Fletcher–Powell) method, were developed for general optimization problems.

BHHH (Berndt-Hall-Hall-Hausman) gets the same guarantee out of statistics rather than out of the update formula.

Maximizing a log-likelihood with score contributions $s_i = \nabla_\theta \ln L_i(\theta)$,
it uses the outer product $B = \sum_i s_i s_i'$ in place of $-\nabla^2 \ln L$. That matrix
is positive semi-definite by construction, and by the information identity it estimates
$-\mathbb{E}\big[\nabla^2 \ln L\big]$, so the step

$$
\theta^{(g+1)} = \theta^{(g)} + \lambda\, B^{-1} \sum_i s_i(\theta^{(g)})
$$

always climbs. BHHH, being based on the statistical properties of the scores, is more
specific to econometric M-estimation problems — and for that reason it is often *not*
found in standard optimization packages.


:::{note} Where this comes back

The NFXP estimator is a Newton-type optimizer with analytical derivatives
on the outside, and a Newton–Kantorovich solver for the Bellman equation on the
inside. Both loops are the algorithms of this class, so it pays to have them clear now.

The full treatment of M-estimation, asymptotics and the information identity is
reference reading in the course — see the source notes listed below.
:::

(task4.1)=
````{warning} Practical task: Newton fractals

Find the notebook `newton_fractals.ipynb` in the lecture notes code repository and 
study the code for the simpler examples. 

Adjust it to show the basins of attraction of the four local maxima and other critical points of the function $F(x,y)$ we've been using in this section.

````

(4_solvers_references)=
````{note} References and additional resources

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

- 📖 {cite:t}`berndtEstimationInferenceNonlinear1974` "Estimation and Inference in
  Nonlinear Structural Models" — the original BHHH paper
  [link](https://www.nber.org/chapters/c10206)
````
