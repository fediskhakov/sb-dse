---
title: "☕️ Beauty of coding: towers of Hanoi"
short_title: ☕️ Towers of Hanoi
subtitle: Class 2 — Thursday, August 27 (if time allows)
exports:
  - format: typst
    output: exports/2_recursion.pdf
downloads:
  - file: 2_recursion.md
    title: MyST Markdown
kernelspec:
  name: python3
  display_name: Python 3
---

# Towers of Hanoi puzzle

A classic puzzle: given a board with three pegs, move a stack of disks of different
size from the left-most peg to the right-most peg, moving one disk at a time and
following the rule that no larger disk can be placed on top of a smaller one.

```{image} _static/img/hanoi.jpg
:alt: Towers of Hanoi
:width: 60%
:align: center
```

````{margin}
```{image} _static/img/recursion.gif
:alt: Recursion
:width: 90%
```
````

````{attention} Definition

A recursive algorithm is an algorithm that calls itself in order to solve a problem.
````

A surprisingly powerful technique in scientific programming — and the structure of
every dynamic programming solver we write later in the course.

The Hanoi puzzle can be solved nicely by breaking it into small parts using the following
algorithm:

```
def move(from,to):
  move one disk from --> to

def main_algorithm(n,source,aux,target):
  '''
  Inputs: number of disks n
        source peg
        auxiliary peg
        target peg
  '''
  if n==0:
    do nothing, return
  if n==1:
    move(source,target)
  if n>0:
    main_algorithm(n-1,source,target,aux)
    move(source,target)
    main_algorithm(n-1,aux,source,target)
```

(task_hanoi_)=
````{warning} Practical task: Towers of Hanoi

Code up the recursive solution using the algorithm above, and print the sequence of
moves. How many moves does the solution for $n$ disks take? What is the complexity
class of the algorithm (see next lecture)?

Navigate to the directory you chose to save the course materials (must be different from the homework repository!), and clone the code repo once:

```bash
git clone https://github.com/fediskhakov/sb-dse-code.git
cd sb-dse-code
```

The starter code for this problem is `session03-sep1/hanoi.py`. Open it in VS Code directly or with

```bash
jupyter lab session03-sep1/algorithms.ipynb
```

The repository is read-only for you. Nothing in it is submitted, so experiment
freely — but once you have edited a file in place `git pull` refuses to update it.
So either copy anything you want to keep out of this repo, or commit to a separate branch, and remove all your changes by running

```bash
git reset --hard HEAD
```
Setting up the Python environment is covered in
[](2_workflow.md#python-install).

````

`````{tip} Solution
:class: dropdown

The whole puzzle collapses into three lines: to move $n$ disks, first move the top
$n-1$ out of the way onto the auxiliary peg, move the bottom disk across, then move
those $n-1$ back on top of it. Each of those two sub-problems is the same problem
with one disk fewer, so the function calls itself.

```{code-cell} python3
def hanoi(n, source='A', aux='B', target='C', verbose=True):
  '''Move n disks from source peg to target peg, using aux as intermediary.
  Returns the number of moves made.
  '''
  if n <= 0:
    return 0                                          # nothing left to move
  moves = hanoi(n-1, source, target, aux, verbose)     # free the bottom disk
  if verbose:
    print(f'{source} --> {target}')                    # move it across
  moves += hanoi(n-1, aux, source, target, verbose)    # rebuild on top of it
  return moves + 1

print(f'{hanoi(3)} moves for 3 disks')
```

Note that the recursive calls swap the roles of the pegs: the auxiliary peg of one
call is the target of the next. Only the base case `n <= 0` stops the recursion.

`````

The solution for $n$ disks requires $2^n-1$ moves, so 15 for the four disks below —
the illustration stops at configuration 13, two moves short of the goal.

```{image} _static/img/hanoi_solution.jpg
:alt: Towers of Hanoi solution
:width: 40%
:align: center
```

(2_recursion_references)=
````{note} References and additional resources

- 📺 Same problem in greater details by prof. Thorsten Altenkirch, University of Nottingham
  [video, 12 min](https://www.youtube.com/watch?v=8lhxIOAfDss)

````
