---
title: "🔬 Beauty of coding: towers of Hanoi"
short_title: 🔬 Towers of Hanoi
subtitle: Class 2 — Thursday, August 27 (if time allows)
exports:
  - format: typst
    output: exports/4_recursion.pdf
downloads:
  - file: 4_recursion.md
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

```{code-cell} python3
:tags: [hide-input]

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from IPython.display import HTML


def initial_position(n, source='A', target='C', auxiliary='B'):
    """Return the initial contents of the three pegs."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError('n must be a positive integer')
    if len({source, target, auxiliary}) != 3:
        raise ValueError('the three peg names must be different')

    return {
        source: list(range(n, 0, -1)),
        auxiliary: [],
        target: [],
    }


def hanoi_moves(n, source='A', target='C', auxiliary='B'):
    """Generate the moves in the same order as hanoi()."""
    if n > 0:
        yield from hanoi_moves(n - 1, source, auxiliary, target)
        yield source, target
        yield from hanoi_moves(n - 1, auxiliary, target, source)


def hanoi_states(n, source='A', target='C', auxiliary='B'):
    """Generate the initial position and every position after a move."""
    pegs = initial_position(n, source, target, auxiliary)
    yield 0, None, {name: disks.copy() for name, disks in pegs.items()}

    for step, (old_peg, new_peg) in enumerate(
        hanoi_moves(n, source, target, auxiliary), start=1
    ):
        disk = pegs[old_peg].pop()
        if pegs[new_peg] and pegs[new_peg][-1] < disk:
            raise RuntimeError('illegal move: a larger disk cannot cover a smaller disk')
        pegs[new_peg].append(disk)
        position = {name: disks.copy() for name, disks in pegs.items()}
        yield step, (disk, old_peg, new_peg), position

def draw_hanoi(position, title=None, ax=None):
    """Draw one position of the puzzle and return its axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    else:
        ax.clear()

    peg_names = list(position)
    n = sum(len(disks) for disks in position.values())
    disk_height = 0.65
    base_y = 0.25

    ax.plot([-0.55, 2.55], [base_y, base_y], color='0.25', linewidth=3)
    for x, (name, disks) in enumerate(position.items()):
        ax.plot(
            [x, x], [base_y, base_y + disk_height * (n + 0.6)],
            color='0.35', linewidth=3, zorder=0,
        )
        for level, disk in enumerate(disks):
            width = 0.35 + 0.95 * disk / n
            rectangle = Rectangle(
                (x - width / 2, base_y + level * disk_height),
                width, disk_height * 0.82,
                facecolor=plt.cm.viridis(0.15 + 0.7 * disk / n),
                edgecolor='0.15',
                linewidth=1.2,
                zorder=1,
            )
            ax.add_patch(rectangle)
            ax.text(
                x, base_y + (level + 0.41) * disk_height, str(disk),
                ha='center', va='center', color='white', fontweight='bold',
                zorder=2,
            )

    ax.set_xticks(range(3), peg_names)
    ax.set_xlim(-0.7, 2.7)
    ax.set_ylim(0, base_y + disk_height * (n + 1))
    ax.set_yticks([])
    ax.tick_params(axis='x', length=0, labelsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title)
    return ax


def animate_hanoi(n, source='A', target='C', auxiliary='B', interval=700):
    """Return a notebook animation, beginning with the initial position."""
    states = list(hanoi_states(n, source, target, auxiliary))
    figure, ax = plt.subplots(figsize=(8, 4))

    def update(frame):
        step, move_data, position = states[frame]
        if move_data is None:
            title = 'Initial position'
        else:
            disk, old_peg, new_peg = move_data
            title = f'Move {step} of {len(states) - 1}: disk {disk}, {old_peg} → {new_peg}'
        draw_hanoi(position, title=title, ax=ax)

    animation = FuncAnimation(
        figure, update, frames=len(states), interval=interval, repeat=False
    )
    plt.close(figure)
    return HTML(animation.to_jshtml())

animate_hanoi(8)
```    

(4_recursion_references)=
````{note} References and additional resources

- 📺 Same problem in greater details by prof. Thorsten Altenkirch, University of Nottingham
  [video, 12 min](https://www.youtube.com/watch?v=8lhxIOAfDss)

- To sharpen your Python skills, see numerous resources online, talk to the AI agent, and maybe check my online course [Computational Economics](https://fedor.iskh.me/compecon)
````
