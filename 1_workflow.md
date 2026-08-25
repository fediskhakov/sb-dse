---
title: 💻 Work environment and submission workflow
short_title: 💻 Setup and workflow
subtitle: Class 2 — Thursday, August 27
authors:
  - name: Fedor Iskhakov
    url: https://fedor.iskh.me
    affiliations:
      - Stony Brook University
  - name: Claude Code
    url: https://claude.com/claude-code
---

```{admonition} Disclosure
:class: note

This page was written by Claude Code along the draft I wrote myself.
I have checked and edited all of the text written by the assistant.

It is here as a worked example of the [disclosure policy](#disclosure-policy) below:
this is what the required note looks like when the assistant did more than editing.

```

In addition to theory 📖 this course has a strong ⚙️ practical component, with
hands-on exercises and a semester-long project. To follow the course effectively you
need an environment in which you can *run* code, *version* code, and — this is the
new part — *collaborate with an AI assistant* on code without losing control of what
it does.

This class sets up all three, plus the workflow you will use to submit every
homework: a pull request in your own course repository.

```{admonition} Three layers
:class: hint

1. **Run** — a Python interpreter and the scientific stack. Fails when your machine
   and mine resolve different package versions.
2. **Version** — Git and GitHub. Fails when work exists only on one laptop, or when
   a notebook merge conflict eats an afternoon.
3. **Assist** — an AI coding tool. Fails silently: the code runs, the plot looks
   plausible, the estimates are wrong.

Most of the time you lose to debugging in a structural project is spent due to failing with 2.
All are essential for your productivity, so we'll build good practices for all the tooling in this class.

```

```{admonition} Local install, or the cloud
:class: hint

This page describes a *local install*, which lets you run course code on your own
machine and is what I recommend. Cloud environments — [Google
Colab](https://colab.research.google.com), [GitHub
Codespaces](https://github.com/features/codespaces) — come preconfigured and are a
perfectly good fallback if your laptop fights you; see [](#cloud) below. Estimation
runs in the second half of the course are long enough that you will want a machine
you control.
```

## Prerequisites

- A reasonably modern computer: desktop or laptop, Windows or macOS or Linux
- Administrative access to it (you can install programs)
- A terminal you are willing to type into — PowerShell or Windows Terminal on
  Windows, Terminal or iTerm on macOS, anything on Linux

```{admonition} Software components
:class: note

1. A **Python** environment, installed with **uv** (or conda, see below)
2. **Git**, and optionally a graphical client for it
3. A **GitHub** account with two-factor authentication and an SSH key
4. A **code editor** — VS Code or one of its AI-enabled relatives
5. At least one **AI coding assistant**, configured and understood
```

(python-install)=
## Python

The course primarily uses `numpy`, `scipy`, `matplotlib` and `sympy`, and Jupyter to run the
notebooks. Nothing fancy is required; no deep learning frameworks, no heavy toolboxes.

### The current default: uv

[uv](https://docs.astral.sh/uv/) is a Python package and project manager that has
become the default way to set up scientific Python. It installs Python itself, so
you do not need a separate Python install, and it resolves and installs the whole
stack in seconds rather than minutes.

```bash
# install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
# on Windows, in PowerShell:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# then, inside the course repository
uv venv                      # create .venv with a suitable Python
uv pip install -r requirements.txt
uv run jupyter lab           # run without activating anything
```

For your own project, prefer a `pyproject.toml` and let uv maintain a lock file:

```bash
uv init                      # writes pyproject.toml
uv add numpy scipy matplotlib sympy jupyter
uv sync                      # reproduces the environment exactly from uv.lock
```

`uv.lock` records the exact version of every package, transitively. Commit it. It is
the difference between "works on my machine" and a replication package someone can
actually run.

### The alternative: conda

If you already work in conda, or you need compiled libraries that conda packages
better, use [Miniforge](https://conda-forge.org/download/), which is the conda-forge
installer:

```bash
conda env create -f environment.yml
conda activate eco629
```

```{admonition} Why Miniforge rather than the Anaconda Distribution
:class: note

Anaconda changed its terms of service in 2024: use of the Anaconda-hosted
`defaults` channel by larger organizations — a category that has been read to
include many universities outside of classroom instruction — requires a paid
licence. Several universities responded by moving off `defaults` entirely.
Miniforge installs the same `conda` tooling but points at the community-run
`conda-forge` channel, which carries no such condition.
```

[pixi](https://pixi.sh) is a newer conda-compatible manager with uv-style lock files,
worth knowing about if you end up in a conda-heavy group.

### Versions matter more than you think

- **Python 3.13 and 3.14** introduced free-threaded builds — an interpreter without
  the global interpreter lock. It is officially supported as of 3.14, and it changes
  how you would parallelize a solver.
- **NumPy 2.x** is the default and removed a number of long-deprecated aliases. 
  Code written before 2024 may fail on the removal, not on anything conceptual.
- **SciPy** keeps moving optimizers around; pin it in your project.

It is a good practice to record the versions you use in README. When your work cannot be
reproduced two years from now, this is almost always why, and it is a problem.

:::{div}
:class: discussion

Your solver converges on your laptop but does not on a classmate's, same code, same
data, same seed. Even worse if it is you and the data editor's team at the journal!
What are the plausible causes?
:::

## Where you write code

A good editor is *invaluable* for editing source files — and it is now also the place
where AI assistance lives, which makes the choice less cosmetic than it was.

- [**VS Code**](https://code.visualstudio.com/) — the default recommendation. Free,
  cross-platform, first-class Python and Jupyter support, Git built in, and GitHub
  Copilot integrates directly. If you have no strong preference, install this.
- [**Cursor**](https://cursor.com) and [**Windsurf**](https://windsurf.com) — forks
  of VS Code built around AI editing. Same keybindings and extensions, deeper
  assistant integration, subscription-based.
- [**Positron**](https://positron.posit.co) — Posit's data-science IDE, VS Code-based,
  with a data explorer and a session-oriented layout familiar from RStudio.
- [**Zed**](https://zed.dev) — fast native editor with built-in AI, if VS Code feels
  heavy.
- [**PyCharm**](https://www.jetbrains.com/pycharm/) — a full IDE; excellent debugger
  and refactoring, free for students.
- [**JupyterLab**](https://jupyter.org) / Notebook 7 — where you will actually run
  the course notebooks.

## Git and GitHub

Git is the version control system; GitHub hosts repositories and is where course
materials and all homework are distributed and collected.

- [Git](https://git-scm.com/) — the command line tool
- [Git GUI clients](https://git-scm.com/downloads/guis) —
  [GitHub Desktop](https://desktop.github.com/), [Sourcetree](https://www.sourcetreeapp.com/),
  [Fork](https://git-fork.com), or the Git panel built into VS Code
- [GitHub CLI](https://cli.github.com) (`gh`) — creates pull requests, checks out
  someone else's PR, views CI logs, all from the terminal. Install it; it removes
  most of the browser round-trips from the submission workflow.

### Account setup, once

1. Register on [GitHub](https://github.com/join). *Be mindful about the personal data
   you use when registering.*
2. Enable **two-factor authentication** — required for accounts that contribute code.
   A passkey or an authenticator app is fine; SMS is the weakest option.
3. Add an **SSH key** (or use `gh auth login`, which sets one up for you).
   Password authentication for Git operations has not worked since 2021; if a
   tutorial tells you to type your GitHub password into a terminal, it predates the
   current world.
4. Apply for the [Student Developer Pack](https://education.github.com/pack) — it
   includes GitHub Pro, Copilot Pro, extra Codespaces hours and a long list of other
   services, free while you are a student.
5. Send me your GitHub username. I will create a private repository for you and
   invite you to it, together with the repository the tasks are posted in — see
   [](#submission) below. **Accept both invitations**; nothing works until you do.

### Commands worth knowing

```bash
git pull upstream main         # collect newly posted tasks
git switch -c hw1              # create and move to a branch (modern `checkout -b`)
git status                     # what is changed, staged, untracked
git add -p                     # stage selected hunks — read your own diff
git commit -m "HW1: bisection solver"
git push -u origin hw1
gh pr create --fill            # open the pull request without leaving the terminal
```

```{admonition} Things not to commit
:class: attention

- **Secrets** — API keys for AI tools, tokens, passwords. GitHub scans public pushes
  for known key formats and will alert, but assume anything pushed is public forever.
  Keep keys in environment variables or a `.env` file that is listed in `.gitignore`.
- **Large or restricted data.** Anything above a few tens of megabytes belongs in
  [Git LFS](https://git-lfs.com) or outside the repository, with a download script
  committed instead. Licensed data is never committed.
- **Build output and environments** — `_build/`, `.venv/`, `__pycache__/`,
  `.ipynb_checkpoints/`. Start every repository with a Python `.gitignore`.
```

## Notebooks, and their limits

Editing course notebooks and writing project modules are different activities. Use
the notebook for exploration and the editor for anything you will reuse.

Jupyter notebooks are an excellent way to present and discuss code — this entire
course is taught with them — and a good instrument for developing ideas. They are
saved as JSON with text, LaTeX math, code and code output interleaved.

Their limitations have not changed, and they matter for your project:

- **Not a good place to store developed code.** Put solvers, simulators and
  estimators in `.py` modules and import them into the notebook.
- **Not good for version control.** Execution counts, metadata and cell outputs are
  all tracked as changes; merging two edited notebooks can break the JSON outright.
- **Hidden state.** Cells run out of order produce results that no fresh run
  reproduces. Before you submit anything: *Restart kernel and run all*.

However, the tooling around these problems is now good enough that there
is no excuse for fighting them:

- [**jupytext**](https://jupytext.readthedocs.io) pairs every notebook with a plain
  `.py` or `.md` file. You commit the text file, diffs are readable, merges work.
  The percent format (`# %%` cell markers) is understood by VS Code, PyCharm and
  Spyder, which run those `.py` files as notebooks directly.
- **nbstripout** and [**nbdime**](https://nbdime.readthedocs.io) strip outputs
  before commit and diff notebooks cell by cell. Install nbstripout as a
  [pre-commit](https://pre-commit.com) hook and forget about it.
- [**marimo**](https://marimo.io) is a reactive Python notebook stored as an ordinary
  `.py` file: no hidden state, since changing a cell re-runs everything downstream,
  and it is git-friendly by construction. A serious option for your project.
- **MyST Markdown** notebooks are how these lecture notes are written — Markdown with
  executable code blocks, built by [Jupyter Book 2](https://jupyterbook.org). Look at
  the source of any chapter of this book to see the format.

````{admonition} Minimum viable hygiene
:class: hint

In your project repository, before your first commit:

```bash
pip install nbstripout && nbstripout --install
```

That one command removes the single most common source of unreadable diffs and
unmergeable conflicts in code-heavy coursework.
````

(ai)=
## AI coding assistants

This course is AI-friendly: see the [AI policy](#ai-policy) on the front
page. The policy is short — *you are responsible for everything you submit, and you
must understand every line* — and it is examined orally.
You also must disclose the use of AI assistants beyond text editing: see the [disclosure policy](#disclosure-policy) below.

### Three kinds of tool

| Kind | Examples | Good for |
| :-- | :-- | :-- |
| **Chat** | [Claude](https://claude.ai), [ChatGPT](https://chatgpt.com), [Gemini](https://gemini.google.com) | deriving, explaining, checking algebra, talking through a modelling choice |
| **In-editor** | [GitHub Copilot](https://github.com/features/copilot), Cursor, Windsurf, Colab's assistant | completion, small refactors, docstrings, tests, "explain this cell" |
| **Agentic CLI** | [Claude Code](https://claude.com/claude-code), [Codex CLI](https://developers.openai.com/codex/cli/), [Gemini CLI](https://github.com/google-gemini/gemini-cli) | multi-file work: writing a solver module, running it, reading the traceback, fixing it, repeating |

The third category is what changed most recently, and it is the one worth your time
this semester. An agentic tool runs in your repository, reads and writes files, and
executes commands — so it can close the loop between writing code and finding out
that the code is wrong, which is exactly the loop a structural project consists of.

Copilot has a free tier, and Copilot Pro is included in the Student Developer Pack.
Colab includes assistance at no cost. Anthropic and OpenAI both offer inexpensive
entry plans that include their CLIs. You do not need to pay for all of these; pick
one from the second or third row and learn it properly.

### Configure the context, not just the prompt

Every one of these tools reads a plain-text instructions file from the repository
root — `CLAUDE.md`, `AGENTS.md`, or the editor's equivalent — and treats it as
standing instructions. This is where you write down what the assistant cannot infer:

```markdown
# Project: Rust bus engine replacement, ECO 629

- Python, numpy/scipy only. No deep learning frameworks.
- Value functions are indexed by [state]; the two replacement-specific value is the first element of the vector; keep that order everywhere.
- Always check Bellman convergence with a monotonicity test before trusting output.
- Never edit files under data/raw/.
- Leave all git operations to me.
```

Five lines like these are worth more than any amount of prompt phrasing. Commit the
file: it documents the project's conventions for humans too.

### Where AI helps

- **Scaffolding** — grids, interpolation helpers, plotting, argument parsing, tests.
  Tedious, well-specified, easy to check.
- **Translation** — a lot of code is historically written in MATLAB and Fortran. Porting it
  to Python or Julia is a task assistants do well, and you should verify numerically against
  the original output.
- **Debugging** — pasting a traceback plus the relevant function is the single
  highest-value use.
- **Reading** — "what does equation (12) in this paper assume about the error term"
  against a PDF you have open.
- **Derivations** — with `sympy` in the loop so the algebra is checked, not asserted.

### Where it will quietly hurt you

Numerical code fails silently. A wrong Jacobian, a mis-signed log-likelihood, a
transition matrix whose rows do not sum to one — all of these produce output, plots,
and estimates. Nothing raises an exception. Assistants are fluent at producing
plausible numerical code, which means the burden of verification is entirely yours.

```{admonition} Verification habits that catch AI errors
:class: attention

- **Analytic special cases.** Solve the model where you know the answer — one period,
  zero discounting, degenerate uncertainty — and compare.
- **Check the derivative numerically.** Compare any analytic gradient against a
  finite-difference approximation before it goes into an optimizer.
- **Check the probabilities.** Transition matrix rows sum to one; choice
  probabilities sum to one; likelihood contributions are negative.
- **Vary the grid.** Results that move with the grid size are not converged.
- **Recover known parameters.** Simulate data from your own model at known
  parameters, estimate, and confirm you get them back. This single test catches most
  estimator bugs, AI-authored or not.
- **Read the diff.** Accepting a multi-file change unread is how a project acquires
  a bug it cannot locate three weeks later.
```

(disclosure-policy)=
### Disclosure

```{admonition} Disclosure policy
:class: attention

**Any use of an AI assistant that goes beyond text editing must be disclosed** in the
pull request that submits the work.

Beyond text editing means: the assistant wrote or rewrote code, derived or checked
mathematics, produced prose that survives into your submission, or ran and debugged
your code. Completion of a variable name, reformatting, and spell checking do not
need to be reported.

State, in a few lines, *which tool* and *for what part of the work*.
```

This is not an obstacle to clear — it is normal professional practice, and
increasingly journal policy. A complete disclosure is short:

> Claude Code wrote the EGM interpolation helper and the plotting code; I verified
> the helper against the analytic solution for the cake-eating case. ChatGPT was used
> to check the algebra of the Euler equation derivation in Section 2.

Disclosure changes nothing about responsibility. The work is yours, the errors are
yours, and the oral exam is about your understanding of every line you submit —
[see the AI policy](#ai-policy). Disclosure exists so that we can talk about *how* you
worked, which is a subject of this course in its own right.

:::{div}
:class: discussion

An assistant hands you a working NFXP estimator in five minutes. What do you have,
and what do you not have? What would you need to do before you would defend it at
the board?
:::

## Reproducibility, from day one

Your project is a small replication package. Structural estimates that nobody can
reproduce — including you, in six months — are not results.

- **Pin the environment**: commit `uv.lock` or `environment.yml`, and note the Python
  version.
- **Seed everything**: create one `np.random.default_rng(seed)` and pass it down.
  Never call the legacy global `np.random.*` functions in project code.
- **Separate stages**: a script that solves, a script that simulates, a script that
  estimates, each writing its output to disk. Estimation runs are long; you do not
  want to redo the solve because a plot was ugly.
- **Log the runs**: parameters in, starting values, convergence flag, wall time,
  final objective. A CSV of runs beats scrollback.
- **Write the README as you go**: how to install, what to run, in what order, and
  roughly how long each step takes.

The [AEA Data Editor's replication policy](https://aeadataeditor.github.io/) is the
standard your future submissions will be held to, and it is a good template now.

(cloud)=
## Cloud fallback

- [**Google Colab**](https://colab.research.google.com) — free tier, no setup,
  built-in AI assistance, easy to share. Sessions are time-limited and the filesystem
  is ephemeral; mount Drive or push to GitHub before you close the tab.
- [**GitHub Codespaces**](https://github.com/features/codespaces) — a full VS Code
  in the browser running on a container defined by a `.devcontainer/` file in the
  repository. Monthly free allowance, larger with the Student Developer Pack. The
  closest cloud equivalent of a local install, and the environment is defined in
  code.
- [**Binder**](https://mybinder.org) — builds a temporary environment from any public
  repository. Capacity-limited; good for demos, not for work.
- **Stony Brook's [SeaWulf](https://it.stonybrook.edu/services/high-performance-computing)
  cluster** — if your project needs to run many estimations, or a bootstrap, or a
  Monte Carlo, this is what it is for. Talk to me early if you want to go this route.

(submission)=
## Homework submission workflow

You do not fork anything, and you never push to the course repository. Two
repositories are involved:

| Repository | What it is | Your access |
| :-- | :-- | :-- |
| `fediskhakov/sb_dse_class` | the **task** repository — I push each assignment here | read |
| `fediskhakov/sb_dse_<your-username>` | **your** repository for the whole semester | write |

Your repository is created as a clone of the task repository, so it shares its
history: pulling from the task repository brings each new assignment straight into
your working copy, all semester, without any copying by hand.

### Once, at the start of the semester

1. Send me your GitHub username
2. Accept the two invitations GitHub emails you — one to your own repository, one to
   the task repository
3. Clone your repository and add the task repository as a second remote:

```bash
git clone git@github.com:fediskhakov/sb_dse_<your-username>.git
cd sb_dse_<your-username>
git remote add upstream git@github.com:fediskhakov/sb_dse_class.git
pip install nbstripout && nbstripout --install
```

`origin` is now your repository, `upstream` is the task repository. Confirm with
`git remote -v`.

### For each assignment

```bash
git pull upstream main                    # the new task appears in tasks/hwN/
git switch -c hw1                         # one branch per assignment
#  ... work in solutions/hw1/ ...
git add . && git commit -m "HW1: inventory model solver"
git push -u origin hw1
gh pr create --fill --reviewer fediskhakov
```

### While you work

The line `git add . && git commit -m "..."` above compresses a whole assignment into
one commit. Do not actually work that way. Commit every time something starts
working — a function that returns the right value, a plot that comes out, a bug you
have just fixed:

```bash
git status                      # what have I changed?
git diff                        # what exactly did I change?
git add solutions/hw1/vfi.py    # stage one file...
git add -p                      # ...or stage selected hunks, reading as you go
git commit -m "VFI converges on the deterministic case"
git push                        # after the first push -u, plain git push is enough
```

Ten small commits over three evenings, then the pull request:

- **A commit is a save point.** When an experiment goes wrong, `git restore .` puts
  you back at the last one — cheaper and safer than undoing edits by hand.
- **Pushing is your backup.** A laptop that dies the night before the deadline costs
  you nothing if the branch is already on GitHub.
- **The history shows how you worked.** Useful to you at the board, when you have to
  explain why the estimator changed shape halfway through; useful to me, because a
  pull request that arrives as one commit of 400 lines cannot be reviewed.
- **Small commits get better comments.** Twelve commits with meaningful messages
  attract useful review; one commit called "hw1" attracts none.

Write messages that say what changed and why — "fix bug" is worth nothing three
weeks later, "clip the value function at the borrowing constraint" is worth a lot.

Three commands that recover from the usual mistakes:

```bash
git commit --amend    # fix the message or add a file to the last commit,
                      # but only if you have not pushed it yet
git restore <file>    # discard uncommitted changes to one file
git stash             # park changes to try something else; git stash pop brings them back
```

```{admonition} Keep a cheat sheet open
:class: hint

No need to memorizes all Git commands. Everyone looks them up.

- [GitHub's Git cheat sheet](https://education.github.com/git-cheat-sheet-education.pdf)
  — one page, and every command used in this course is on it
- [Atlassian's Git cheat sheet and tutorials](https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet)
  — the same commands with worked explanations of branching and merging
- [Dangit, Git!?!](https://dangitgit.com) — recipes for undoing whatever you just
  did to your repository, by symptom rather than by command name
```

### Submitting

The pull request is the submission, and its creation time is the timestamp of
record. It stays open until we have discussed the assignment in class; then it is
merged into your `main`, so your `main` ends the semester as a record of your
accepted work.

```{admonition} The one rule
:class: attention

**Never edit anything under `tasks/`.** Your work goes in `solutions/hwN/`.

Because your repository shares history with the task repository, an untouched
`tasks/` folder makes every `git pull upstream main` a clean fast-forward. Edit a
task file and you will be resolving merge conflicts on every future assignment
instead of doing the assignment. If you want to modify code I handed out, copy it
into your solutions folder first and modify the copy.
```

Homework is discussed at the start of the class that follows it, with one student
presenting the solution at the board. The presenter rotates, so plan on presenting
several times over the semester — including code that does not work yet, which is
usually the more instructive case.

```{admonition} Pull request checklist
:class: hint

- Runs top to bottom from a fresh kernel
- Nothing under `tasks/` is touched
- No outputs, checkpoints, `.venv/` or data dumps in the diff
- A sentence on what the code does, and the AI disclosure if any assistant did more
  than text editing
```

````{admonition} Practical task
:class: warning

1. Install Python with uv (or Miniforge) and confirm `import numpy, scipy, sympy`
   works
2. Enable two-factor authentication on GitHub and add an SSH key
3. Send me your GitHub username, accept both invitations, and clone your repository
4. Add `upstream` and install `nbstripout`, as above
5. Create a branch, add a file under `solutions/`, and edit it in your editor
6. Stage and commit the change, and read the diff before you push
7. Push the branch and open a pull request with me as reviewer
8. Install one AI coding assistant, and ask it to explain one function from the
   course code back to you. Judge the explanation — it is the assistant that is being
   examined here, not the code

The first pull request of the semester is the student survey — no code required.
````

````{admonition} References and additional resources
:class: note

- QuantEcon on setting up a local environment
  [link](https://python-programming.quantecon.org/getting_started.html)

- Workspace setup lecture from the *Foundations of Computational Economics* course
  [YouTube video](https://youtu.be/UrZnRv3_IUc)

- uv documentation [link](https://docs.astral.sh/uv/)

- Simple guide to Git [link](https://rogerdudler.github.io/git-guide/)

- Full Git reference [link](https://git-scm.com/doc)

- Git cheat sheet, one page [pdf](https://education.github.com/git-cheat-sheet-education.pdf)

- Undoing Git mistakes by symptom [link](https://dangitgit.com)

- GitHub intro [30 min online course](https://education.github.com/experiences/intro_to_github)

- Understanding Markdown [20 min online course](https://education.github.com/experiences/understanding_markdown)

- Jupytext documentation [link](https://jupytext.readthedocs.io)

- marimo, a reactive git-friendly notebook [link](https://marimo.io)

- AEA Data Editor, replication package requirements [link](https://aeadataeditor.github.io/)
````
