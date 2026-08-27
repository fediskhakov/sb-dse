---
title: 🏛 Dynamic Structural Models
short_title: 🏛 Home
subtitle: ECO 629 — Studies in Quantitative Methods, Fall 2026
site:
  hide_outline: false
---

**Graduate course at Stony Brook University, Fall 2026**

How to solve dynamic structural models numerically, and how to estimate them from
data. Single-agent discrete and continuous choice, micro-founded equilibrium models,
and dynamic games — each with the estimation toolkit that goes with it: NFXP, MPEC,
CCP, NPL and EPL.

[Course entry in the SBU catalog](https://catalog.stonybrook.edu/preview_course_nopop.php?catoid=12&coid=27734)

## 🧑‍🏫 Instructor

:::{div}
:class: instructor-card

```{image} _static/img/iskhakov2.jpg
:height: 200 px
:class: faceimg
```

- **Fedor Iskhakov**
- Professor of Economics, *Department of Economics*
- Email: `fediskhakov@gmail.com`
- Web: [Personal page](https://fedor.iskh.me)
- PhD from University of Oslo, 2009

:::

I’m an *applied microeconometrician* and a *computational economist* working in the field of **structural estimation of dynamic models** of individual and strategic
choice, with applications to `labor economics`, `public economics`, `durable goods markets`, `household finance`, `industrial organization` and `dynamic games`.

:::{div}
:class: discussion
How many days have I spent on Long Island before coming to this lecture?
:::

## 👥 Class introductions

:::{div}
:class: discussion
- What is your background?
- What are your research interests?
- What is your coding experience? Which languages?
- What is your prior experience with dynamic programming and Bellman equations?
- What is your experience in empirical applications and data analysis?
:::

(schedule)=
## 📆 Course outline

Seven parts over 15 weeks, including two of project presentations.
Each class may combine theory with live code.
Homework is discussed at the start of the class that follows it on rotating basis.

The plan below is provisional and will be adjusted as we go.

| Date | Topic |
| :-- | :-- |
| | **Part I — Foundations and computational toolkit** |
| Tue Aug 25 | [Introduction to structural estimation](1_intro.md) — what a structural project consists of |
| Thu Aug 27 | [Work environment and submission workflow](1_workflow.md) Practical tasks on version control |
| Tue Sep 1 | [Algorithms and complexity](2_algo.md) |
| Thu Sep 3 | [Root finding and optimization](2_solvers.md) — bisection, Newton–Raphson, BHHH |
| | **Part II — Dynamic programming** |
| Tue Sep 8 | Bellman equations and backwards induction in finite horizon |
| Thu Sep 10 | Contraction mappings, infinite horizon, value function and policy iteration |
| | **Part III — Single-agent dynamic discrete choice** |
| Tue Sep 15 | The Rust bus engine replacement model |
| Thu Sep 17 | Newton–Kantorovich iterations and the poly-algorithm |
| Tue Sep 22 | Nested fixed point estimation (NFXP) |
| Thu Sep 24 | Mathematical programming with equilibrium constraints (MPEC) |
| Tue Sep 29 | Conditional choice probabilities and identification |
| Thu Oct 1 | Two-step CCP estimation |
| Tue Oct 6 | Nested pseudo-likelihood (NPL) |
| Thu Oct 8 | Unobserved heterogeneity and the EM algorithm |
| Tue Oct 13 | *Fall break — no class* |
| | **Part IV — Continuous choice and simulation-based estimation** |
| Thu Oct 15 | Cake eating on a grid; function approximation |
| Tue Oct 20 | The same model solved many ways — accuracy and speed compared |
| Thu Oct 22 | The endogenous gridpoint method and consumption-savings models |
| Tue Oct 27 | DC-EGM for discrete-continuous choice |
| Thu Oct 29 | Method of simulated moments |
| | **Part V — Equilibrium models** |
| Tue Nov 3 | Micro-founded equilibrium models — equilibrium trade in used cars &nbsp;·&nbsp; **project proposal due** |
| Thu Nov 5 | Doubly nested fixed point estimation &nbsp;·&nbsp; **project clinic** |
| | **Part VI — Games** |
| Tue Nov 10 | Static games of incomplete information and multiplicity of equilibria |
| Thu Nov 12 | Dynamic entry games and their estimation |
| Tue Nov 17 | Directional dynamic games — finding *all* equilibria |
| Thu Nov 19 | Estimation of directional dynamic games |
| | **Part VII — Guest lecture and project presentations** |
| Tue Nov 24 | Guest lecture — speaker and topic to be announced |
| Thu Nov 26 | *Thanksgiving — no class* |
| Tue Dec 1 | Project presentations I |
| Thu Dec 3 | Project presentations II; course wrap-up |

:::{div}
:class: discussion
We have a quite unique opportunity for course co-creation.
- Which topics would you add to the list?
- Which topics would you remove from the list?
:::


## 🏡 When and where

```{list-table}
:header-rows: 0
:widths: 25 75

* - **Time**
  - Tuesday and Thursday, 09:30 – 10:50
* - **Room**
  - SOCBEHAV SCI N601
* - **Office hours**
  - Tuesdays after the lecture, any time by appointment

```

:::{div}
:class: discussion
In other courses, what happens when you miss a lecture?
:::


## 🎓 Assessment

| Weight | Component |
| :-- | :-- |
| **70%** | Individual project, presentation and oral exam |
| **30%** | Homework and in-class participation |

### Project

An individual structural estimation project, carried out in the second half of the semester, with a written report and code submitted, and
presented in class in the last week. Ideally it includes all the steps of a
structural estimation project:

- theoretical model description
- solver code
- simulator code
- estimator code
- application of the model to real or simulated data
- counterfactual simulations of some sort

The model may be one of your own interest, or an extension/modification of one of the models covered in class — the bus engine replacement model, the inventory management model, the consumption-savings model, or one of the entry games.

**Project proposal** is due **Tuesday, November 3**, and should give a brief (max 2 pages) description of the project you intend to carry out together with a roadmap of the
steps you intend to take. All proposals are discussed in the project clinic on
**Thursday, November 5**.

**Presentations** take place on **December 1 and 3**, approximately 30 minutes each,
with room for a live demo of the solver and estimator code, not just slides.

**Oral exam** — paper and documented code should be submitted before the exam week — and will be a *conversation starter* in the oral exam during the exam period.

### Homework

Homework is typically given on Tuesday and is due the following Thursday.

- Each homework assignment will be discussed at the start of the class that follows it
- One student (volunteer or chosen at random) is to present the solution at the board or using slides
- With rotation everyone presents several times over the semester
- This is part of the grade

Homework is submitted as a pull request in your own private course repository, which I create for you and have access to; the workflow is set up in the first week, see [](1_workflow.md#submission).

## 💻 Software

The course is intended in Python, yet you can discover equivalent code online in Matlab, Julia and other languages.
Particular language is less important these days because we will be working with small examples.

By next time sets up the work environment, see [](1_workflow.md).

Computers are expected in the lectures, you are encouraged to run the code examples in the lecture, and interact with the code and the slides in class.

The lecture notes are published at [dse.iskh.me](https://dse.iskh.me)
and can be downloaded as a single PDF: [eco629-dynamic-structural-models.pdf](/eco629-dynamic-structural-models.pdf).
Each chapter also carries a download menu with that chapter as PDF or MyST Markdown.

Tasks and exercise code are posted in the private repository `fediskhakov/sb-dse-class`, and you work in a private repository of your own, `fediskhakov/sb-dse-<your-username>`.
Send me your `GitHub username` (register now if not yet) and I will set both up for you — see [](1_workflow.md#submission).

(ai-policy)=
## 🤖 AI policy

The course is designed to be *AI-friendly* and to encourage the use of AI tools in the learning process.

You are welcome to use AI tools, but you bear all responsibility for the produced work.

In particular, it implies that you have to understand deeply every line of code or prose that you let AI produce on your behalf.

You have to *verify* the correctness of the code, and your understanding of the AI content and ability to modify it on the fly will be tested in the exam.

In addition, you have to disclose the use of AI assistants beyond text editing: see the [disclosure policy](#disclosure-policy).

See [AI adoption stages](https://claude.ai/code/artifact/bfdfaef9-bc62-4dfe-ba9e-c58a26c9accf) by [Boris Cherny](https://borischerny.com/about/), the man behind Anthropic's Claude Code.


:::{div}
:class: discussion
Where level are you in using AI today?
:::



