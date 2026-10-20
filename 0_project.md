---
title: 🛟 The course project help
short_title: 🛟 Project help
subtitle: How to choose, build, write up and present your structural estimation project
downloads:
  - file: 0_project.md
    title: MyST Markdown
---

The project is where the course comes together: you build one structural model from the
ground up, solve it, simulate it, estimate it and use it for a counterfactual. It carries
**70% of the grade**, together with the presentation and the oral exam, and it is
individual.

## Timeline

| When | What |
| :-- | :-- |
| now | think about a model and a question; talk to me early |
| **Thursday, November 5** | **project proposal** due, at most 2 pages |
| **Tuesday, November 10** | **project clinic**: every proposal is discussed in class |
| November | build the model, the solver, the simulator and the estimator |
| **Tuesday, December 1 and Thursday, December 3** | **presentations**, about 30 minutes each, with a live demo |
| before the exam week | the **paper and the documented code** are submitted |
| exam period | **oral exam**, with the paper and the code as the starting point |

Do not leave the code for the last two weeks. The solver usually takes longer than
planned, and everything else depends on it.

## What a structural project consists of

The [components of a structural estimation project](1_intro.md#project-components) of
Class 1 are the checklist. A complete project has:

1. **A model**: agents, their choices, states and transitions, preferences and
   constraints, written down precisely enough that someone else could code it
2. **A solver**: code that computes the policy (and value) functions for any parameter
   values, with evidence that it is accurate
3. **A simulator**: code that generates data from the model at given parameters
4. **An estimator**: NFXP, CCP, NPL, MPEC or MSM, with standard errors
5. **Data**: real data, or simulated data from the model with known parameters
6. **Identification**: an argument for which features of the data pin down which
   parameters
7. **A counterfactual**: a change in policy or environment, simulated with the estimated
   model, that answers a question

The aim is a project that works end to end, from the model to the counterfactual. A
small model done completely beats an ambitious model that never gets estimated.

(project-paths)=
### The main line and its alternatives

Every project starts from the same place, a model written as Bellman equations. From
there the main line takes the model to **real data**, but two other kinds of projects are
equally welcome:

```{figure} _static/img/project_paths.svg
:width: 95%
:align: center

Three paths from the Bellman equations to a project.
```

- **Real data**, the main line: describe the data, estimate the structural parameters,
  assess the goodness of fit, and use the estimated model for counterfactuals
- **Synthetic data**: simulate data from the model, calibrate it to some aggregate
  moments, and run a Monte Carlo study of the estimation procedure and the code; the
  counterfactuals are computed as on the main line. A good choice when the data you need
  are not available in the time of the course
- **A theoretical paper**: new solution algorithms, a class of problems they solve,
  econometric methods, or a characterization of the solution, followed by a Monte Carlo
  study of the performance of the proposed methods

The structure of [the paper](#project-paper) stays the same on every path; what changes
is the content of the application, the data and the results sections.

## Choosing a model

There are two routes, and both are fine.

**Extend a model from class.** The course has built the full pipeline for several
models, and the code is in the course code repository:

- the bus engine replacement model of [Classes 9–14](9_zurcher.md), estimated by NFXP,
  CCP and NPL
- the inventory management model of [Class 7](7_dp.md)
- the consumption-savings model of [Classes 15–17](15_cake_eating.md), solved by VFI, time
  iterations and EGM
- the consumption-savings-retirement model of [Class 18](18_dcegm.md), solved by DC-EGM
  and estimated by MSM in [Class 20](20_msm_practice.md)
- the equilibrium and game models of the last part of the course

An extension adds something economically meaningful: a new state variable (health,
experience, a second asset), a new choice (part-time work, a second machine), unobserved
types, a different market structure, or a policy instrument (a pension, a tax, a safety
net). Mechanical changes of a parameter value are not an extension.

**Bring your own model.** A model from your own research interest is welcome, and often
makes the best project. Keep it small:

- one or two state variables
- one discrete choice, or one continuous choice, or one of each
- a handful of structural parameters
- a solver you can run in seconds, otherwise you'll spend too much time waiting

Before you commit, check three things: that the model can be solved with a method from
the course, that its parameters are identified from data you can get or simulate, and
that there is an interesting counterfactual.

:::{div}
:class: discussion

- What question would your counterfactual answer, and who would care about the answer?
- Which method of the course solves your model, and which estimates it? Why those?
:::

## The proposal

The proposal is due on **Thursday, November 5**, at most two pages, and it is the plan
we discuss in the project clinic on **Tuesday, November 10**. It should contain:

- **the question**: one or two sentences, and the counterfactual that answers it
- **the model**: states, choices, transitions, utility, constraints, the timing, and the
  parameters to estimate
- **the solution method**, and why it fits the model
- **the data**: the source, or the plan to simulate it, and the observed variables
- **the estimation method** and the moments or the likelihood
- **a roadmap**: the steps in order, what you already have and what is the main risk
- **references**: the two or three papers the project builds on

A good proposal is specific. "I will estimate a model of retirement" is not a plan;
"I will add a part-time work option to the retirement model of Class 18 and estimate the
disutility of work by MSM from simulated employment profiles" is.

### Make sure that your code works as expected

A structural estimate is only as credible as the code behind it. Convince yourself that
the code is right, and include some tests in the final product.

- **Accuracy of the solver.** Use the checks of the course: an analytic special case,
  [Keane's test](15_cake_eating.md#keane-test),
  [Euler equation residuals](15_cake_eating.md#euler-residuals), convergence as the grid
  is refined
- **Recovery of known parameters.** Simulate data from the model at parameters you
  choose, estimate, and check that you get them back within the standard errors. A
  small Monte Carlo, as in [Class 20](20_msm_practice.md), is the strongest evidence that
  the estimator works
- **Identification.** Plot the likelihood or the criterion around the estimate, along
  each parameter. A flat direction means a parameter the data do not pin down
- **Robustness of the optimizer.** Start from several points; report what happens

```{hint} Start with simulated data

Even if you plan to use real data, estimate the model on simulated data first. If the
estimator cannot recover the parameters that generated the data, it will not recover
anything from real data either, and you will not be able to tell why.
```

(project-paper)=
## The paper

The paper is individual work, and it contains the main steps of a structural research
paper, in this order:

1. **Introduction and literature review**: the question, why it matters, what you do and
   what you find, and how the project relates to the closest papers
2. **The application domain**: the economic setting that motivates the model, the
   institutions and the facts the model has to capture, and the data, real or
   simulated, with descriptive statistics
3. **Model**: the agents, their choices, states and transitions, preferences and
   constraints, the timing, and the **Bellman equations**; every symbol defined
4. **Solution and estimation method, identification**: the algorithm that solves the
   model, with its accuracy checks and run time; the estimator, the likelihood or the
   moments, and the standard errors; which features of the data identify which
   parameters
5. **Estimation results and goodness of fit**: the estimates with standard errors, and
   how well the estimated model fits the data, including the features it does not fit
6. **Counterfactuals**: the change in policy or environment, what the estimated model
   predicts, by how much, and why
7. **Conclusion, limitations and further directions**: what was learned, what the model
   leaves out and how that could matter, and what you would do next

Put the details of the algorithm and the additional checks in an appendix. Every table
and figure must come out of the notebook, and the text must agree with the numbers in
them.

Write for an economist who has not taken this course: define the notation, explain the
choices, and say where the method or the model breaks. Cite the papers you build on with
full references.

## The presentation

The presentations take place on **December 1 and 3**, about 30 minutes each, with
questions. Plan for about 20 minutes of talk:

- the question and the model, in a few slides
- the solver and the estimator **running live**: show the code, solve the model, show
  the simulated data and the fit
- the estimates and the counterfactual
- what does not work yet, and what you would do with more time

A live demo that fails is not a disaster; a demo that you cannot explain is. Rehearse it
on the computer you will present from.

## The oral exam (to be confirmed)

The paper and the code are submitted before the exam week, and they are the
*conversation starter* of the oral exam during the exam period. Expect questions about:

- the economics of the model and the counterfactual
- the solution method and why it works for your model
- the estimator, the standard errors and identification
- **any line of your code**: what it does, why it is written that way, and how you would
  change it

## Using AI tools

The [AI policy](index.md) of the course applies in full. You may use AI tools for the
project, and you are responsible for every line of code and prose they produce for you:
you have to understand it, verify it, and be able to modify it on the spot in the oral
exam. Disclose the use of AI assistants beyond text editing as described in the
[disclosure policy](2_workflow.md#disclosure-policy).

```{hint} A project checklist

- [ ] the model is written down completely: states, choices, transitions, utility,
      constraints, timing
- [ ] the solver passes an accuracy check, and its run time is known
- [ ] the simulator generates data that look like the data you will use
- [ ] the estimator recovers known parameters from simulated data
- [ ] the standard errors are computed and checked
- [ ] the identification of every parameter is argued and illustrated
- [ ] the counterfactual is computed and interpreted
- [ ] one notebook reproduces every number and figure of the paper
- [ ] the README explains how to run it
- [ ] the use of AI tools is disclosed
```
