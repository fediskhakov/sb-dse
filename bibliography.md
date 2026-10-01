---
title: 📚 Reading list
short_title: 📚 Reading list
downloads:
  - file: bibliography.md
    title: MyST Markdown
---

References by course part; 🔑 marks the key reading of a class.

## Textbooks and general references

- {cite:t}`adda2023DynamicEconomicsQuantitative` — *Dynamic Economics: Quantitative
  Methods and Applications*, the closest thing to a textbook for this course
- {cite:t}`sargent2025DynamicProgrammingFinite` — *Dynamic Programming: Finite States*,
  the theory behind Part II, with open-source code; freely readable at
  [dp.quantecon.org](https://dp.quantecon.org/)
- {cite:t}`judd1998NumericalMethodsEconomics` — *Numerical Methods in Economics*
- {cite:t}`train2009DiscreteChoiceMethods` — *Discrete Choice Methods with Simulation*
- {cite:t}`wooldridge2010EconometricAnalysisCross` — *Econometric Analysis of Cross
  Section and Panel Data*, for the M-estimation asymptotics
- {cite:t}`neweyLargeSampleEstimation1994` — large sample estimation and hypothesis
  testing, *Handbook of Econometrics*

## Part I — foundations and computational toolkit

- {cite:t}`keaneStructuralVsAtheoretic2010` — structural vs atheoretic approaches
- {cite:t}`wolpin2013LimitsInferenceTheory` and the review by
  {cite:t}`rustLimitsInferenceTheory2014`
- {cite:t}`sargent2024CritiqueConsequence` — critique and consequence
- {cite:t}`wilf2002AlgorithmsComplexity` — *Algorithms and Complexity*

Random utility and discrete choice:

- {cite:t}`thurstone1959MeasurementValues` — the measurement of values
- {cite:t}`luce1959IndividualChoiceBehavior` — *Individual Choice Behavior*
- {cite:t}`marschak1960` — binary choice constraints and random utility indicators
- {cite:t}`mcfadden1974ConditionalLogitAnalysisa` — conditional logit analysis of
  qualitative choice behavior
- {cite:t}`mcfadden1981econometric` — econometric models of probabilistic choice

## Part II — single-agent dynamic programming

### Dynamic programming

- {cite:t}`Rust2016` — "Dynamic programming", *The New Palgrave Dictionary of Economics*
- {cite:t}`aguirregabiriaDynamicDiscreteChoice2010` — survey of dynamic discrete
  choice structural models
- {cite:t}`maDynamicProgrammingDeconstructed2021` — dynamic programming deconstructed

### The bus engine model and NFXP

- 🔑 {cite:t}`rustOptimalReplacementGMC1987` — the bus engine replacement model
  ([Class 9](9_zurcher.md))
- {cite:t}`rustNestedFixedPoint2000` — the NFXP manual
- {cite:t}`berndtEstimationInferenceNonlinear1974` — the BHHH algorithm
- {cite:t}`suConstrainedOptimizationApproaches2012` — MPEC, and the
  {cite:t}`ecma_comment` "Comment" on it

### Applications of dynamic discrete choice models

- {cite:t}`rustHowSocialSecurity1997` — social security, Medicare and retirement
- {cite:t}`keaneCareerDecisionsYoung1997` — career decisions of young men
- {cite:t}`ecksteinWhyYouthsDrop1999` — why youths drop out of high school
- {cite:t}`francesconiJointDynamicModel2002` — fertility and labor supply
- {cite:t}`arcidiaconoAbilitySortingReturns2004` — ability sorting and the returns to
  college major
- {cite:t}`kennanEffectExpectedIncome2011` — expected income and migration
- {cite:t}`buchinskyResidentialLocationWork2014` — residential location and work location
- {cite:t}`erdemDecisionMakingUnder1996` — brand choice under uncertainty and learning
- {cite:t}`ackerbergAdvertisingLearningConsumer2003` — advertising, learning and consumer
  choice
- {cite:t}`hendelMeasuringImplicationsSales2006` — sales and consumer inventory
- {cite:t}`fosgerauLinkBasedNetwork2013` — link-based route choice in networks
- {cite:t}`agarwalEquilibriumAllocationsUnder2021` — allocations under alternative
  waitlist designs
- {cite:t}`lee_structural_2025` — opioid misuse, health, labor and policy

### CCP estimation and identification

- 🔑 {cite:t}`hotz1993ConditionalChoiceProbabilitiesb` — the CCP inversion
  ([Class 12](12_ccp.md))
- 🔑 {cite:t}`arcidiaconoConditionalChoiceProbability2011` — the representation of
  conditional value functions, finite dependence, unobserved heterogeneity and EM
  ([Class 12](12_ccp.md))
- {cite:t}`altug1998EffectWorkExperience` — the origin of finite dependence, and minimum
  distance CCP estimation
- {cite:t}`hotz1994SimulationEstimatorDynamicc` — the CCP simulation estimator
- {cite:t}`pesendorfer2008AsymptoticLeastSquares` — asymptotic least squares estimators
- {cite:t}`arcidiacono2019NonstationaryDynamicModels` — nonstationary models with finite
  dependence
- {cite:t}`arcidiaconoIdentifyingDynamicDiscrete2020` — identification off short panels
- {cite:t}`abbringIdentifyingDiscountFactor2020` — identifying the discount factor
- {cite:t}`kalouptsidi2021IdentificationCounterfactualsDynamic` — identification of
  counterfactuals

### Nested pseudo-likelihood

- 🔑 {cite:t}`aguirregabiriaSwappingNestedFixed2002` — swapping the nested fixed point,
  NPL ([Class 14](14_npl.md))
- {cite:t}`pesendorfer2010SequentialEstimationDynamic` — convergence of NPL, a comment
- {cite:t}`aguirregabiriaImposingEquilibriumRestrictions2021` — convergence of NPL in
  games, and the spectral algorithm

## Part III — continuous choice and simulation-based estimation

- {cite:t}`carroll2006MethodEndogenousGridpoints` — the endogenous gridpoint method
- {cite:t}`egm` — DC-EGM for discrete-continuous problems
- {cite:t}`megm` — the multidimensional endogenous gridpoint method
- {cite:t}`druedahl2017GeneralEndogenousGridb` — a general EGM
- {cite:t}`mcfaddenMethodSimulatedMoments1989` and
  {cite:t}`pakes1989SimulationAsymptoticsOptimizationa` — method of simulated moments

## Part IV — equilibrium models

- {cite:t}`iruc2` — "Equilibrium Trade in Automobiles", the running example

## Part V — games

- {cite:t}`ericsonMarkovPerfectIndustryDynamics1995` — Markov perfect industry dynamics
- {cite:t}`aguirregabiriaSequentialEstimationDynamic2007` — dynamic discrete games, NPL
- {cite:t}`bajariEstimatingDynamicModels2007` — BBL
- {cite:t}`suEstimatingDiscretechoiceGames2014` — estimation under multiplicity
- {cite:t}`juddFindingAllPureStrategy2012` — finding all pure strategy equilibria
- {cite:t}`dearing2025EfficientConvergentSequential` — EPL
- {cite:t}`rls` — recursive lexicographical search: finding *all* Markov perfect
  equilibria of directional dynamic games
- {cite:t}`nrls` — structural estimation of directional games with multiple equilibria
- {cite:t}`leapfrogging_io` — Bertrand price competition with cost-reducing investments
