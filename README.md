# Learn.jl

**WORK IN PROGRESS**

## Overview

Machine learning package with implementations of common algorithms and techniques.

## Installation

```julia
Pkg.clone("https://github.com/equinetic/Learn.jl")
using Learn
```

## Current Capabilities

- **Supervised Algorithms**
  - Linear Regression
  - Logistic Regression

## Planned
- **Algorithms**: Reimplementation of [LightML.jl](https://github.com/memoiry/LightML.jl) models
- **Model Evaluation**
  - Comprehensive build out of classification / regression cost functions
  - Extensible ROC Curves
  - Learning Curves
- **Utilities**
  - Variable binner (discretization)
  - Feature scaler
  - Pipeline
  - Easy partitioning

## To do

See the [TODO.md](TODO.md) for a list of short to mid-term tasks.


## Examples
* [Linear Regression](examples/linear_regression_example.jl)
* [Logistic Regression](examples/logistic_regression_example.jl)
