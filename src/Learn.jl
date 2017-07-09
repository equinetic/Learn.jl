#__precompile__()
module Learn

using   Reexport
@reexport using LossFunctions
@reexport using PenaltyFunctions
@reexport using MLDataUtils

using   Optim

export  Model,
          Algorithm,
            LinearRegression,
          Objective,
            MSE,
          Penalty,
          Solver,
            GradientDescent,

        # Low level API
        obj_cost,
        obj_grad,
        penalty,
        penalty_grad,
        solve!,

        # High level API
        predict,
        learn!,
        cost

# Parent types
abstract type Algorithm end
abstract type Objective end
abstract type Solver end

# Include type definitions
include(joinpath("regularizers", "regularizers.jl"))
include(joinpath("objectives", "objectives.jl"))

mutable struct Model
  weights::AbstractVecOrMat
  algorithm::A where A <: Algorithm
  objective::O where O <: Objective
  penalty::P where P <: Penalty
  solver::S where S <: Solver
end

# Include solvers, implementations
include(joinpath("solvers", "solvers.jl"))
include(joinpath("supervised", "supervised.jl"))

# API
function predict(m::Model, x)
  predict(m.algorithm, m.weights, x)
end

function cost(m::Model, x, y)
  obj_cost(m.objective, y, predict(m, x))
end


end
