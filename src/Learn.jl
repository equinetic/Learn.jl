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
            LogisticRegression,
          Objective,
            MSE,
            Logit,
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
        cost,

        # Utilities
        onehot,
        unhot,

        QuantileBinner,
        NumericBinner,

        FeatureScaler,
          StandardizeScaler,
            standardize,
          RescaleScaler,
            rescale,
          UnitLengthScaler,
            unitlength,

        # Model Evaluation
        ClassificationStatistics,
        ConfusionMatrix,
        ROCCurve,
            roc,

        # Classification Eval Functions
        accuracy,
        classification_error,
        f1score,
        false_discovery_rate,
            fdr,
        false_negative_rate,
            fnr,
        false_positive_rate,
            fallout,
            fpr,
        informedness,
        markedness,
        matthews_corrcoef,
            mcc,
        negative_predictive_value,
            npv,
        positive_predictive_value,
            ppv,
            precision,
        true_negative_rate,
            spc,
            specificity,
            tnr,
        true_positive_rate,
            recall,
            sensitivity,
            tpr

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

# Include solvers, implementations, utilities
include(joinpath("solvers", "solvers.jl"))
include(joinpath("supervised", "supervised.jl"))
include(joinpath("model_evaluation", "model_evaluation.jl"))
include(joinpath("utils", "utils.jl"))
include(joinpath("utils", "binning.jl"))
include(joinpath("utils", "scaling.jl"))

# API
function predict(m::Model, x)
  predict(m.algorithm, m.weights, x)
end

function cost(m::Model, x, y)
  obj_cost(m.objective, y, predict(m, x))
end


end
