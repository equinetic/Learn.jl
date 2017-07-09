type MSE <: Objective end

# Mean squared error
function obj_cost(o::MSE, y, ŷ)
  value(L2DistLoss(), y, ŷ, AvgMode.Mean())
end

# MSE Gradient
function obj_grad(o::MSE, y, ŷ, x)
  deriv(L2DistLoss(), y, ŷ) * x
end
