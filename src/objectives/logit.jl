type Logit <: Objective end

# Logit loss
function obj_cost(o::Logit, y, ŷ)
  value(LogitDistLoss(), y', ŷ, AvgMode.Mean())
end

# Logit loss gradient
function obj_grad(o::Logit, y, ŷ, x)
  deriv(LogitDistLoss(), y', ŷ) * x
end
