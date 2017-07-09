# Wraps PenaltyFunctions.jl
mutable struct Penalty
  func
  λ::Float64
  intercept::Bool
  function Penalty()
    new(NoPenalty(), 0.0, false)
  end
  function Penalty(f, λ::Float64=0.0, intercept::Bool=false)
    new(f, λ, intercept)
  end
end

# Penalty cost
function penalty(p::Penalty, θ)
  pen = zeros(size(θ))
  pen .= value.(p.func, θ)
  if p.intercept pen[1] = 0. end
  p.λ * pen
end

# Penalty gradient
function penalty_grad(p::Penalty, θ)
  pen = grad(p.func, θ)
  if p.intercept pen[1] = 0. end
  p.λ * pen
end

# Penalty cost with scale
function penalty(p::Penalty, θ, s)
  pen = zeros(size(θ))
  pen .= value.(p.func, θ, s)
  if p.intercept pen[1] = 0. end
  p.λ * pen
end

# Penalty gradient with scale
function penalty_grad(p::Penalty, θ, s)
  pen = grad(p.func, θ, s)
  if p.intercept pen[1] = 0. end
  p.λ * pen
end
