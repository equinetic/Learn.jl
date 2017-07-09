# Wraps PenaltyFunctions.jl
mutable struct Penalty
  func
  λ::Float64
  function Penalty()
    new(NoPenalty(), 0.0)
  end
  function Penalty(f, λ::Float64=0.0)
    new(f, λ)
  end
end

# Penalty cost
function penalty(p::Penalty, θ)
  p.λ * value.(p.func, θ)
end

# Penalty gradient
function penalty_grad(p::Penalty, θ)
  p.λ * grad(p.func, θ)
end

# Penalty cost with scale
function penalty(p::Penalty, θ, s)
  p.λ * value.(p.func, θ, s)
end

# Penalty gradient with scale
function penalty_grad(p::Penalty, θ, s)
  p.λ * grad(p.func, θ, s)
end
