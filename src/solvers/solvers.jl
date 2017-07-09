# Shared solver functions
shape_theta(θ::Vector, shape::Tuple) = reshape(θ, shape)

# Include
include("gradient_descent.jl")
