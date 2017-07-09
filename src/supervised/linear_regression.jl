type LinearRegression <: Algorithm end

function Model(a::LinearRegression)::Model
  Model([], LinearRegression(), MSE(), Penalty(), GradientDescent())
end

function predict(a::LinearRegression, θ, x)
  θ * x'
end

function learn!(m::Model, x, y; args...)
  solve!(m.solver, m, x, y; args...)
end
