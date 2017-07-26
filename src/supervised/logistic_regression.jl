type LogisticRegression <: Architecture end

function Model(a::LogisticRegression)::Model
  Model([], LogisticRegression(), Logit(), Penalty(), GradientDescent())
end

function predict(a::LogisticRegression, θ ,x)
  1 ./ (1 .+ exp(1) .^ (-θ*x'))
end

function learn!(m::Model, x, y; args...)
  if length(m.weights)==0 m.weights=zeros(1, size(x, 2)) end
  solve!(m.solver, m, x, y; args...)
end
