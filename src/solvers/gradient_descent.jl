type GradientDescent <: Solver end

function solve!(s::GradientDescent,
                m::Model,
                x,
                y; args...)
    function j(wts::Vector)
      obj_cost(m.objective, y, predict(m.algorithm, shape_theta(wts, shape), x))
    end

    function g!(storage::Vector, wts::Vector)
      g = vec(obj_grad(m.objective, y, predict(m.algorithm, shape_theta(wts, shape), x), x))
      for i in eachindex(g)
        storage[i] = g[i]
      end
    end

    shape = size(m.weights)
    wts = vec(m.weights)
    res = optimize(j, g!, wts, Optim.GradientDescent(), Optim.Options(args...))
    m.weights = reshape(res.minimizer, shape)
end
