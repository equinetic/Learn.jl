using Learn

# Data
x = rand(25, 3)
truth_theta = [1.4, 2.8, 3.6]'
theta = zeros(size(truth_theta))

y = truth_theta*x'

model = Model(LinearRegression())
model.weights = theta

predict(model, x)
learn!(model, x, y)
cost(model, x, y)

println("Truth: ", truth_theta)
println("Learned: ", model.weights)
