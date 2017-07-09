# Data
x = hcat(ones(25), rand(25, 3))
truth_theta = [2.5, 1.4, 2.8, 3.6]'
y = reshape(truth_theta*x', (25,1))
theta = randn(1, size(x, 2))

model = Model(LinearRegression())
model.weights = theta

predict(model, x)
learn!(model, x, y)
@test cost(model, x, y) <= 1e-10

yh = predict(model, x)
y

obj_cost(MSE(), y, yh)
obj_grad(MSE(), y, yh, x)
