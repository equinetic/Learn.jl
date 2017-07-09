using Learn

# Data
n = 100
x = hcat(ones(n), rand(n, 3))
truth_theta = [2.5, -1.4, 2.8, -3.6]'
v = 1 ./ (1 .+ exp(1) .^ (-truth_theta*x'))
b = QuantileBinner(vec(v), 0.:(1//3):1.)
y = predict(b, vec(v), as_int=true)

labels = sort!(label(y))
y = onehot(y)

theta = randn(size(y, 2), size(x, 2))
model = Model(LogisticRegression())
model.penalty = Penalty(L2DistLoss(), .001, true)
model.weights = theta
model

predict(model, x)
learn!(model, x, y)
cost(model, x, y)

println("Truth: ", truth_theta)
println("Learned: ", model.weights)

yh = classify(predict(model, x)[1,:], LabelEnc.ZeroOne(Float64, .80))
y = y[1,:]


tp = sum(y .== 1 .== yh)
tn = sum(y .== 0 .== yh)
fp = sum(y .== 0 .!= yh)
fn = sum(y .== 1 .!= yh)

@benchmark predict(model, x)
@benchmark obj_grad(model.objective, y, predict(model, x), x)
