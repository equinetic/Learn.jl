# Generate binary and multiclass sets
n = 100
x = hcat(ones(n), rand(n, 3))
truth_theta = [2.5, -1.4, 2.8, -3.6]'
v = 1 ./ (1 .+ exp(1) .^ (-truth_theta*x'))

# Binary set
y_binary = reshape(classify(vec(v), LabelEnc.ZeroOne(Float64, .75)), (100, 1))

# Multiclass set
b = QuantileBinner(vec(v), 0.:(1//3):1.)
y_multiclass = predict(b, vec(v), as_int=true)
labels = sort!(label(y_multiclass))
y_mc_onehot = onehot(vec(y_multiclass))

# Test Binary
theta = zeros(1, size(x,2))
model = Model(LogisticRegression())
model.penalty = Penalty(L1DistLoss(), 1e-10)
model.weights = theta

predict(model, x)
cost(model, x, y_binary)
learn!(model, x, y_binary)


# Test Multiclass
theta = randn(size(y_mc_onehot, 2), size(x, 2))
model = Model(LogisticRegression())
model.penalty = Penalty(L2DistLoss(), .001, true)
model.weights = theta
model

predict(model, x)
learn!(model, x, y_mc_onehot)
cost(model, x, y_mc_onehot)

yh = predict(model, x)'
pred = unhot(yh, labels)

sum(y_multiclass .== pred) / length(pred)
