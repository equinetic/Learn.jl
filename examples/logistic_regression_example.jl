using Learn

# Data
x = hcat(ones(25), rand(25, 3))
truth_theta = [2.5, -1.4, 2.8, -3.6]'
v = 1 ./ (1 .+ exp(1) .^ (-truth_theta*x'))
y = reshape(classify(v[1,:], LabelEnc.ZeroOne(Float64, .80)), size(v))

theta = randn(1, size(x, 2))
model = Model(LogisticRegression())
model.weights = theta

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
