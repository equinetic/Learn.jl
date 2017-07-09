using Learn, RDatasets, Plots
gr()

# Data
mtcars = dataset("datasets", "mtcars")

x = Matrix(mtcars[:, 3:end])
y = Matrix(mtcars[:, :MPG]')'

model = Model(LinearRegression())
model.weights = zeros(1, size(x, 2))

learn!(model, x, y)
cost(model, x, y)

yh = predict(model, x)
scatter(1:32, vec(y), label="MPG", title="MTCARS MPG", ylab="MPG", xlab="Record")
plot!(vec(yh), m=3, c="red", label="Predicted")
