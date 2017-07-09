feature = 10rand(100) .* sin.(randn(100))

s = StandardizeScaler()
learn!(s, feature)
predict(s, feature)

s = RescaleScaler()
learn!(s, feature)
predict(s, feature)

s = UnitLengthScaler()
learn!(s, feature)
predict(s, feature)
