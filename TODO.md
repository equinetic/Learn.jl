# TODO
- **Data type support**
  - [ ] DataFrames
- **Solvers**
  - [ ] Ensure Optim.jl can remain the go-to solution. Not sure if it can handle stochastic / mini-batch, parallel operations, etc. Should look into the state of StochasticOptimization.
- **Supervised Algorithms**
  - [x] Linear Regression
  - [x] Logistic Regression
  - [ ] GLM
  - [ ] GLRM
  - [ ] Gradient Boosting Tree
  - [ ] AdaBoost
  - [ ] Decision Tree
  - [ ] Random Forests
  - [ ] Gaussian Discriminant Analysis
  - [ ] Hidden Markov Model
  - [ ] K-Nearest Neighbors
  - [ ] Naive Bayes
  - [ ] Multilayer Perceptron
  - [ ] Support Vector Machine
- **Unsupervised Algorithms**
  - [ ] Gaussian Mixture Model
  - [ ] K-Means
  - [ ] Large Scale Spectral Clustering
  - [ ] PCA
- **Model Evaluation**
  - [ ] Update, complete and integrate classification eval
  - [ ] Learning curves - utilize ValueHistories
  - [ ] ROC Curves
- **Extended Capabilities**
  - [ ] Ensemble
  - [ ] Boosting
  - [ ] Bagging
  - [ ] Complete binning.jl
  - [x] Complete scaling.jl - implemented, subject to renaming and other changes
  - [ ] Wrap MLDataUtils.jl for partitioning
  - [ ] Create pipeline function/macro
  - [ ] Maybe more convenient dimensioning. Right now there are strict requirements:
  
  ```
  y = N x K
  ŷ = K x N
  θ = K x M
  x = N X M

  where
    N = number of observations
    M = number of features
    K = number of predicted dimensions
  ```
- **Tests** - get a baseline going; subject to improvements
  - [x] Linear Regression
  - [x] Logistic Regression
  - [ ] Binning
  - [x] Feature Scaling
