# classification-logistic_reg-
The model builds a regression model to predict the probability that a given data entry belongs to the category numbered as “1”. 
Just like Linear regression assumes that the data follows a linear function, Logistic regression models the data using the sigmoid
function.

g(z) = \frac{1}{1 + e^-^z}\ 

What is the Sigmoid Function?
In order to map predicted values to probabilities, we use the Sigmoid function. The function maps any real value into another value
between 0 and 1. In machine learning, we use sigmoid to map predictions to probabilities.

Hypothesis Representation
When using linear regression we used a formula of the hypothesis i.e.
hΘ(x) = β₀ + β₁X
For logistic regression we are going to modify it a little bit i.e.
σ(Z) = σ(β₀ + β₁X)
We have expected that our hypothesis will give values between 0 and 1.
Z = β₀ + β₁X
hΘ(x) = sigmoid(Z)
i.e. hΘ(x) = 1/(1 + e^-(β₀ + β₁X)


