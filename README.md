# Logistic Regression with Gradient Descent
As a supervised learning algorithm, our dataset tell us how data examples are labeled. Our job is to generate a decision boundary that separetes our data in positive and negative examples (labeled as y=0 or y=1)

Once we have our decision boundary, we can predict label given a new example data

![logistic_regression](https://github.com/JhonSanz/logistic_regression/blob/master/logistic_regression.png?raw=true)


# Logistic regression model

## Hypotesis hθ(X)
Our hypotesis is going to use sigmoid function, because it is usefull to describe data, like this:

- g(z) = 1 / 1 + e^-z
- hθ(X) = g(θTX)

y = 1 if x >= 0; y = 0 if x < 0

θTX is going to describe our decision boundary because:

- θTX = θ0X0 + θ1X1 + ... + θnXn

and predict y = 1 is to say θ0X0 + θ1X1 + ... + θnXn >= 0; because:
- hθ(X) = g(z)
- g(z) >= 0.5 if z >= 0
- hθ(X) = g(θTX) where θTX >= 0

And again we want to choose the best decision boundary choosing parameters θ

## Cost Function

Our cost function is going to be squared error function, that computes distance for every training examples. We need to minimice this function, because our job is to fit the better decision boundary that separates our data, so:

- J(θ) = 1/2m sum([(hθ(Xi) - yi)**2 for i in range(m)])

There is a problem, our actually hθ(X) do no describe a convex function, so we are going to write this like so:

- J(θ) = 1/m sum([yi * log(hθ(xi)) + (1-yi) * log(1 - hθ(xi)) for i in range(m)])

Written vectorized, simplifies our equation

- J(θ) = 1/m (-yT * log(hθ(xi)) - (1-y)T * log(1 - hθ(xi)))

# Gradient Descent

To minimice using gradient descent we need derivate terms of our cost function, so, we use some calculus over our cost function and it gave us this:

- θ = 1/m sum([(hθ(Xi) - yi) for i in range(m)])xj

We have derivated over θ

See derivation here https://medium.com/analytics-vidhya/derivative-of-log-loss-function-for-logistic-regression-9b832f025c2d

Written vectorized, simplifies our equation

- θ = 1/2m XT (hθ(Xi) - y)

And Gradient Descent algorithm is defined by

- θ = θ - α d/dθ J(θ)

Where α is the learning rate

So

- θ = θ - α (1/2m XT (hθ(X) - y))

Thanks to Coursera and Andrew Ng, I encourage you to take this course:
https://www.coursera.org/learn/machine-learning/home/welcome

Regards :)
