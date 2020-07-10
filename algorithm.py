""" Linear regression with gradient descent """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fmin_tnc



def sigmoid(x):
    """ Sigmoid function or activation function

    Parameters:
        X (vector): Training examples data
    
    Retunrs:
        (number): Probabilty of training example
    """
    return 1 / (1 + np.exp(-x))

def hθ(theta, X):
    """ Logistic regretion hypotesis, to describe our data

    Parameters:
        theta (vector): Parameters verctor
        X (matrix): Training examples data
    
    Retunrs:
        (vector): Vector of probabilities
    """
    return sigmoid(np.dot(X, theta))

def J(theta, X, y):
    """ Cost function J, convex sigmoid function

    Parameters:
        theta (vector): Parameters verctor
        X (matrix): Training examples data
        y (vector): Labels for our data

    Retunrs:
        (number): Convex sigmoid function
    """

    cost = 1/len(y) * (
        np.dot((-y).transpose(), np.log(hθ(theta, X))) -
        np.dot((1 - y).transpose(), np.log(1 - hθ(theta, X))))
    return cost

def derivated_term_J(theta, X, y):
    """ Logistic regression function derivated

    Parameters:
        theta (vector): Parameters verctor
        X (matrix): Training examples data
        y (vector): Right answers for our data

    Retunrs:
        (vector): Tentative parameters theta
    """
    θ =  1/len(y) * (np.dot(X.T, hθ(theta, X) - y))
    return θ

def gradient_descent(theta, X, y, α, iterations):
    """ Gradient descent algorithm

    Parameters:
        theta (vector): Parameters verctor
        X (matrix): Training examples data
        y (vector): Right answers for our data
        iterations (number): Number of iterations to converge

    Retunrs:
        theta (vector): Parameters theta found
        cost_history (vector): History of J values for each iteration
    """
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        theta = theta - α * derivated_term_J(theta, X, y)
        cost_history[i]  = J(theta, X, y)

    return theta, cost_history

def plot_results(data, theta):
    """ Plotting results, our straight line and data examples

    Parameters:
        theta (vector): Parameters verctor
        size (number): Max size to plot straight line
    """
    data.plot.scatter(x="x1", y="x2", c="y", colormap="jet")
    x = np.linspace(20,100,100)
    y = ((-1 * theta[0]) - theta[1] * x) / theta[2]
    plt.plot(x, y, 'r-')
    plt.show()

def gradient_descent_debugger(iterations, cost_history):
    """ Plotting history for each iteration, it helps to know
        what is happening with our algorithm

    Parameters:
        iterations (number): Number of iterations to converge
        cost_history (vector): History of J values for each iteration
    """
    fig,ax = plt.subplots(figsize=(12,8))
    ax.set_ylabel('J(θ)')
    ax.set_xlabel('Iterations')
    ax.plot(range(iterations), cost_history, 'b.')
    plt.show()

def pro_min_functions(theta, X, y):
    """ Python minimizing tool kit <3 """
    return fmin_tnc(
        func=J, x0=theta, fprime=derivated_term_J,
        args=(X, y))[0]

if __name__ == "__main__":
    data = pd.read_csv("training_data.csv")
    data["bias"] = 1
    X = data[["bias", "x1", "x2"]].values
    y = data["y"]

    """ Symple gradient descent works, but I spend
        10000000 iterations to reach optima!!!, pro
        python minimizing tool kit did it immediatly hahaha
        it is amazing 
    """
    # α = 0.001
    # iterations = 10000000
    theta = np.array([0, 0, 0]).reshape(3, 1)

    """ I left commented my dear gradient descent :) """
    # theta, cost_history = gradient_descent(
    #     theta, X, y, α, iterations)
    # gradient_descent_debugger(iterations, cost_history)

    theta = pro_min_functions(theta, X, y.values)
    plot_results(data, theta)
