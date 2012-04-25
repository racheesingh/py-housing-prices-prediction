#!/usr/bin/python

from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plotData( data ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100
    for c, m, zl, zh in [('r', 'o', -50, -25)]:
        xs = data[:, 0]
        ys = data[:, 1]
        zs = data[:, 2]
        ax.scatter(xs, ys, zs, c=c, marker=m)

    ax.set_xlabel('Size of the House')
    ax.set_ylabel('Number of Bedrooms')
    ax.set_zlabel('Price of the House')

    plt.show()

def feature_normalize(X):
    '''
    Normalizing the feature vector X.
    '''
    mean_r = []
    std_r = []

    X_norm = X
    
    # n_c is the number of features X has
    n_c = X.shape[1]
    
    # For each and every feature, compute mean and standard deviation
    for i in range(n_c):
        m = mean(X[:, i])
        s = std(X[:, i])
        mean_r.append(m)
        std_r.append(s)
        X_norm[:, i] = (X_norm[:, i] - m) / s

    return X_norm, mean_r, std_r

def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''

    #Number of training samples
    m = y.size
    predictions = X.dot(theta)
    sqErrors = (predictions - y)
    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)

    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = zeros( shape=( num_iters, 1 ) )

    for i in range( num_iters ):
        predictions = X.dot( theta )
        theta_size = theta.size

        errors = ( predictions - y )
        theta = theta - ( alpha * 1.0 )/m * X.T.dot( errors )
        
        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history

def main():
    data = loadtxt( 'data_housing', delimiter=',' )

    plotData( data )
    num_features = data.shape[1] - 1
    
    # Assuming all featues are listed first in each training example
    # followed by the target variable

    X = data[ :, : num_features ]
    y = data[ :, num_features ]
    
    # Number of training samples
    m = y.size

    y.shape = (m, 1)

    # Performing feature normalization
    # X_norm = normalized features,
    # mean_vector = vector of means for each feature,
    # std_vector = vector of standard deviations of each feature
    
    X_norm, mean_vector, std_vector = feature_normalize( X )

    # Add intercept term to X
    it = ones( shape=( m, num_features + 1 ) )
    it[:, 1:num_features + 1] = X_norm

    # Grdient descent iternations and step size
    iterations = 100
    alpha = 0.01

    # Initial Theta initialized to 0
    theta = zeros( shape=(num_features + 1, 1) )
    
    theta, J_history = gradient_descent(it, y, theta, alpha, iterations)
    print theta
    plot(arange(iterations), J_history)
    xlabel('Iterations')
    ylabel('Cost Function')
    show()

    #Predict price of a 1650 sq-ft 3 br house
    price = array([1.0, ((1650.0 - mean_vector[0]) / std_vector[0]), \
                   ((3 - mean_vector[1]) / std_vector[1])]).dot(theta)
    print 'Predicted price of a 1650 sq-ft, 3 br house: %f' % (price)

if __name__ == "__main__":
    main()
