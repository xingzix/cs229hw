# Important note: you do not have to modify this file for your homework.

import util
import numpy as np


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape
    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(2*X, Y, theta)
        theta = theta + learning_rate * (1/(i**2)) * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)

        '''
        if i/10000 == 1:
            util.plot(X, Y, theta, '/Users/cindyxu/Documents/cs229/assignments/ps2/src/stability/ds_2_1.png')
        if i/10000 == 2:
            util.plot(X, Y, theta, '/Users/cindyxu/Documents/cs229/assignments/ps2/src/stability/ds_2_2.png')
        if i/10000 == 3:
            util.plot(X, Y, theta, '/Users/cindyxu/Documents/cs229/assignments/ps2/src/stability/ds_2_3.png')
        '''

        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    '''
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya)
    '''
    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb)


if __name__ == '__main__':
    main()
