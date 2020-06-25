import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')

factor = 2.0


class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        a = X.T.dot(X)
        b = X.T.dot(y)
        self.theta = np.linalg.solve(a, b)
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        for i in range(2, k + 1):
            added = (X[:, 1] ** i).reshape(X.shape[0], 1)
            X = np.hstack((X, added))
        return X
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        added2 = (np.sin(X[:, 1])).reshape(X.shape[0], 1)
        X = np.hstack((X, added2))
        return X
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        h_x_hat = X.dot(self.theta)
        return h_x_hat
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor * np.pi, factor * np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        model = LinearModel()
        phi_x = model.create_poly(k, train_x)
        if sine:
            phi_x = model.create_sin(k, phi_x)
        model.fit(phi_x, train_y)

        phi_plot_x = model.create_poly(k, plot_x)
        if sine:
            phi_plot_x = model.create_sin(k, phi_plot_x)
        plot_y = model.predict(phi_plot_x)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all experiments
    '''
    # *** START CODE HERE ***
    run_exp(train_path, ks=[3], filename='Q4(b).png')
    run_exp(train_path, ks=[3, 5, 10, 20], filename='Q4(c).png')
    run_exp(train_path, sine=True, ks=[0, 1, 2, 3, 5, 10, 20], filename='Q4(d).png')
    run_exp(small_path, ks=[1, 2, 5, 10, 20], filename='Q4(e).png')
    # *** END CODE HERE ***


if __name__ == '__main__':
    main(train_path='/Users/cindyxu/Documents/cs229/assignments/ps1/src/featuremaps/train.csv',
         small_path='/Users/cindyxu/Documents/cs229/assignments/ps1/src/featuremaps/small.csv',
         eval_path='/Users/cindyxu/Documents/cs229/assignments/ps1/src/featuremaps/test.csv')
