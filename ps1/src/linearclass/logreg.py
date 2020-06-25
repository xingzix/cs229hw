import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    logistic = LogisticRegression()
    logistic.fit(x_train, y_train)
    y_valid = logistic.predict(x_valid)

    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path
    np.savetext(save_path, y_valid)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient(self, x, y):
        h_theta = self.sigmoid(x @ self.theta)
        m = x.shape[0]
        gradient = 1 / m * (x.T @ (h_theta - y))
        return gradient

    def hessian(self, x):
        h_theta = self.sigmoid(x @ self.theta)
        m = x.shape[0]
        D = h_theta @ (1 - h_theta).T
        hessian = 1 / m * (D @ x @ x.T)
        return hessian

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        m, n = np.shape(x)  # m =n_examples; n = n_dim
        old_theta = np.zeros((n, 1))
        self.theta = np.ones((n, 1))
        n_iter = 0
        while np.linalg.norm(old_theta - self.theta) > self.eps and n_iter < self.max_iter:
            g = self.gradient(x, y)
            h = self.hessian(x)
            old_theta = self.theta
            self.theta = self.theta - self.step_size * (np.linalg.solve(h, g))
            n_iter += 1
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        y_hat = self.sigmoid(x.dot(self.theta))
        return y_hat
        # *** END CODE HERE ***


if __name__ == '__main__':
    main(train_path='/Users/cindyxu/Documents/cs229/assignments/ps1/src/linearclass/ds1_train.csv',
         valid_path='/Users/cindyxu/Documents/cs229/assignments/ps1/src/linearclass/ds1_valid.csv',
         save_path='/Users/cindyxu/Documents/cs229/assignments/ps1/src/linearclass/logreg_pred_1.txt')

    main(train_path='/Users/cindyxu/Documents/cs229/assignments/ps1/src/linearclass/ds2_train.csv',
         valid_path='/Users/cindyxu/Documents/cs229/assignments/ps1/src/linearclass/ds2_valid.csv',
         save_path='/Users/cindyxu/Documents/cs229/assignments/ps1/src/linearclass/logreg_pred_2.txt')
