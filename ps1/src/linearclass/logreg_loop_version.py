import numpy as np
import matplotlib.pyplot as plt
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path,add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path,add_intercept = True)



    # *** START CODE HERE ***

    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set
    # Use np.savetxt to save predictions on eval set to save_path
    y_train = np.expand_dims(y_train,1)
    clf = LogisticRegression()
    theta = clf.fit(x_train,y_train)
    y_pred = clf.predict(x_valid)
    np.savetxt('/Users/cindyxu/Documents/cs229/assignments/ps1/src/linearclass/Q1(b)probabilities.txt',y_pred)
    util.plot(x_valid, y_valid, theta, save_path)
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

    def sigmoid(self, x):
        # Numerically stable sigmoid function.
        z = np.exp(x)
        return z / (1 + z)

    def gradient(self, x, y):
        h_theta = self.sigmoid(x@self.theta)
        m = np.shape(x)[0]
        gradient = 1 / m * (x.T) @ (h_theta - y)
        return gradient

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        m, d = np.shape(x)  # m =n_examples; n = n_dim
        old_theta = np.ones((d, 1))
        self.theta = np.zeros((d, 1))
        n_iter = 0
        while (np.linalg.norm(old_theta-self.theta) > self.eps and n_iter < self.max_iter):
            g = self.gradient(x,y)
            h = np.zeros((d,d))
            for i in range(m):
                x_i = x[i].reshape((3,1))
                h_theta = self.sigmoid(x[i] @ self.theta)
                h_i = x_i @ (x_i.T) * (h_theta @ (1-h_theta))
                h = h + h_i
            h = h / m
            #print("g",g.shape)
            #print("h",h.shape)
            #print("theta",self.theta.shape)
            h_inv = np.linalg.inv(h)
            old_theta = self.theta.copy()
            self.theta = self.theta - self.step_size*h_inv@g
            n_iter += 1
            #print(np.linalg.norm(old_theta-self.theta))
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        y_prediction = self.sigmoid(x@ self.theta)
        #print(y_prediction)
        return y_prediction
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path="/Users/cindyxu/Documents/cs229/assignments/ps1/src/linearclass/ds1_train.csv",
         valid_path="/Users/cindyxu/Documents/cs229/assignments/ps1/src/linearclass/ds1_valid.csv",
         save_path='/Users/cindyxu/Documents/cs229/assignments/ps1/src/linearclass/logreg_pred_1.png')

    main(train_path="/Users/cindyxu/Documents/cs229/assignments/ps1/src/linearclass/ds2_train.csv",
         valid_path="/Users/cindyxu/Documents/cs229/assignments/ps1/src/linearclass/ds2_valid.csv",
         save_path='/Users/cindyxu/Documents/cs229/assignments/ps1/src/linearclass/logreg_pred_2.png')
