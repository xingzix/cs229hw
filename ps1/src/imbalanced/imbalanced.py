import numpy as np
import util
import sys
from random import random

sys.path.append('../linearclass')

# NOTE : You need to complete logreg implementation first!

from logreg_loop_version import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1


def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(validation_path, add_intercept=True)
    y_train = np.expand_dims(y_train, 1)
    #"""
    vanilla = LogisticRegression()
    vanilla.fit(x_train, y_train)
    y_prediction = vanilla.predict(x_valid)
    np.savetxt(output_path_naive, y_prediction)
    util.plot(x_valid, y_valid, vanilla.theta, '/Users/cindyxu/Documents/cs229/assignments/ps1/src/imbalanced/vanilla.png')
    return confusionMatrix(y_valid, y_prediction)
    #"""
    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times

    """
    m = x_train.shape[0]
    for i in range(m):
        if y_train[i, 0] == 1.0:
            for j in range(int(1/kappa)):
                x_train = np.vstack((x_train, x_train[i, :]))
                y_train = np.vstack((y_train, y_train[i, :]))

    upsampling = LogisticRegression()
    upsampling.fit(x_train, y_train)
    y_prediction_up = upsampling.predict(x_valid)
    np.savetxt(output_path_upsampling, y_prediction_up)
    util.plot(x_valid, y_valid, upsampling.theta, '/Users/cindyxu/Documents/cs229/assignments/ps1/src/imbalanced/upsampling.png')
    return confusionMatrix(y_valid, y_prediction_up)
    """
    # *** END CODE HERE


def confusionMatrix(y_true, y_pred):
    TN, FP, FN, TP = 0, 0, 0, 0
    for i in range(y_pred.shape[0]):
        if y_pred[i, 0] > 0.5:
            y_pred[i, 0] = 1.0
        if y_pred[i, 0] < 0.5:
            y_pred[i, 0] = 0.0

        if y_pred[i, 0] == y_true[i]:
            if y_pred[i, 0] == 0.0:
                TN += 1
            if y_pred[i, 0] == 1.0:
                TP += 1
        if y_pred[i, 0] - y_true[i] == 1.0:
            FP += 1
        if y_pred[i, 0] - y_true[i] == -1.0:
            FN += 1

    A = (TP + TN) / y_true.shape[0]
    A0, A1 = TN / (TN + FP), TP / (TP + FN)
    A_balanced = 1 / 2 * (A0 + A1)
    print("A: ", A)
    print("A0: ", A0)
    print("A1: ", A1)
    print("A_balanced: ", A_balanced)
    return A, A_balanced, A0, A1


if __name__ == '__main__':
    main(train_path='/Users/cindyxu/Documents/cs229/assignments/ps1/src/imbalanced/train.csv',
         validation_path='/Users/cindyxu/Documents/cs229/assignments/ps1/src/imbalanced/validation.csv',
         save_path='/Users/cindyxu/Documents/cs229/assignments/ps1/src/imbalanced/imbalanced_X_pred.txt')
