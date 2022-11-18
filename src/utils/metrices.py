import numpy as np


def accuracy(Y_Pred, Y_True) -> float:
    return (Y_Pred == Y_True).sum() / Y_True.shape[0]


def accuracy1(x_test, y_test, weight, actfunc) -> float:
    nb_correct = 0
    for x, d in zip(x_test, y_test):
        net = np.dot(x, weight)
        y = actfunc(net)
        if y == d:
            nb_correct += 1
    return nb_correct / x_test.shape[0]
