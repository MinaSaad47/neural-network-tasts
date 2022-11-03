import numpy as np


def accuracy(x_test, y_test, weight, actfunc) -> float:
    nb_correct = 0
    for x, d in zip(x_test, y_test):
        net = np.dot(x, weight)
        y = actfunc(net)
        if y == d:
            nb_correct += 1
    return nb_correct / x_test.shape[0]
