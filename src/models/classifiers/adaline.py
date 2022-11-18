import numpy as np
from utils.activition_functions import linear, signum


class Adaline(object):
    def __init__(self, eta=0.01, n_iter=50, bias=True, mse_th=None, a_func=linear):
        self.eta = eta
        self.n_iter = n_iter
        self.mse_th = mse_th
        self.bias = bias
        self.a_func = a_func

    def fit(self, X, Y):

        if self.bias:
            self.w_ = np.random.rand(1 + X.shape[1])
            b = np.ones((X.shape[0], 1))
            X = np.append(X, b, axis=1)
        else:
            self.w_ = np.random.rand(X.shape[1])

        for i in range(self.n_iter):
            for xi, yi in zip(X, Y):
                output = self.activation(xi)
                err = (yi - output)
                self.w_ += self.eta * xi.dot(err)

            errors = Y - self.activation(X)
            self.mse_ = ((errors ** 2) / 2).sum() / X.shape[0]
            if self.mse_th and self.mse_ < self.mse_th:
                break

        return self

    def net(self, X):
        """ Calculate net input """

        return np.dot(X, self.w_)

    def activation(self, X):
        """ Compute linear activation """

        return self.a_func(self.net(X))

    def predict(self, X):
        """ Return class label after unit step """

        if self.bias:
            b = np.ones((X.shape[0], 1))
            X = np.append(X, b, axis=1)

        return np.where(self.activation(X) >= 0, 1, -1)
