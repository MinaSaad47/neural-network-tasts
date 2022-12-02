import numpy as np


class DLM():
    def __init__(self, sizes, epochs=1000, lr=0.01, usebias=True, usetanh=False):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr
        self.usebias = usebias
        self.actfunc = DLM.sigmoid if not usetanh else DLM.tanh
        self.params_init()

    def params_init(self):
        self.params = {}
        for i in range(1, len(self.sizes)):
            self.params[f'W{i}'] = np.random.rand(
                self.sizes[i], self.sizes[i - 1])
            if self.usebias:
                self.params[f'B{i}'] = np.random.rand(
                    self.sizes[i]).reshape(-1, 1)
            else:
                self.params[f'B{i}'] = np.zeros(self.sizes[i]).reshape(-1, 1)

    def tanh(v, dv=False):
        return np.tanh(v) if not dv else (1 - v) * (1 + v)

    def sigmoid(v, dv=False):
        return 1 / (1 + np.exp(-v)) if not dv else v * (1 - v)

    def forward_prop(self, x):
        self.params['A0'] = x
        for i in range(1, len(self.sizes)):
            z = np.dot(
                self.params[f'W{i}'], self.params[f'A{i - 1}']) + self.params[f'B{i}']
            self.params[f'A{i}'] = self.actfunc(z)

        return self.params[f'A{len(self.sizes) - 1}']

    def backword_prop(self, o, y):
        self.params[f'G{len(self.sizes) - 1}'] = (y - o) * \
            self.actfunc(o, dv=True)
        for i in reversed(range(2, len(self.sizes))):
            self.params[f'G{i - 1}'] = np.dot(self.params[f'W{i}'].T, self.params[f'G{i}']) * \
                self.params[f'A{i - 1}'] * (1 - self.params[f'A{i - 1}'])

    def update_prop(self):
        for i in range(1, len(self.sizes)):
            self.params[f'W{i}'] += self.lr * np.dot(
                self.params[f'G{i}'].reshape(-1, 1), self.params[f'A{i-1}'].reshape(-1, 1).T)
            if self.usebias:
                self.params[f'B{i}'] += self.lr * self.params[f'G{i}']

    def caclutate_accuracy(pred, true):
        return np.mean(np.argmax(pred, axis=0) == np.argmax(true, axis=0))

    def predict(self, x):
        return self.forward_prop(x).T

    def train(self, x, y):
        accuracies = []
        for e in range(self.epochs):
            for xi, yi in zip(x.T, y.T):
                xi = xi.reshape(-1, 1)
                yi = yi.reshape(-1, 1)
                o = self.forward_prop(xi)
                self.backword_prop(o, yi)
                self.update_prop()
            y_pred = self.forward_prop(x)
            accuracy = DLM.caclutate_accuracy(y_pred, y)
            accuracies.append(accuracy)
            print(f'accuracy[{e}]: {accuracy}')

        return accuracies
