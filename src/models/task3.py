from models.classifiers import DLM
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils.preprocessing import label_encode


def train(df: pd.DataFrame, layers: [int], lr: float, epochs: int, usebias: bool, usetanh: bool):
    df['species'] = label_encode(df['species'])

    df['gender'] = label_encode(df['gender'])
    df['gender'][df['gender'] == -1] = df['gender'].max()
    df['gender'].describe()

    scaler = MinMaxScaler()
    df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

    labels = df.iloc[:, 0].to_numpy()
    tmp = np.zeros((labels.size, labels.max() + 1))
    tmp[np.arange(labels.size), labels] = 1
    labels = tmp

    features = df.iloc[:, 1:].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.4, shuffle=True)

    model = DLM(sizes=layers, epochs=epochs, lr=lr,
                usebias=usebias, usetanh=usetanh)
    accuracies = model.train(x_train.T, y_train.T)
    plt.plot(accuracies)
    plt.xlabel('epochs')
    plt.ylabel('accuracies')
    plt.show()

    y_pred = model.predict(x_test.T)
    test_accuracy = DLM.caclutate_accuracy(y_pred.T, y_test.T)
    return accuracies[-1], test_accuracy
