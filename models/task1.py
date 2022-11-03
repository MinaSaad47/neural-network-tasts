import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils.preprocessing import label_encode
from utils.activition_functions import signum
from utils.metrices import accuracy
from utils.visualization import visualize_features, plot_decesion_boundary


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    features = df.columns.values[1:]
    class_column = df.columns.values[0]
    classes = df[class_column].unique().tolist()
    print(f'features: {features}')

    # label encoding
    print('before encoding\n', df.head(n=10))
    df['gender'] = label_encode(df['gender'])
    print('ater encoding\n', df.head(n=10))

    # removing nans
    print('number of gender nan: ', df['gender'].isna().count())
    df['gender'][df['gender'] == -1] = df['gender'].max()
    print('number of gender nan: ', df['gender'].isna().count())
    print('ater encoding\n', df.head(n=10))
    return df


def visualize(df: pd.DataFrame, f1, f2):
    class_column = 'species'
    visualize_features(df, f1, f2, class_column)


def train(df: pd.DataFrame, f1, f2, c1, c2, eta, nb_epochs, is_bias) -> float:
    features = [f1, f2]
    W = np.array([-1, 0])

    # check if bias requested
    if is_bias:
        df['bias'] = 1
        features += ['bias']
        W = np.append(W, 0.2)

    # select classes
    class_column = 'species'

    C1 = df[df[class_column] == c1].copy()
    C1[class_column] = 1

    C2 = df[df[class_column] == c2].copy()
    C2[class_column] = -1

    # select features
    C1_X = C1[features]
    C1_D = C1[class_column]

    C2_X = C2[features]
    C2_D = C2[class_column]

    # split 3:2 for train:test
    C1_X_Train, C1_X_Test, C1_D_Train, C1_D_Test = train_test_split(
        C1_X, C1_D, test_size=0.4)
    C2_X_Train, C2_X_Test, C2_D_Train, C2_D_Test = train_test_split(
        C2_X, C2_D, test_size=0.4)

    # combine 2 train classes and 2 test classes
    X_Train = pd.concat([C1_X_Train, C2_X_Train]).to_numpy()
    D_Train = pd.concat([C1_D_Train, C2_D_Train]).to_numpy()
    X_Test = pd.concat([C1_X_Test, C2_X_Test]).to_numpy()
    D_Test = pd.concat([C1_D_Test, C2_D_Test]).to_numpy()

    for i in range(nb_epochs):
        error_found = False
        for x, d in zip(X_Train, D_Train):
            net = (x * W).sum()
            y = signum(net)
            if y != d:
                error_found = Tr
                error = d - y
                W = W + (eta * x * error)

        if not error_found:
            break

    Y_Train = np.zeros(shape=D_Train.shape)
    for i, x in enumerate(X_Train):
        y = (x * W).sum()
        Y_Train[i] = y

    df = df[(df[class_column] == c1) | (df[class_column] == c2)]
    plot_decesion_boundary(df, f1, f2, class_column, X_Train, Y_Train)

    signum_vectorizor = np.vectorize(signum)
    return accuracy(X_Test, D_Test, W, signum_vectorizor)
