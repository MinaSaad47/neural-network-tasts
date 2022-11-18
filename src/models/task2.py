import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from utils.preprocessing import label_encode
from utils.activition_functions import linear
from utils.metrices import accuracy
from utils.visualization import visualize_features, plot_decesion_boundary

from models.classifiers import Adaline
from utils.metrices import accuracy
from utils.visualization import plot_decesion_boundary


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


def mse(x: pd.core.series.Series, w: pd.core.series.Series, d) -> float:
    mse = 0
    for xi, di in zip(x, d):
        mse += (xi * di).sum()

    return mse / xi.shape[0]


def train(df: pd.DataFrame, f1, f2, c1, c2, eta, nb_epochs, is_bias, mse_th) -> (float, float):
    features = [f1, f2]
    
    # nomalize feature to prevent overflow
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

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
    
    model = Adaline(eta=eta, n_iter=nb_epochs, bias=is_bias, mse_th=mse_th)
    model.fit(X_Train, D_Train)
    
    D_Pred = model.predict(X_Train)
    create_confusion_matrix(D_Pred, D_Train)
    plot_decesion_boundary(df[(df[class_column] == c1) | (df[class_column] == c2)], f1, f2, class_column, model.w_)

    D_Pred = model.predict(X_Test)
    return accuracy(D_Pred, D_Test), model.mse_



def create_confusion_matrix(D_Pred, D_True):
    conf_y_train = np.zeros(shape=D_True.shape)

    for i in D_True:
        conf_y_train[i] = linear(D_True[i])

    conf_y_pred = np.zeros(shape=D_Pred.shape)

    for i in range(40):
        conf_y_pred[i] = linear(D_Pred[i])

    Numb_of_classes = np.unique(conf_y_train)
    conf_matrix = np.zeros((len(Numb_of_classes), len(Numb_of_classes)))
    for i in range(len(Numb_of_classes)):
        for j in range(len(Numb_of_classes)):
            conf_matrix[i, j] = np.sum((conf_y_train == Numb_of_classes[i]) & (
                conf_y_pred == Numb_of_classes[j]))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=None)
    disp.plot()
    plt.show()