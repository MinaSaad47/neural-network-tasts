import matplotlib.pylab as plt
import pandas as pd


def visualize_features(df: pd.DataFrame, f1: str, f2: str, class_column: str):
    classes = df[class_column].unique().tolist()
    for cls in classes:
        cls = df[df[class_column] == cls]
        plt.scatter(cls[f1], cls[f2])
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.legend(classes)
    plt.show()


def plot_decesion_boundary(df: pd.DataFrame, f1: str, f2: str, class_column: str, x_train, y_pred):
    classes = df[class_column].unique().tolist()
    ax = plt.axes()

    for cls in classes:
        cls = df[df[class_column] == cls]
        ax.scatter(cls[f1], cls[f2], zorder=1)

    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.plot(x_train, y_pred, zorder=2)
    ax.legend(classes)
    plt.show()
