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


def plot_decesion_boundary(df: pd.DataFrame, f1: str, f2: str, class_column: str, w):
    classes = df[class_column].unique().tolist()
    ax = plt.axes()
    mi = 9999999999999999999

    for cls in classes:
        cls = df[df[class_column] == cls]
        mi = min(cls[f1].min(), cls[f2].min())
        ax.scatter(cls[f1], cls[f2])

    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    print(w)
    ax.plot([0, -w[2]/w[1]], [-w[2]/w[0], 0])
    ax.legend(classes)
    plt.show()
