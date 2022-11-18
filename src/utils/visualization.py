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

    for cls in classes:
        cls = df[df[class_column] == cls]
        ax.scatter(cls[f1], cls[f2])

    ax.set_xlabel(f1)
    ax.set_ylabel(f2)

    p1x1 = 0
    p2x1 = 1

    p1x2 = -(w[2] + w[0]*p1x1) / w[1]
    p2x2 = -(w[2] + w[0]*p2x1) / w[1]

    ax.plot([p1x1, p2x1], [p1x2, p2x2])
    ax.legend(classes)
    plt.show()
