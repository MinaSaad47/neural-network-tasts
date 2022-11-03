import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import sklearn.model_selection

from .core import Task
from ..utils.activition_functions import signum

import sys
import os
import random


class Task1(Task):
    def __init__(self):
        super().__init__('Task 1')

        # begin task 1
        self.file_path = tk.StringVar(value='Select Dataset Source')

        frame = tk.Frame(self)

        lb = tk.Label(frame, textvariable=self.file_path)
        lb.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        btn = tk.Button(frame, text='Browse', command=self.select_dataset)
        btn.grid(row=1, column=0)

        btn = tk.Button(frame, text='Load', command=self.load_dataset)
        btn.grid(row=1, column=1)

        frame.pack(fill='x')

    # step l: select dataset file
    def select_dataset(self):
        self.file_path.set(filedialog.askopenfilename(
            title='select csv file', filetypes=[('csv file', '*.csv'), ('all files', '*')]))

    # step 2: load dataset file to dataframe
    def load_dataset(self):
        try:
            csv_file = self.file_path.get()
            self.dataset = pd.read_csv(csv_file)
        except Exception as e:
            messagebox.showerror(title="coul't not load dataset", message=e)
            return

        # start step 3
        self.visualize_dataset()

    # step 3: visualize dataset
    def visualize_dataset(self):
        features = self.dataset.columns.values[1:]
        class_name = self.dataset.columns.values[0]
        classes = self.dataset[class_name].unique().tolist()
        print(f'features: {features}')

        # label encoding
        print('before encoding\n', self.dataset.head(n=10))
        self.dataset['gender'] = self.dataset['gender'].astype('category')
        self.dataset['gender'] = self.dataset['gender'].cat.codes
        print('ater encoding\n', self.dataset.head(n=10))

        # removing nans
        print('number of gender nan: ', self.dataset['gender'].isna().count())
        self.dataset['gender'][self.dataset['gender']
                               == -1] = self.dataset['gender'].max()
        print('number of gender nan: ', self.dataset['gender'].isna().count())
        print('ater encoding\n', self.dataset.head(n=10))

        feature1 = tk.StringVar(value=features[0])
        feature2 = tk.StringVar(value=features[1])

        def visualize_cb():
            f1 = feature1.get()
            f2 = feature2.get()
            plt.figure(f'{f1} against {f2}')
            for cls in classes:
                cls = self.dataset[self.dataset[class_name] == cls]
                plt.scatter(cls[f1], cls[f2])
            plt.xlabel(f1)
            plt.ylabel(f2)
            plt.legend(classes)
            plt.show()

        visualize_frame = tk.Frame(
            self, highlightbackground='red', highlightthickness=2)

        tk.Label(visualize_frame, text='feature 1').grid(row=0, column=0)
        drop = tk.OptionMenu(visualize_frame, feature1,
                             *features)
        drop.grid(row=0, column=1, sticky=tk.W)

        tk.Label(visualize_frame, text='feature 2').grid(row=1, column=0)
        drop = tk.OptionMenu(visualize_frame, feature2,
                             *features)
        drop.grid(row=1, column=1, sticky=tk.W)

        btn = tk.Button(visualize_frame, text='Visualize',
                        command=visualize_cb)
        btn.grid(row=2, column=0, columnspan=2)

        visualize_frame.pack(fill='x')

        # training
        class1 = tk.StringVar(value=classes[0])
        class2 = tk.StringVar(value=classes[1])
        learning_rate = tk.DoubleVar()
        number_of_epocs = tk.IntVar()
        bios = tk.BooleanVar(value=False)

        def train_cb():
            f1 = feature1.get()
            f2 = feature2.get()

            df = self.dataset[[class_name, f1, f2]]

            c1_df = df[df[class_name] == class1.get()]
            c1_x = c1_df[[f1, f2]]
            c1_y = c1_df[class_name]
            c1_x_train, c1_x_test, c1_y_train, c1_y_test = model_selection.train_test_split(
                c1_x, c1_y, test_size=0.4)

            c2_df = df[df[class_name] == class2.get()]
            c2_x = c2_df[[f1, f2]]
            c2_y = c2_df[class_name]
            c2_x_train, c2_x_test, c2_y_train, c2_y_test = model_selection.train_test_split(
                c2_x, c2_y, test_size=0.4)

            X = pd.concat([c1_x_train, c2_x_train])
            Y = pd.concat([c1_y_train, c2_y_train])

            X_Test = pd.concat([c1_x_test, c2_x_test])
            Y_Test = pd.concat([c1_y_test, c2_y_test])

            lr = learning_rate.get()
            noe = number_of_epocs.get()
            b = bios.get()

            W = np.random.normal(0, 0.01, size=(2, ))
            B = random.random()

            for epocs in noe:
                for i, x in enumerate(X):
                    net = np.dot(x, W)
                    if b:
                        net += B
                    y_pred = signum(net)
                    loss = Y - y_pred
                    W= W + lr * (loss)

                    pass

            print('class 1:\n', c1)
            print('class 2:\n', c2)

        traning_frame = tk.Frame(
            self, highlightbackground='green', highlightthickness=2)

        tk.Label(traning_frame, text='class 1').grid(row=0, column=0)
        drop = tk.OptionMenu(traning_frame, class1, *classes)
        drop.grid(row=0, column=1, sticky=tk.W)

        tk.Label(traning_frame, text='class 2').grid(row=1, column=0)
        drop = tk.OptionMenu(traning_frame, class2,
                             *classes)
        drop.grid(row=1, column=1, sticky=tk.W)

        tk.Label(traning_frame, text='learing rate').grid(row=2, column=0)
        entry = tk.Entry(traning_frame, textvariable=learning_rate)
        entry.grid(row=2, column=1, sticky=tk.W)

        tk.Label(traning_frame, text='number of epocs').grid(row=3, column=0)
        entry = tk.Entry(traning_frame, textvariable=number_of_epocs)
        entry.grid(row=3, column=1, sticky=tk.W)

        cb = tk.Checkbutton(traning_frame, variable=bios, text='bios')
        cb.grid(row=4, column=0)

        btn = tk.Button(traning_frame, text='Train',
                        command=train_cb)
        btn.grid(row=5, column=0, columnspan=2)

        traning_frame.pack(fill='x')
