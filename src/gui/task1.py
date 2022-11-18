import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

import sys
import os
import random

from .core import Task
import models


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
        # preprocess
        self.dataset = models.task1.preprocess(self.dataset)

        features = self.dataset.columns[1:]
        feature1 = tk.StringVar(value=features[0])
        feature2 = tk.StringVar(value=features[1])

        def visualize_cb():
            f1 = feature1.get()
            f2 = feature2.get()
            if f1 == f2:
                messagebox.showerror(
                    title="duplicate class name", message='please choose different classes')
                return

            df = self.dataset.copy()
            models.task1.visualize(df, f1, f2)

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

        classes = self.dataset.iloc[:, 0].unique().tolist()

        # training
        class1 = tk.StringVar(value=classes[0])
        class2 = tk.StringVar(value=classes[1])
        learning_rate = tk.DoubleVar(value=0.0001)
        number_of_epocs = tk.IntVar(value=1000)
        bias = tk.BooleanVar(value=True)

        def train_cb():
            c1 = class1.get()
            c2 = class2.get()
            if c1 == c2:
                messagebox.showerror(
                    title="duplicate class name", message='please choose different classes')
                return
            f1 = feature1.get()
            f2 = feature2.get()
            if f1 == f2:
                messagebox.showerror(
                    title="duplicate feature name", message='please choose different features')
                return
            is_bias = bias.get()
            eta = learning_rate.get()
            nb_epochs = number_of_epocs.get()
            df = self.dataset.copy()
            accuracy = models.task1.train(
                df, f1, f2, c1, c2, eta, nb_epochs, is_bias)

            messagebox.showinfo(title="finished training",
                                message=f'accuracy: {accuracy * 100} %')

            # confusion matrix

        # gui
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

        cb = tk.Checkbutton(traning_frame, variable=bias, text='bias')
        cb.grid(row=4, column=0)

        btn = tk.Button(traning_frame, text='Train',
                        command=train_cb)
        btn.grid(row=5, column=0, columnspan=2)

        traning_frame.pack(fill='x')
