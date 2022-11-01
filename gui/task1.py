import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pylab as plt

from .core import Task

import sys
import os


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
        print(f'features: {features}')

        print(self.dataset.head())
        
        feature1 = tk.StringVar(value=features[0])
        feature2 = tk.StringVar(value=features[1])
        
        def visualize_cb():
            f1 = feature1.get()
            f2 = feature2.get()
            plt.figure(f'{f1} against {f2}')
            plt.scatter(self.dataset[f1], self.dataset[class_name])
            plt.scatter(self.dataset[f2], self.dataset[class_name])
            plt.xlabel('features')
            plt.ylabel(f'{class_name}')
            plt.show()

        frame = tk.Frame(self)

        tk.Label(frame, text='feature 1').grid(row=0, column=0)
        drop = tk.OptionMenu(frame, feature1, *self.dataset.columns.values)
        drop.grid(row=0, column=1, sticky=tk.W)

        tk.Label(frame, text='feature 2').grid(row=1, column=0)
        drop = tk.OptionMenu(frame, feature2, *self.dataset.columns.values)
        drop.grid(row=1, column=1, sticky=tk.W)

        btn = tk.Button(frame, text='Visualize', command=visualize_cb)
        btn.grid(row=3, column=0, columnspan=2)

        frame.pack(fill='x')
