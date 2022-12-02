import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np

import sys
import os
import random

from .core import Task
import models
from models.classifiers import Adaline


class Task3(Task):
    def __init__(self):
        super().__init__('Task 3')

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

        self.user_input()

    # setep 3 user input
    def user_input(self):
        tkvar_layers = tk.StringVar(value='5,10,3')
        tkvar_lr = tk.DoubleVar(value=0.01)
        tkvar_epochs = tk.IntVar(value=1000)
        tkvar_usebias = tk.BooleanVar(value=True)
        tkvar_usetanh = tk.BooleanVar(value=False)

        def train_cb():
            df = self.dataset
            layers = list(map(int, tkvar_layers.get().split(',')))
            lr = tkvar_lr.get()
            epochs = tkvar_epochs.get()
            usebias = tkvar_usebias.get()
            usetanh = tkvar_usetanh.get()
            train_accuracy, test_accuracy = models.task3.train(
                df, layers, lr, epochs, usebias, usetanh)
        
            messagebox.showinfo(title='accuracies', message=f'train accuracy: {train_accuracy}\ntest_accuracy: {test_accuracy}')

        traning_frame = tk.Frame(
            self, highlightbackground='green', highlightthickness=2)

        tk.Label(traning_frame, text='layers (ie: 5,10,3)').grid(
            row=0, column=0)
        entry = tk.Entry(traning_frame, textvariable=tkvar_layers)
        entry.grid(row=0, column=1, sticky=tk.W)

        tk.Label(traning_frame, text='learing rate').grid(row=1, column=0)
        entry = tk.Entry(traning_frame, textvariable=tkvar_lr)
        entry.grid(row=1, column=1, sticky=tk.W)

        tk.Label(traning_frame, text='number of epocs').grid(row=2, column=0)
        entry = tk.Entry(traning_frame, textvariable=tkvar_epochs)
        entry.grid(row=2, column=1, sticky=tk.W)

        cb = tk.Checkbutton(traning_frame, variable=tkvar_usebias, text='bias')
        cb.grid(row=3, column=0)
        
        cb = tk.Checkbutton(traning_frame, variable=tkvar_usetanh, text='tanh')
        cb.grid(row=4, column=0)

        btn = tk.Button(traning_frame, text='Train',
                        command=train_cb)
        btn.grid(row=5, column=0, columnspan=2)

        traning_frame.pack(fill='x')
