import tkinter as tk
from tkinter import *

from .core import Task


class Task1(Task):
    def __init__(self):
        super().__init__('Task 1')
        tk.Label(self, text='hello world').pack()
