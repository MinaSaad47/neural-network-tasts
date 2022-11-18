import tkinter as tk
from tkinter import ttk


class Task(tk.Frame):
    def __init__(self, title):
        super().__init__()
        self.title = title
