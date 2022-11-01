import tkinter as tk
from tkinter import ttk

from .core import Task


class Application:
    def __init__(self, title: str, **kw):
        self.__root = tk.Tk()
        self.__root.title(title)
        if (kw['width'] and kw['height']):
            self.__root.geometry(f"{kw['width']}x{kw['height']}")

        self.__notebook = ttk.Notebook(self.__root)

    def tasks(self, tasks):
        if self.tasks:
            for task in tasks:
                self.__notebook.add(task, text=task.title)
            self.__notebook.pack(expand=True, fill='both')

    def run(self):
        self.__root.mainloop()
