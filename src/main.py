import sys

from gui import Application, Task1, Task2


def main(argv):
    app = Application('neural network tasks', width=800, height=600)
    app.tasks([Task2(), Task1()])
    app.run()


main(sys.argv)
