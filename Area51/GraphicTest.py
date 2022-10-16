# Create game tic tac toe with graphic interface

import sys
import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Tic Tac Toe")
        self.root.resizable(0, 0)
        self.root.protocol("WM_DELETE_WINDOW", self.exit)

        self.frame = tk.Frame(self.root)
        self.frame.pack()

        self.canvas = tk.Canvas(self.frame, width=300, height=300, bg="white")
        self.canvas.pack()

        self.canvas.create_line(100, 0, 100, 300, width=3)
        self.canvas.create_line(200, 0, 200, 300, width=3)
        self.canvas.create_line(0, 100, 300, 100, width=3)
        self.canvas.create_line(0, 200, 300, 200, width=3)

        self.canvas.bind("<Button-1>", self.click)

        self.root.mainloop()

    def click(self, event):
        x = event.x
        y = event.y
        print(x, y)

    def exit(self):
        self.root.destroy()
        sys.exit()

if __name__ == "__main__":
    TicTacToe()