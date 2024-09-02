import threading
import tkinter as tk
from PIL import Image, ImageTk

from java_api import HeadlessInterface


class TkinterUI():
    def __init__(self, game, width, height, verbose=False):
        self.interface = HeadlessInterface(game, width, height, verbose=verbose)
        self.img_path = self.interface.query(f"setup|{width}|{height}|{game}")
        self.img = None  # Initialize the image object

        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=width, height=height)
        self.canvas.pack()

        # Bind the click event
        self.canvas.bind("<Button-1>", self.on_click)

        self.update_image()  # Initial update call
        self.root.mainloop()

    def update_image(self):
        # Load and display the image in a separate thread to avoid blocking the main event loop
        threading.Thread(target=self.update).start()

    def update(self):
        img = Image.open(self.img_path)
        self.img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
        self.root.update_idletasks()  # Perform UI updates

    def on_click(self, event):
        self.canvas.config(cursor="watch")  # Change cursor to indicate processing
        x, y = event.x, event.y
        self.img_path = self.interface.query(f"click|{x}|{y}")
        self.update_image()
        self.canvas.config(cursor="")  # Reset cursor after the update


if __name__ == "__main__":
    tic_tac_toe = open("../ludii_data/games/expanded/board/space/line/Tic-Tac-Toe.lud", "r").read()
    ui = TkinterUI(tic_tac_toe, 1000, 500)
