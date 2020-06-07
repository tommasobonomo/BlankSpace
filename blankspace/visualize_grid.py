import os
import numpy as np
from tkinter import Toplevel, Canvas, Tk
from colour import Color
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import blankspace.matrix_gen as mg
import blankspace.chunk_functions as cf

from blankspace.utils import get_image_collection


def nandifference(first: np.ndarray, last: np.ndarray):
    mean_f, mean_l = np.nanmean(first), np.nanmean(last)
    first = np.where(np.isnan(first), mean_f, first)
    last = np.where(np.isnan(last), mean_l, last)
    return last - first


class Cell:
    def __init__(
        self,
        master,
        x,
        y,
        row_size,
        col_size,
        matrix,
        color_range,
        max_difference,
        min_difference,
        range_max,
        range_min,
    ):
        """ Constructor of the object called by Cell(...) """
        self.master = master
        self.abs = x
        self.ord = y
        self.row_size = row_size
        self.col_size = col_size
        self.range_max = range_max
        self.range_min = range_min
        self.clicked = False
        self.matrix = matrix
        self.color_range = color_range
        self.max_difference = max_difference
        self.min_difference = min_difference

    def _switch(self):
        """ Switch if the cell has been clicked or not. """
        self.clicked = not self.clicked

    def draw(self):
        """ order to the cell to draw its representation on the canvas """
        if self.master is not None:

            # calculate index in color range
            if np.isnan(self.matrix[-1, self.abs, self.ord]) or np.isnan(
                self.matrix[0, self.abs, self.ord]
            ):
                fill = Color("black")
            else:
                mat_difference = (
                    self.matrix[-1, self.abs, self.ord]
                    - self.matrix[0, self.abs, self.ord]
                )
                difference = int(round(mat_difference * 100))
                scaled_difference = (difference - self.min_difference) / (
                    self.max_difference - self.min_difference
                )
                index_range = int(round(scaled_difference * (len(color_range) - 1)))

                # assign color
                fill = color_range[index_range]
            outline = "black"

            xmin = self.abs * self.col_size
            xmax = xmin + self.col_size
            ymin = self.ord * self.row_size
            ymax = ymin + self.row_size

            self.master.create_rectangle(
                xmin, ymin, xmax, ymax, fill=fill, outline=outline
            )

    def show_graphics(self):
        """show statistics of this cell"""
        if self.clicked:

            window = Toplevel()
            window.title("Cell Trend")

            y = self.matrix[:, self.abs, self.ord]
            x = [f"Day {x+1}" for x in range(matrix.shape[0])]

            fig = plt.figure(figsize=(8, 4))
            plt.plot(x, y, linewidth=4)
            plt.ylim((self.range_min, self.range_max))
            plt.xticks(rotation=45)

            canvas = FigureCanvasTkAgg(fig, master=window)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, ipadx=40, ipady=20)

        self.clicked = False


class CellGrid(Canvas):
    def __init__(
        self,
        master,
        rowNumber,
        columnNumber,
        row_size,
        col_size,
        matrix,
        color_range,
        max_difference,
        min_difference,
        *args,
        **kwargs,
    ):
        Canvas.__init__(
            self,
            master,
            width=col_size * columnNumber,
            height=row_size * rowNumber,
            *args,
            **kwargs,
        )

        range_max = np.nanmax(matrix)
        range_min = np.nanmin(matrix)

        self.row_size = row_size
        self.col_size = col_size

        self.grid = []
        for row in range(rowNumber):

            line = []
            for column in range(columnNumber):
                line.append(
                    Cell(
                        self,
                        column,
                        row,
                        row_size,
                        col_size,
                        matrix,
                        color_range,
                        max_difference,
                        min_difference,
                        range_max,
                        range_min,
                    )
                )

            self.grid.append(line)

        # bind click action
        self.bind("<Button-1>", self.handleMouseClick)
        # bind release button action - clear the memory of midified cells.
        # self.bind("<ButtonRelease-1>", lambda event: self.switched.clear())

        self.draw()

    def draw(self):
        for row in self.grid:
            for cell in row:
                cell.draw()

    def _eventCoords(self, event):
        row = int(event.y / self.row_size)
        column = int(event.x / self.col_size)
        return row, column

    def handleMouseClick(self, event):
        row, column = self._eventCoords(event)
        cell = self.grid[row][column]
        cell._switch()
        cell.show_graphics()


if __name__ == "__main__":
    app = Tk()

    # parameters
    n_row = 120
    n_col = 120
    resolution = 1.7  # scales the dimensions of the rectangles (more rows and cols require higher resolution)

    # retrieve images
    base_path = os.path.join("data", "Coastal-InSAR-two-years")
    try:
        img_paths = [
            os.path.join(base_path, img_path) for img_path in os.listdir(base_path)
        ]
        matrix = np.array(get_image_collection(img_paths, collate=False))
    except ModuleNotFoundError:
        matrix = mg.load_numpy_pkl(base_path)

    matrix, row_size, col_size = mg.generate_array_of_grids(
        matrix, cf.mean, n_row=n_row, n_col=n_col
    )
    row_size, col_size = row_size * resolution, col_size * resolution

    # compute color scale
    diff = nandifference(matrix[0], matrix[-1])
    diff = np.round(diff * 100).astype(int)
    max_difference = np.max(diff)
    min_difference = np.min(diff)
    half_color_range = (max_difference - min_difference) // 2
    color_range = list(Color("red").range_to(Color("white"), half_color_range)) + list(
        Color("white").range_to(
            Color("blue"), (max_difference - min_difference) - half_color_range
        ),
    )

    # create grid
    grid = CellGrid(
        app,
        n_row,
        n_col,
        row_size,
        col_size,
        matrix,
        color_range,
        max_difference,
        min_difference,
    )
    grid.pack()

    app.mainloop()
