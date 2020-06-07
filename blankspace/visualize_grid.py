import numpy as np
from tkinter import *
import random
from colour import Color
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matrix_gen as mg
import chunk_functions as cf

def nandifference(first: np.ndarray, last: np.ndarray):
    mean_f, mean_l = np.nanmean(first), np.nanmean(last)
    first = np.where(np.isnan(first), mean_f, first)
    last = np.where(np.isnan(last), mean_l, last)
    return last - first



class Cell():

    def __init__(self, master, x, y, size, matrix, color_range, max_difference, min_difference):
        """ Constructor of the object called by Cell(...) """
        self.master = master
        self.abs = x
        self.ord = y
        self.size= size
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
        if self.master != None :

            #calculate index in color range
            if np.isnan(self.matrix[-1, self.abs, self.ord]) or np.isnan(self.matrix[0, self.abs, self.ord]):
                fill = Color("black")
            else:
                difference = self.matrix[-1, self.abs, self.ord] - self.matrix[0, self.abs, self.ord]
                difference = int(round(difference * 100))
                index_range = difference + abs(self.min_difference)-1 if self.min_difference < 0 else difference-1

                #assign color
                fill = color_range[index_range]
            outline = "black"

            xmin = self.abs * self.size
            xmax = xmin + self.size
            ymin = self.ord * self.size
            ymax = ymin + self.size

            self.master.create_rectangle(xmin, ymin, xmax, ymax, fill = fill, outline = outline)

    def show_graphics(self):
        """show statistics of this cell"""
        if(self.clicked):

            window = Toplevel()
            window.title("Cell Trend")

            y = self.matrix[0:10,self.abs,self.ord]
            x = [f'Day {x+1}' for x in range(matrix.shape[0])]

            fig = plt.figure(figsize=(8,4))
            plt.plot(x, y, linewidth = 4)
            plt.xticks(rotation=45)

            canvas = FigureCanvasTkAgg(fig, master=window)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, ipadx=40, ipady=20)

        self.clicked = False

        

class CellGrid(Canvas):
    def __init__(self,master, rowNumber, columnNumber, cellSize, matrix, color_range, max_difference, min_difference, *args, **kwargs):
        Canvas.__init__(self, master, width = cellSize * columnNumber , height = cellSize * rowNumber, *args, **kwargs)

        self.cellSize = cellSize

        self.grid = []
        for row in range(rowNumber):

            line = []
            for column in range(columnNumber):
                line.append(Cell(self, column, row, cellSize, matrix, color_range, max_difference, min_difference))

            self.grid.append(line)

        #bind click action
        self.bind("<Button-1>", self.handleMouseClick)  
        #bind release button action - clear the memory of midified cells.
        #self.bind("<ButtonRelease-1>", lambda event: self.switched.clear())

        self.draw()


    def draw(self):
        for row in self.grid:
            for cell in row:
                cell.draw()

    def _eventCoords(self, event):
        row = int(event.y / self.cellSize)
        column = int(event.x / self.cellSize)
        return row, column

    def handleMouseClick(self, event):
        row, column = self._eventCoords(event)
        cell = self.grid[row][column]
        cell._switch()
        cell.show_graphics()
        

if __name__ == "__main__":
    app = Tk()

    # parameters
    n_row = 91
    n_col = 91
    resolution = 10

    # retrieve images
    matrix = mg.load_numpy_pkl()
    matrix = mg.generate_array_of_grids(matrix, cf.mean, n_row=n_row, n_col=n_col)

    # compute color scale
    diff = nandifference(matrix[0], matrix[-1])
    diff = np.round(diff*100).astype(int)
    max_difference = np.max(diff)
    min_difference = np.min(diff)
    color_range = list(Color("red").range_to(Color("blue"), max_difference - min_difference))

    # create grid
    grid = CellGrid(app, n_row, n_col, resolution, matrix, color_range, max_difference, min_difference)
    grid.pack()

    app.mainloop()
