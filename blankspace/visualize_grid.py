import numpy as np
from tkinter import *
import random
from colour import Color
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
            difference = self.matrix[0][self.abs][self.ord] - self.matrix[len(self.matrix)-1][self.abs][self.ord]
            index_range = difference - min_difference


            print(index_range)
            print(len(self.color_range))
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
            window.title("Vegetation Trend")

            y = self.matrix[0:10,self.abs,self.ord]
            x = ['Col A', 'Col B', 'Col C', 'Col D', 'Col E', 'Col F', 'Col G', 'Col H', 'Col I', 'Col L']

            fig = plt.figure(figsize=(5,3))
            plt.bar(x=x, height=y)

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

    #matrix
    matrix = np.array([[[random.randint(0,100) for i in range (20)] for i in range (20)] for i in range (10)])

    diff = matrix[9] - matrix[0]
    max_difference = np.max(diff)
    min_difference = np.min(diff)

    color_range = list(Color("blue").range_to(Color("red"),abs(max_difference-min_difference)+1))

    grid = CellGrid(app, 20, 20, 20, matrix, color_range, max_difference, min_difference)
    grid.pack()

    app.mainloop()
