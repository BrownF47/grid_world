import numpy as np
import matplotlib.pyplot as plt

class SPACE():

    def __init__(self, x, y, type='empty'):
        self.x = x
        self.y = y
        self.type = type
        if self.type == 'empty':
            self.space_color = 'black'
        elif self.type == 'mine':
            self.space_color = 'red'
        elif self.type == 'gate':
            self.space_color = 'blue'
    
    def print_space(self):
        print(f"Space: ({self.x}, {self.y}) , {self.type}")

    def scatter_space(self):
        plt.scatter(self.x, self.y, marker='s', facecolors='none', edgecolors=self.space_color, s=1000)
        


class GRIDWORLD():

    def __init__(self, size: tuple):
        self.size = size
        self.space_list = []

        for i in range(size[0]):
            for j in range(size[1]):
                self.space_list.append(SPACE(i, j))
                
    def list_spaces(self):
        for space in self.space_list:
            space.print_space()

    def visualise_spaces(self):
        for space in self.space_list:
            space.scatter_space()
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim((-1,self.size[0]))
        plt.ylim((-1,self.size[1]))
        plt.show()

myGrid = GRIDWORLD((3,5))
myGrid.list_spaces()
myGrid.visualise_spaces()