import numpy as np
import matplotlib.pyplot as plt
import random

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

    def __eq__(self, other):
        return isinstance(other, SPACE) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))
    
    def print_space(self):
        print(f"Space: ({self.x}, {self.y}) , {self.type}")

    def scatter_space(self):
        plt.scatter(self.x, self.y, marker='s', facecolors='none', edgecolors=self.space_color, s=1000)
        

class AGENT(): 

    def __init__(self, x, y, type='random'):
        self.x = x
        self.y = y
        self.agent_color = 'green'
        self.type = type
        self.valid_moves = [(1,0),(0,1),(-1,0),(0,-1)]

    def print_agent(self):
        print(f"Agent: ({self.x}, {self.y})")
        
    def scatter_agent(self):
        plt.scatter(self.x, self.y, marker='o', color=self.agent_color, s=200)
    
    def choose_move(self):
        if self.type == 'random':
            return self.choose_random_move()
    
    def choose_random_move(self):
        move = random.choice(list(self.valid_moves))
        return move




class GRIDWORLD():

    def __init__(self, size: tuple):
        self.size = size
        self.spaces = set()
        self.end = False

        for i in range(size[0]):
            for j in range(size[1]):
                self.spaces.add(SPACE(i, j))
        
        self.agent_list = [AGENT(0,0)]
                
    def list_spaces(self):
        for space in self.spaces:
            space.print_space()
    
    def list_agents(self):
        for agent in self.agent_list:
            agent.print_agent()

    def visualise_grid(self):
        for space in self.spaces:
            space.scatter_space()
        for agent in self.agent_list:
            agent.scatter_agent()
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim((-1,self.size[0]))
        plt.ylim((-1,self.size[1]))
        plt.show()
    
    def move_agent(self, agent: AGENT,  move: tuple):
        new_x = agent.x + move[0]
        new_y = agent.y + move[1]
        new_space = SPACE(new_x, new_y) 
        if new_space in self.spaces:
            agent.x = new_x
            agent.y = new_y
            if new_space == SPACE(1,1):
                self.end = True

        
myGrid = GRIDWORLD((2,2))
myGrid.list_spaces()
myGrid.list_agents()
myGrid.visualise_grid()
while not myGrid.end:
    myGrid.move_agent(myGrid.agent_list[0], myGrid.agent_list[0].choose_move())
    myGrid.visualise_grid()