#!/usr/bin/env python3.5

import time
import numpy as np
import matplotlib.pyplot as plt

class Game:
    def __init__(self):
        self.nrows = 100
        self.ncols = 100

        #np.random.seed(0)
        self.ker = self.init_kernel()
        self.gen = self.init_random_gen()

        self.generation = 0
        self.lifespan = 1000

        # figure configuration
        plt.ion()
        self.fig = plt.figure()
        self.fig_ax = plt.gca()
        self.fig_ax.axes.xaxis.set_ticks([])
        self.fig_ax.axes.yaxis.set_ticks([])
        self.frame = plt.imshow(self.gen[1:self.nrows,1:self.ncols], cmap='gist_gray_r', vmin=0,vmax=1)
        self.fig.canvas.flush_events()

        self.start()

    def start(self):
        """Start simulation"""
        #mean = 0
        while self.generation<self.lifespan:
            t0 = time.time()
            self.update_gen_values() # update the grid values
            self.update_gen_frame() # plot the grid without the padding
            self.generation += 1
            #mean = (mean+(time.time()-t0)/self.generation)/(1+1/self.generation)
            #print(time.time()-t0)

    def init_random_gen(self):
        """Generates a grid with padding to perform the kernel convolution"""
        grid = np.zeros((self.nrows+2,self.ncols+2)) # add padding
        grid[1:self.nrows+1,1:self.ncols+1] = np.random.randint(0,2,(self.nrows,self.ncols)) # randomize array
        return grid

    def init_kernel(self):
        """Define a kernel matrix to apply in the convolution process"""
        ker = np.ones((3,3))
        ker[1,1] = 0
        return ker

    def update_gen_values(self):
        """ Change to 1 if surrounded by 2or3 1, change to 0 if lessthan3 1"""
        new_gen = np.zeros((self.nrows+2,self.ncols+2)) # add padding
        # convolution
        for i in range(1,self.nrows):
            for j in range(1,self.ncols):
                new_gen[i,j] = self.question_life(i,j) # apply kernel

        self.gen = new_gen

    def update_gen_frame(self):
        self.frame.set_data(self.gen[1:self.nrows,1:self.ncols])
        self.fig.canvas.flush_events()

    def question_life(self, row, col):
        """Figure out whether to change or to maintain the same value of the
        cell in [row,col]"""
        neighbourhood = self.get_neighbourhood(row, col)
        life_condition = self.apply_kernel(neighbourhood)

        return self.gen[row,col]*(2<=life_condition<=3) + (self.gen[row,col]==0)*(life_condition==3)

    def get_neighbourhood(self, row, col):
        """Get the neighbourhood around the element in [row,col] of the gen matrix"""
        return self.gen[row-1:row+2,col-1:col+2]

    def apply_kernel(self, neighbourhood):
        """Multiply the kernel to te neighbourhood matrix and return sum"""
        return sum(sum(np.multiply(neighbourhood,self.ker)))

Conway = Game()
