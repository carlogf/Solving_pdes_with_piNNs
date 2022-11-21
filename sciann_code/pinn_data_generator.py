# Class that generates training and validation data in the format of tensor 
# So that it can be used with a pytorch neural network function that solves a pde.
# This is inspired by the sciann data generator class

import torch
import numpy as np
import matplotlib.pyplot as plt

class dataGeneratorXT:

    """
    Generates a rectangular domain in the plane with
    * Initial condition
    * Boundary conditions
    * Colocation points
    """

    def __init__(self, xm = 0, xM = 1, t0 = 0, T = 1, number_of_points = 100):

        # We define our domain as $[-1,1] x [0,4]$
        #xm = -1
        #xM = 1
        #t0 = 0 # t siempre empieza en cero
        #T = 4

        self.xm = xm
        self.xM = xM
        self.t0 = t0
        self.T = T
        self.number_of_points = number_of_points

        # (X, T)

        #Initial condition
        x_init_train = torch.linspace(xm,xM,number_of_points)
        y_init_train = np.exp(-x_init_train**2 )- 1/np.e

        # Reshape x
        x_init_train = torch.reshape(x_init_train, (number_of_points, 1))
        t_zeros = torch.zeros(number_of_points)
        t_zeros = torch.reshape(t_zeros, (number_of_points,1))
        x_init_train = torch.cat((x_init_train,t_zeros),dim = 1)

        # Reshape y
        y_init_train = torch.reshape(y_init_train, (number_of_points,1))

        #Boundary condition
        t_bc= torch.linspace(t0,T,number_of_points)
        t_bc = torch.reshape(t_bc, (number_of_points,1))

        x_bc = torch.ones(number_of_points)
        x_bc = torch.reshape(x_bc,(number_of_points,1))

        t_bc1_train = torch.cat((-1*x_bc,t_bc), dim = 1)
        t_bc2_train = torch.cat((x_bc, t_bc), dim = 1)

        #y_bc1_train = torch.ones((t_bc1_train.shape[0],1))
        #y_bc2_train = torch.ones((t_bc2_train.shape[0],1))
        y_bc1_train = torch.zeros((t_bc1_train.shape[0],1))
        y_bc2_train = torch.zeros((t_bc2_train.shape[0],1))



        t_bc_train = torch.cat((t_bc1_train, t_bc2_train), dim = 0)
        y_bc_train = torch.cat((y_bc1_train, y_bc2_train), dim = 0)

