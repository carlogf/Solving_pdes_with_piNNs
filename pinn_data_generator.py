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

    def __init__(self, xm = 0, xM = 1, t0 = 0, T = 1): #, number_of_points = 100):

        # We define our domain as $[-1,1] x [0,4]$
        #xm = -1
        #xM = 1
        #t0 = 0 # t siempre empieza en cero
        #T = 4

        self.xm = xm
        self.xM = xM
        self.t0 = t0
        self.T = T
        #self.number_of_points = number_of_points


        self.XT_init_train = None
        self.y_init_train = None

        self.XT_bc_train = None
        self.y_bc_train = None

        self.XT_coloc_train = None

        # (X, T)

        #Initial condition
    def generate_initial_condition_data(self, number_of_points, f_init = None):
        xm = self.xm
        xM = self.xM
        
        x_init_train = torch.linspace(xm,xM,number_of_points)
        y_init_train = np.exp(-x_init_train**2 )- (np.e**(-4))

        # Reshape x
        x_init_train = torch.reshape(x_init_train, (number_of_points, 1))
        
        # Define t = 0
        t_zeros = torch.zeros(number_of_points)
        t_zeros = torch.reshape(t_zeros, (number_of_points,1))
        
        # create xt vector
        XT_init_train = torch.cat((x_init_train,t_zeros),dim = 1)

        self.XT_init_train = XT_init_train
        
        # Reshape y
        y_init_train = torch.reshape(y_init_train, (number_of_points,1))
        self.y_init_train = y_init_train


        return XT_init_train.clone(), y_init_train.clone()


    #Boundary condition
    def generate_boundary_condition_data(self, number_of_points, eventually_more = None):
        t_bc= torch.linspace(self.t0, self.T, number_of_points)
        t_bc = torch.reshape(t_bc, (number_of_points,1))

        x_bc = torch.ones(number_of_points)
        x_bc = torch.reshape(x_bc,(number_of_points,1))

        t_bc1_train = torch.cat((self.xm * x_bc,t_bc), dim = 1)
        t_bc2_train = torch.cat((self.xM * x_bc, t_bc), dim = 1)

        #y_bc1_train = torch.ones((t_bc1_train.shape[0],1))
        #y_bc2_train = torch.ones((t_bc2_train.shape[0],1))

        #condiciones de contorno, hardcodeadas
        y_bc1_train = torch.zeros((t_bc1_train.shape[0],1))
        y_bc2_train = torch.zeros((t_bc2_train.shape[0],1))



        self.XT_bc_train = torch.cat((t_bc1_train, t_bc2_train), dim = 0)
        self.y_bc_train = torch.cat((y_bc1_train, y_bc2_train), dim = 0)

        return self.XT_bc_train.clone(), self.y_bc_train.clone()

    
    def generate_collocation_points_data(self, number_of_points):
        # Colocation points
        #number_of_points = 10000
        xm, xM = self.xm, self.xM
        t0, T = self.t0, self.T

        x_coloc_train = xm + (xM-xm)*torch.rand(number_of_points)
        t_coloc_train = t0 + (T-t0)*torch.rand(number_of_points)

        x_coloc_train = torch.reshape(x_coloc_train,(number_of_points,1))
        t_coloc_train = torch.reshape(t_coloc_train,(number_of_points,1))

        X_coloc_train = torch.cat((x_coloc_train,t_coloc_train), dim = 1)
        self.XT_coloc_train = X_coloc_train

        return X_coloc_train.clone()

    def plot_data(self):

        # Graficamos los puntos
        fig, ax = plt.subplots(figsize = (10,10))

        ax.set_xlabel("x")
        ax.set_ylabel("t")

        # Plot the domain
        xm, xM, t0, T = self.xm, self.xM, self.t0, self.T

        ax.plot([xm,xM],[t0,t0], color = 'blue', label= "Domain")
        ax.plot([xm,xm],[t0,T], color = 'blue')
        ax.plot([xM,xM],[t0,T], color = 'blue')
        ax.plot([xm,xM],[T,T], color = 'blue')

        #plot ic
        x_init_train = self.XT_init_train
        plt.scatter(x_init_train[:,0], x_init_train[:,1], label = "Initial condition")

        #plot bc
        t_bc_train = self.XT_bc_train
        plt.scatter(t_bc_train[:,0], t_bc_train[:,1], label = "Boundary condition", color = "orange")
        #plt.scatter(t_bc2_train[:,0], t_bc2_train[:,1], label = "Boundary condition", color = "orange")

        # Plot Collocation points
        X_coloc_train = self.XT_coloc_train
        plt.scatter(X_coloc_train[:,0], X_coloc_train[:,1], label = "Colocation points", color = "gray")

        ax.legend()



