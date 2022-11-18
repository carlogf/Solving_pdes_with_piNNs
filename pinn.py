
# Imports
import torch
import torch.nn as nn
import torch.autograd as autograd


class PINN(torch.nn.Module):

    def __init__(self, layers):
        
        super().__init__() # Instantiate a torch.nn

        "Layers"
        self.layers = layers

        "Activation function"
        self.activation = nn.Tanh() # Up until now it works with Tanh, we can change it to relu or something else

        "Loss Function"
        self.loss_function = nn.MSELoss(reduction = "mean") # This just sums the square of diffferences and divdes. Coudl ahve wrote it myself.

        "Linear layers"
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        # In the documentation it says we can set the weights and bias here, Camilo said the same thing, but I didn't do it

        "Xavier normal initialization"
        for i in range(len(layers)-2): 
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)


    "Forward Pass"
    def forward(self, x):
        # Check if x is NumPy Array or Tensor
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = x.float() #I dont understand why the .float()

        for i in range(len(self.layers)-2): # number of inner hidden layers
            z = self.linears[i](a)
            a = self.activation(z)
        
        # Last layer
        a = self.linears[-1](a)

        return a

    "Loss Functions"
    # En el sciann lo que hace es pasar una lista de loss functions y 
    # los puntos es una lista tambien que se corresponde

    # Por ahora lo hago igual que antes, creo que en el futuro lo que tengo que hacer
    # Es una unica loss function y hacerme cargo a mano de como le doy los datos.

    # La responsabilidad de la red es ser la red. La responsabilidad de la loss function es de la loss function

    # Dicho esto
    def loss_ic(self, X, y_ic_trues):
        # Toma tensores X de la forma (x,0), y evalua y compara
        loss_ic = self.loss_function(self.forward(X), y_ic_trues)
        return loss_ic


    def loss_bc(self, X, y_bc_trues):
        #toma tensores X de la forma (-1,t) y (1,t) y los pasa por la red y compara. Hace lo mismo que la anterior
        loss_bc = self.loss_function(self.forward(X), y_bc_trues)
        return loss_bc


    def residual_pde(self, xt):
        """
        Calculates the loss for the pde term for a single data entry (x,t)
        """
        # x, t --- >>> (x-R, x+R, 0.1) = int_noloc
        inp = xt.clone()
        inp.requires_grad = True

        # Jacobian and Hessian. Higher order derivatives must be done manually.
        Jf = autograd.functional.jacobian(self.forward, inp, create_graph = True)[0] # Returns matrix of 1x2
        Hf = autograd.functional.hessian(self.forward, inp)

        # Names
        fnn_t = Jf[1]
        fnn_xx = Hf[0,0]

        # Residual
        ret = fnn_t - fnn_xx #int_noloc
        return ret

    def loss_pde(self, X):
        #g = x.clone() #good practice
        #g.float().to(device)
        
        # Start recording grads on g:
        #g.requires_grad = True

        #fnn = self.forward(g)

        # Calculate derivatives for each data input

        n = X.shape[0]

        ret = torch.ones((n,1))

        for k in range(n):
            ret[k] = self.residual_pde(X[k])
        
        y_aux = torch.zeros(ret.shape)

        return self.loss_function(ret, y_aux)
        
    def loss (self, X_ic, y_ic, X_bc, y_bc, X_coloc):
        l_ic = self.loss_ic(X_ic, y_ic)
        l_bc = self.loss_bc(X_bc, y_bc)
        l_pde = self.loss_pde(X_coloc)

        return l_ic + l_bc + l_pde # aca hay lugar para poner parametros que escalen cada sumando. tipo lamda 1 2 y 3.


