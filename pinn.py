
# Copiar de aca abajo todo lo que sirva, pero adaptandolo a una PDE con dos coordenadas.
# 3 coordenadas (dos espaciales y una temporal) queda para el futuro.


# We have the data, we will now define our neural network using the torch.nn class

class PINN(torch.nn.Module): # Defino mi clase Pinn como una heredera de la neural network de pytorch
    def __init__(self, layers):
        super().__init__() #instanciamos una torch.nn 

        #Defino atributos de mi red
        "layers"
        self.layers = layers
        
        "activation function"
        self.activation = nn.Tanh() # las pinns necesitan una funcion de activacion especial? >Probar con otras activations a ver si va mejro Relu, miche > tipo relu

        "Loss Function"
        self.loss_function = nn.MSELoss(reduction = "mean") #

        # Instancio las funciones lineales de mi red en una lista
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]) # Puedo apsarle xavier initizaliation a la nn.Linears

        #self.iter = 0 # Esto lo define porque usa otro optimizador en vez del adam. Por eso define la closure que tampoco voy a usar.

        "Xavier Normal initialization" # Inicializamos los pesos usando este metodo que se llama asi. Hace unas normales en un cierto rango. #cajanegra
        for i in range(len(layers)-2): # No entiendo pq no setea pesos en la ultima funcion
            nn.init.xavier_normal_(self.linears[i].weight.data, gain = 1.0) #gain=1 es default. #nn.init es una clase con un monton de metodos(funciones)

            nn.init.zeros_(self.linears[i].bias.data)
        
    "Forward pass" #Evaluo mi red en un punto
    def forward(self,x):
        if not torch.is_tensor(x):
            #x = torch.Tensor(x)
            x = torch.from_numpy(x)
        a = x.float()

        for i in range(len(self.layers)-2): #todas menos la ultima
            z = self.linears[i](a)
            a = self.activation(z)
        
        a = self.linears[-1](a) #Evaluo la ultima para devolver
        return a

    "loss functions"
    
    #"initial condition"
    def loss_ic(self,x,y_true):
        loss_ic = self.loss_function(self.forward(x), y_true) #paso x por la red, y lo comparo con el y verdadero [(xk,0) ]
        return loss_ic

    # PDE
    def loss_pde(self,x): #en realidad es una ODE pero se entiende.
        g = x.clone() #clonamos los datos para evitar tocar los gradientes de x #capaz puedo hacer esto para cuando agregue el termino no local
        g.float().to(device)
        g.requires_grad=True #lo preparo para ser derivado

        fnn = self.forward(g)

        fnn_x = autograd.grad(fnn, g, torch.ones([g.shape[0],1]).to(device), retain_graph=True) #no entiendo que hace el grad outputs ni el retain_graph

        
        pde = fnn_x[0] - (fnn * (1- fnn))
        #print(pde.get_device())
        y_trues = torch.zeros(pde.shape) #defino zeros pq quiero que la pde sea lo mas chica posible.
        y_trues=y_trues.float().to(device)
        #print(y_trues)
        return self.loss_function(pde, y_trues)

    def loss(self, x_ic, y_ic, x_coloc):
        l1 = self.loss_ic (x_ic,y_ic)
        l2 = self.loss_pde(x_coloc)
        return l1+l2 #aca hay lugar para poner un gamma multiplicando una de las dos y regular la intensidad de la condicion inicial vs la pde.


    #Aca el definia una funcion enclosure pero creo que para adam no aplica y para sgd tampoco.
