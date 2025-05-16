import torch
from torch import nn
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
import matplotlib.animation as animation

# make sure device is defined
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    """
    class that defines our pytorch neural network. almost all other files import this
    so that I do not need to copy + paste this class into every file.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # neural network structure
        self.layer_stack = nn.Sequential(
            nn.Linear(2, 32), # 2 inputs (x (position), t (time))
            nn.SiLU(), # can't use ReLUs because that yields a 0 second derivative :( (i learned this the hard way)
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1), # 1 output
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layer_stack(x)
        return logits
    

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x, steps):
        batch_size, nx = x.shape
        h = torch.zeros(1, batch_size, self.gru.hidden_size, device=x.device)
        preds = []
        for _ in range(steps):
            x_input = x.unsqueeze(1)  # shape: [batch, 1, nx]
            out, h = self.gru(x_input, h)
            x = self.decoder(out.squeeze(1))
            preds.append(x)
        return torch.stack(preds)  # shape: [steps, batch, nx]


def getDevice():
    """
    used in place of gross
    "device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu""
    in all other files.
    """
    return device
    

def heatEquationSolution(x, t):
    """
    returns the well defined solution to the 1d heat equation at time t and position x
    """
    return (1.5*math.sin(math.pi*x)*math.e**(-1*(math.pi**2)*t)) - (0.5*math.sin(3*math.pi*x)*math.e**(-9*(math.pi**2)*t))


def modelToFxn(model):
    """
    returns a function that takes inputs (x,t) and returns the output of given
    model with those inputs.
    """
    if device == "mps":
        return (lambda x,t : float(model(torch.tensor([[x, t]], dtype=torch.float32,  device=device))))
    else:
        return (lambda x,t : float(model(torch.tensor([[x, t]], dtype=torch.float64,  device=device))))


def graph2D(fxn):
    """
    plots a function that takes two inputs (x,t) and gives one output at t=0, t=0.2, t=0.5, and t=1
    from x=(0,1)
    """
    STEPS = 100
    domain = [i/STEPS for i in range(0,STEPS+1)]
    t0codomain = [fxn(i,0) for i in domain]
    tpt2codomain = [fxn(i,0.2) for i in domain]
    tpt5codomain = [fxn(i,0.5) for i in domain]
    t1codomain = [fxn(i,1) for i in domain]
    plt.plot(domain, t0codomain)
    plt.plot(domain, tpt2codomain)
    plt.plot(domain, tpt5codomain)
    plt.plot(domain, t1codomain)
    plt.show()


def graph3D(fxn):
    """
    plots a function that takes two inputs (x,t) and gives one output in 3D space.
    Code adapted from https://matplotlib.org/stable/gallery/mplot3d/surface3d_3.html#sphx-glr-gallery-mplot3d-surface3d-3-py
    """
    ax = plt.figure().add_subplot(projection='3d')

    # Make data.
    STEPS = 50
    X = np.arange(0, 1, 1/STEPS)
    T = np.arange(0, 1, 1/STEPS)
    X, T = np.meshgrid(X, T)
    U = X+T # arbitrary array of correct size (we overwrite later anyway)
    for x in range(len(X)):
        for t in range(len(T)):
            xIn = float(X[0,x])
            tIn = float(T[t,0])
            U[x][t] = fxn(tIn, xIn)

    # Create an empty array of strings with the same shape as the meshgrid, and
    # populate it with two colors in a checkerboard pattern.
    colortuple = ('w', 'k')
    colors = np.empty(X.shape, dtype=str)
    for y in range(len(T)):
        for x in range(len(X)):
            colors[y, x] = colortuple[(y) % len(colortuple)]

    # Plot the surface with face colors taken from the array we made.
    ax.plot_surface(X, T, U, facecolors=colors, linewidth=0)

    # Customize the z axis.
    ax.set_zlim(-1, 1)
    ax.zaxis.set_major_locator(LinearLocator(6))

    plt.show()

def pdeLoss(inputs, model, alpha):
    inputs.requires_grad_(True)
    u = model(inputs)

    # Compute first degree gradients
    firstGradients = torch.autograd.grad(
        u,
        inputs, 
        grad_outputs=torch.ones_like(u),
        retain_graph=True, 
        create_graph=True
    )[0]

    # get first derivatives
    u_x = firstGradients[:, [0]]
    u_t = firstGradients[:, [1]]

    # get second derivative with respect to x
    u_xx = torch.autograd.grad(
        u_x, 
        inputs, 
        grad_outputs=torch.ones_like(u_x), 
        create_graph=True
    )[0][:, [0]]

    f = u_t - alpha * u_xx # PDE Residual
    
    return torch.mean(f**2)

def graphAnimated2D(fxn):
    # adapted from https://matplotlib.org/stable/gallery/animation/simple_anim.html

    x = np.linspace(0, 1) # x data

    fig, ax = plt.subplots()

    u = [fxn(i,0) for i in x]

    line, = ax.plot(x, u) # comma is python magic to unpack a list :)

    def animate(step):
        t = step / 250
        u = [fxn(i,t) for i in x]
        line.set_ydata(u)
        ax.set_title(f"Time t={t}")
        return line,

    ani = animation.FuncAnimation(
        fig, animate, interval=10, blit=False, save_count=50)

    plt.show()

def graphAnimated3D(fxn):
    """
    plots a function that takes three inputs (x,y,t) and gives one output in 3D space animated over time.
    Code adapted from https://matplotlib.org/stable/gallery/mplot3d/surface3d_3.html#sphx-glr-gallery-mplot3d-surface3d-3-py
    """
    ax = plt.figure().add_subplot(projection='3d')

    # Make data.
    STEPS = 50
    X = np.arange(0, 1, 1/STEPS)
    T = np.arange(0, 1, 1/STEPS)
    X, T = np.meshgrid(X, T)
    U = X+T # arbitrary array of correct size (we overwrite later anyway)
    for x in range(len(X)):
        for t in range(len(T)):
            xIn = float(X[0,x])
            tIn = float(T[t,0])
            U[x][t] = fxn(tIn, xIn)

    # Create an empty array of strings with the same shape as the meshgrid, and
    # populate it with two colors in a checkerboard pattern.
    colortuple = ('w', 'k')
    colors = np.empty(X.shape, dtype=str)
    for y in range(len(T)):
        for x in range(len(X)):
            colors[y, x] = colortuple[(y) % len(colortuple)]

    # Plot the surface with face colors taken from the array we made.
    ax.plot_surface(X, T, U, facecolors=colors, linewidth=0)

    # Customize the z axis.
    ax.set_zlim(-1, 1)
    ax.zaxis.set_major_locator(LinearLocator(6))

    plt.show()