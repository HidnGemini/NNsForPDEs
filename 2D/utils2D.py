import torch
from torch import nn
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

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
            nn.Linear(3, 32), # 3 inputs (x,y (positions), t (time))
            nn.SiLU(), # can't use ReLUs because that yields a 0 second derivative :( (i learned this the hard way)
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1), # 1 output
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layer_stack(x)
        return logits
    

class PeRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=3, hidden_size=32, batch_first=True)
        self.layer_stack = nn.Sequential(
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, input, hidden):
        out_rnn, hidden = self.rnn(input, hidden)
        out = self.layer_stack(out_rnn)
        return out, hidden

def getDevice():
    """
    used in place of gross
    "device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu""
    in all other files.
    """
    return device


def modelToFxn(model):
    """
    returns a function that takes inputs (x,y,t) and returns the output of given
    model with those inputs.
    """
    if device == "mps":
        return (lambda x,y,t : float(model(torch.tensor([[x, y, t]], dtype=torch.float32,  device=device))))
    else:
        return (lambda x,y,t : float(model(torch.tensor([[x, y, t]], dtype=torch.float64,  device=device))))


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
    u_y = firstGradients[:, [1]]
    u_t = firstGradients[:, [2]]

    # get second derivative with respect to x
    u_xx = torch.autograd.grad(
        u_x, 
        inputs, 
        grad_outputs=torch.ones_like(u_x), 
        create_graph=True
    )[0][:, [0]]

    u_yy = torch.autograd.grad(
        u_y, 
        inputs, 
        grad_outputs=torch.ones_like(u_y), 
        create_graph=True
    )[0][:, [1]]

    f = u_t - alpha * (u_xx+u_yy) # PDE Residual
    
    return torch.mean(f**2)


def graphAnimated3D(fxn):
    # Define the spatial domain
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y)

    # Create the figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initial surface plot
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i][j] = fxn(x[i], y[j], 0)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')

    # Set axes labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('u(x, y, t)')

    # Update function for animation
    def update(frame):
        t = frame*0.01
        ax.clear()
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i][j] = fxn(x[i], y[j], t)
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_zlim(0, 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('u(x, y, t)')
        ax.set_title(f"t = {t:.2f}")

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=100, interval=100)
    ani.save("u_animation.gif", writer=animation.PillowWriter(fps=20))

    plt.show()
