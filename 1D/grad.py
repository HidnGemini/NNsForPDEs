import os
import torch
from torch import nn

from utils1D import NeuralNetwork
import utils1D

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


model = torch.load('heatEqSiLU.pth', weights_only=False)

x = 0.8
t = 0

inp = torch.tensor([[x, t]], requires_grad=True, device=device)

out = model(inp)

# First derivative: dy/dx2
firstPartials = torch.autograd.grad(
    outputs=out,
    inputs=inp,
    grad_outputs=torch.ones_like(out),  # ∂y/∂y = 1
    create_graph=True,
    retain_graph=True,
    materialize_grads=True
)[0]  # dy/dx is shape [2], corresponding to x1 and x2

du_dx = firstPartials[0,0]
du_dt = firstPartials[0,1]

# Second derivative: d²y/dx2²
d2u_dx2 = torch.autograd.grad(
    outputs=du_dx,
    inputs=inp,
    grad_outputs=torch.ones_like(du_dx),
    create_graph=True,
    retain_graph=True,
    materialize_grads=True
)[0][0,0]

du_dx.detach()
du_dt.detach()
d2u_dx2.detach()

print(du_dx)
print(du_dt)
print(d2u_dx2)

utils1D.graph2D(utils1D.modelToFxn(model))