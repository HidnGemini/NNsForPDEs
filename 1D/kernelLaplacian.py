import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from utils1D import RNN as HeatRNN
import utils1D as utils

def physics_loss(u_pred, alpha, dx, dt):
    u_t = (u_pred[1:] - u_pred[:-1])[:,0,:] / dt
    u_x = (u_pred[:, 0, 1:] - u_pred[:, 0, :-1]) / dx
    u_xx = (u_x[:,1:] - u_x[:,:-1]) / dx
    residual = u_t[:,:-2] - alpha * u_xx[:-1]

    return nn.MSELoss()(residual, torch.zeros_like(residual))

def finite_diff(u, kernel, dx):
    # kernel: shape [1, 1, k]
    return nn.functional.conv1d(u.unsqueeze(1), kernel, padding=0) / dx

model_file = "firstPeRCNN.pth"
model = torch.load(model_file, weights_only=False, map_location=torch.device("cpu"))

nx = 50
x = torch.linspace(0, 1, nx).unsqueeze(0)  # shape: [1, nx]
u0 = torch.sin(torch.pi * x)  # initial temperature profile

u_pred = model(u0, 2)

dx = 0.01
dt = 0.01

k_x = torch.tensor([[-1, 1]], dtype=u_pred.dtype, device=u_pred.device).view(1, 1, 2)
k_xx = torch.tensor([[1, -2, 1]], dtype=u_pred.dtype, device=u_pred.device).view(1, 1, -1)
k_t = torch.tensor([[-1, 1]], dtype=u_pred.dtype, device=u_pred.device).view(1, 1, 2)

u_pred_squeezed = u_pred.squeeze(1)
ku_x = finite_diff(u_pred_squeezed, k_x, dx)
ku_xx = finite_diff(u_pred_squeezed, k_xx, dx**2)
ku_t = finite_diff(u_pred_squeezed.permute(1, 0), k_t, dt).squeeze(1).permute(1, 0)

u_x = (u_pred[:, 0, 1:] - u_pred[:, 0, :-1]) / dx
u_xx = (u_x[:,1:] - u_x[:,:-1]) / dx
u_t = (u_pred[1:] - u_pred[:-1])[:,0,:] / dt

# u_time = u_pred.permute(2, 1, 0)  # now shape: [nx, 1, nt]

print(f'u_t shape: {u_t.shape}')
print(u_t)
print(f'ku_t shape: {ku_t.shape}')
print(ku_t)

print(u_pred.permute(2, 1, 0)[:, 0, :].shape)
print(u_pred.squeeze(1).permute(1, 0).shape)