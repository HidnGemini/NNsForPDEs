import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from utils1D import RNN
import utils1D as utils

# ---- Parameters ----
alpha = 0.01  # thermal diffusivity
nx = 50       # number of spatial points
nt = 1000      # number of time steps
dx = 1.0 / nx
dt = 0.01

# device = utils.getDevice()
device = "cpu:0"

# ---- Synthetic Initial Condition ----
x = torch.linspace(0, 1, nx) # shape: [1, nx]
u0sin = torch.sin(torch.pi * x) # initial temperature profile in sin shape
# u0 = torch.stack((u0sin,))
u0 = u0sin.unsqueeze(0)
batches, _ = u0.shape
u_true = u0.clone()  # Save initial

# ---- Generate ground truth using finite difference ----
def generate_true_solution(u0, nt, alpha, dx, dt):
    u = u0.clone()
    result = [u]
    for _ in range(nt-1):
        u_next = u.clone()
        u_next[:, 1:-1] = u[:, 1:-1] + alpha * dt / dx**2 * (
            u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]
        )
        result.append(u_next)
        u = u_next
    return torch.stack(result, dim=0)  # shape: [nt+1, 1, nx]

u_data = generate_true_solution(u0, nt, alpha, dx, dt).to(device).detach()

# ---- Physics loss (finite-difference approximation of heat equation) ----
def physics_loss(u_pred, alpha, dx, dt):
    residualsList = []
    for i in range(batches):
        u_t = (u_pred[1:] - u_pred[:-1])[:,i,:] / dt
        u_x = (u_pred[:, i, 1:] - u_pred[:, i, :-1]) / dx
        u_xx = (u_x[:,1:] - u_x[:,:-1]) / dx
        residualsList.append(torch.abs(u_t[:,:-2] - alpha * u_xx[:-1]))

    residuals = torch.stack(residualsList)
    return nn.MSELoss()(residuals, torch.zeros_like(residuals))


# ---- Training ----
model = RNN(input_size=nx, hidden_size=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fxn = nn.MSELoss()

num_epochs = 20000
print_freq = 50
u_pred = model(u0, nt)
k_xx = torch.tensor([[1, -2, 1]], dtype=u_pred.dtype, device=device).view(1, 1, -1)
k_t = torch.tensor([[-1, 1]], dtype=u_pred.dtype, device=device).view(1, 1, 2)
start_time = time.time()

for epoch in range(num_epochs):
    optimizer.zero_grad()
    u_pred = model(u0, nt)  # [nt, 1, nx]
    loss_data_ic = loss_fxn(u_pred, u_data)
    loss_phys = physics_loss(u_pred, alpha, dx, dt)
    loss = loss_data_ic + 10*loss_phys
    loss.backward()
    optimizer.step()
    
    if epoch % print_freq == 0:
        elapsed = time.time() - start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        print(f"Epoch {epoch} / {num_epochs} @ {mins}m {secs:2.0f}s - Data Loss: {loss_data_ic.item():.4f}, Physics Loss: {loss_phys.item():.4f}")

torch.save(model, "PeRCNN2.pth")

print(f"Final Epoch, Data Loss: {loss_data_ic.item():.4f}, Physics Loss: {loss_phys.item():.4f}")
u_pred = model(u0, nt)

# ---- Prediction example ----
plt.imshow(torch.Tensor.cpu(u_pred.squeeze().detach()).numpy(), aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Temperature over Time")
plt.xlabel("X")
plt.ylabel("time")
plt.show()