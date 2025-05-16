import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ---- Parameters ----
alpha = 0.01  # thermal diffusivity
nx = 50       # number of spatial points
nt = 1000      # number of time steps
dx = 1.0 / nx
dt = 0.01

# ---- Synthetic Initial Condition ----
x = torch.linspace(0, 1, nx).unsqueeze(0)  # shape: [1, nx]
u0 = torch.sin(torch.pi * x)  # initial temperature profile
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

u_data = generate_true_solution(u0, nt, alpha, dx, dt).detach()

import torch
import torch.nn as nn

class HeatRNN(nn.Module):
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

# ---- Physics loss (finite-difference approximation of heat equation) ----
def physics_loss(u_pred, alpha, dx, dt):
    u_t = (u_pred[1:] - u_pred[:-1])[:,0,:] / dt
    u_x = (u_pred[:, 0, 1:] - u_pred[:, 0, :-1]) / dx
    u_xx = (u_x[:,1:] - u_x[:,:-1]) / dx
    residual = u_t[:,:-2] - alpha * u_xx[:-1]

    return nn.MSELoss()(residual, torch.zeros_like(residual))

# ---- Training ----
model = HeatRNN(input_size=nx, hidden_size=64)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fxn = nn.MSELoss()

num_epochs = 50000
u_pred = model(u0, nt)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    u_pred = model(u0, nt)  # [nt, 1, nx]
    loss_data_ic = loss_fxn(u_pred, u_data)
    loss_phys = physics_loss(u_pred, alpha, dx, dt)
    loss = loss_data_ic + 10*loss_phys
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Data Loss: {loss_data_ic.item():.4f}, Physics Loss: {loss_phys.item():.4f}")

torch.save(model, "PeRCNN2.pth")

print(f"Final Epoch, Data Loss: {loss_data_ic.item():.4f}, Physics Loss: {loss_phys.item():.4f}")
u_pred = model(u0, nt)

# ---- Prediction example ----
plt.imshow(u_pred.squeeze().detach().numpy(), aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Temperature over Time")
plt.xlabel("X")
plt.ylabel("time")
plt.show()