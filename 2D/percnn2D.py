import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Parameters
alpha = 0.01         # thermal diffusivity
dx = 1.0 / 14        # spatial step (assuming 15 points from 0 to 1)
dt = 0.01            # time step
n_x = 15             # number of spatial points
n_t = 100            # number of time steps

# Laplacian using finite difference
def compute_laplacian(u, dx):
    # u: shape (batch, n_x)
    u_pad = torch.nn.functional.pad(u.unsqueeze(1), (1, 1), mode='replicate').squeeze(1)
    lap = (u_pad[:, 2:] - 2 * u + u_pad[:, :-2]) / dx**2
    return lap

# Residual network for physics correction
class ResidualNet(nn.Module):
    def __init__(self, n_x):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_x, 64),
            nn.Tanh(),
            nn.Linear(64, n_x)
        )
    
    def forward(self, u):
        return self.fc(u)

# RNN Cell: physics + learned residual
class HeatRNNCell(nn.Module):
    def __init__(self, n_x, alpha, dx, dt):
        super().__init__()
        self.alpha = alpha
        self.dx = dx
        self.dt = dt
        self.residual_net = ResidualNet(n_x)
    
    def forward(self, u):
        lap = compute_laplacian(u, self.dx)
        physics_update = self.alpha * self.dt * lap
        correction = self.residual_net(u)
        return u + physics_update + correction

# RNN model for full sequence prediction
class HeatRNN(nn.Module):
    def __init__(self, cell, n_steps):
        super().__init__()
        self.cell = cell
        self.n_steps = n_steps

    def forward(self, u0):
        outputs = [u0]
        u = u0
        for _ in range(self.n_steps):
            u = self.cell(u)
            outputs.append(u)
        return torch.stack(outputs, dim=1)  # shape (batch, time, space)

# Synthetic data (for demo)
def generate_data(u0, n_steps, alpha, dx, dt):
    u = u0.clone()
    history = [u0]
    for _ in range(n_steps):
        lap = compute_laplacian(u, dx)
        u = u + alpha * dt * lap
        history.append(u)
    return torch.stack(history, dim=1)

# Create initial condition batch
batch_size = 32
torch.manual_seed(0)
u0_batch = torch.sin(torch.linspace(0, 3.14, n_x)).repeat(batch_size, 1) * torch.rand(batch_size, 1)
u_true = generate_data(u0_batch, n_t, alpha, dx, dt)

# Model and training
cell = HeatRNNCell(n_x, alpha, dx, dt)
model = HeatRNN(cell, n_t)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
n_epochs = 100000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    u_pred = model(u0_batch)
    loss = loss_fn(u_pred, u_true)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Plot a result
plt.plot(u_true[0, -1].detach(), label="Ground Truth")
plt.plot(u_pred[0, -1].detach(), '--', label="Predicted")
plt.legend()
plt.title("Final State Comparison")
plt.xlabel("Position")
plt.ylabel("u(x, t)")
plt.show()
