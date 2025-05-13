import torch
import torch.nn as nn

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize the network
net = Net()

# Create inputs with requires_grad=True
x1 = torch.tensor([1.0], requires_grad=True)
x2 = torch.tensor([2.0], requires_grad=True)

# Stack into a single input tensor
# x = torch.stack([x1, x2])  # Shape: [2]
x = torch.tensor([1.0, 1], requires_grad=True)

# Forward pass
y = net(x)  # Shape: [1]

# First derivative: dy/dx2
dy_dx = torch.autograd.grad(
    outputs=y,
    inputs=x,
    grad_outputs=torch.ones_like(y),  # ∂y/∂y = 1
    create_graph=True
)[0]  # dy/dx is shape [2], corresponding to x1 and x2

dy_dx2 = dy_dx[1]

# Second derivative: d²y/dx2²
d2y_dx2 = torch.autograd.grad(
    outputs=dy_dx2,
    inputs=x,
    grad_outputs=torch.ones_like(dy_dx2),
    retain_graph=True
)[0][1]

print("dy/dx2:", dy_dx2.item())
print("d²y/dx2²:", d2y_dx2.item())