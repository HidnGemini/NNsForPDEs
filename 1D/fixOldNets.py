import torch
from utils1D import NeuralNetwork
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

oldModel = torch.load("heatEqReLU.pth", weights_only=False)

class NeuralNetwork(nn.Module):
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

old_state_dict = oldModel.state_dict()

print(old_state_dict)

# Rename keys
new_state_dict = {}
for key in old_state_dict:
    new_key = key.replace('linear_relu_stack', 'layer_stack')
    new_state_dict[new_key] = old_state_dict[key]

print(new_state_dict)

# Load into the new model
model = NeuralNetwork().to(device)
model.load_state_dict(new_state_dict)

torch.save(model, "fixed.pth")