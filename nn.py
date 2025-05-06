import torch
from torch import nn

from utils import NeuralNetwork
import utils

# choose device for training
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# which device are we using?
print(f"Using {device} device")
    
inps = [(x,t) for x in [0.01*i for i in range(101)] for t in [0.01*i for i in range(101)]]

def discrete_soln_train_loop(inps, model, loss_fn, optimizer):
    # # calculate intended outputs (yes, this is not at all a PINN, but this is the first
    # # concept)
    # outs = [utils.heatEquationSolution(x, t) for (x,t) in inps]

    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    epochLoss = 0
    
    for i,(x,t) in enumerate(inps):
        # put inputs into a tensor
        inp = torch.zeros(1, 2, device=device)
        inp[0][0] = x
        inp[0][1] = t

        # Compute prediction
        pred = model(inp)

        # get expected output
        expected = torch.zeros(1, 1, device=device)
        expected[0][0] = utils.heatEquationSolution(x,t) # use the known solution for loss

        loss = loss_fn(pred, expected)
        epochLoss += loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Done! Avg Loss: {epochLoss/len(inps):.6f}")
    
# create a model and print the architecture
model = NeuralNetwork().to(device)
# print(model)

# constants
learning_rate = 1*(1e-3)
epochs = 500
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    discrete_soln_train_loop(inps, model, loss_fn, optimizer)

utils.graph2D(utils.heatEquationSolution)

nnFunction = utils.modelToFxn(model)
utils.graph2D(nnFunction)

torch.save(model, 'heatEqRe10.pth')

