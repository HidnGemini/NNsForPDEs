import torch
from torch import nn

from utils import NeuralNetwork
import utils

# choose device for training
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# which device are we using?
print(f"Using {device} device")
    
def train_loop(model, optimizer, num_gen, num_ic, num_bc):
    # Diclosure: I used chatGPT (as per Branson's suggestion) 
    # to learn about inputing multiple inputs at the same time
    # rather than training this one at a time. This is so so so
    # much faster than what I was doing before (training
    # one input at a time, ~10000 inputs per epoch)
    model.train()

    # general case (pde loss)
    x_gen = torch.rand((num_gen, 1))
    t_gen = torch.rand((num_gen, 1))
    in_gen = torch.cat([x_gen, t_gen], dim=1).to(device)
    loss_pde = utils.pdeLoss(in_gen, model, 1.0)

    # initial condition
    x_ic = torch.linspace(0, 1, num_ic).view(-1, 1)
    t_ic = torch.zeros_like(x_ic) # time t = 0
    inputs_ic = torch.cat([x_ic, t_ic], dim=1).to(device)
    u_ic_expected = torch.sin(torch.pi * x_ic).to(device)
    u_ic_actual = model(inputs_ic)
    loss_ic = nn.MSELoss()(u_ic_expected, u_ic_actual)

    # boundry condition (u(0,t) = u(1,t)) 
    # boundry condition is also something that chatGPT suggested that I didn't consider
    t_bc = torch.linspace(0, 1, num_bc).view(-1, 1).to(device)
    x_bc_zeros = torch.zeros_like(t_bc).to(device)
    x_bc_ones = torch.zeros_like(t_bc).to(device)
    inputs_ic_zeros = torch.cat([x_bc_zeros, t_bc], dim=1).to(device) 
    out_bc_zeros = model(inputs_ic_zeros) # calculate u(0,t)s
    inputs_ic_ones = torch.cat([x_bc_ones, t_bc], dim=1).to(device) 
    out_bc_ones = model(inputs_ic_ones)
    loss_bc = nn.MSELoss()(out_bc_zeros, torch.zeros_like(out_bc_zeros)) + nn.MSELoss()(out_bc_ones, torch.zeros_like(out_bc_ones))

    # Total loss
    loss = loss_pde + loss_ic + loss_bc
    loss.backward()
    optimizer.step()

    return loss
    
# create a model and print the architecture
model = NeuralNetwork().to(device)
# print(model)

# constants
learning_rate = 1*(1e-4)
min_epochs = 100000
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []
natural_minimum = float("inf")

i = 0
while (i<min_epochs) or (losses[-1] > natural_minimum):
    if ((i+1) % 25 == 0):
        print(f"Epoch {i+1}/{min_epochs}\n-------------------------------")
        losses.append(train_loop(model, optimizer, 10000, 1000, 1000))
        print(f"Loss: {losses[-1]}")
        print()
    else:
        losses.append(train_loop(model, optimizer, 10000, 1000, 1000))
    i += 1
    if (i == min_epochs):
        natural_minimum = min(losses)


nnFunction = utils.modelToFxn(model)
utils.graph2D(nnFunction)

torch.save(model, 'heatEqPINN(extraEpochs).pth')

