import torch
from torch import nn
import time
import gc

from utils2D import NeuralNetwork
import utils2D

# choose device for training
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# which device are we using?
print(f"Using {device} device")
    
def train_loop(model, optimizer, num_gen, num_ic, num_bc, print_info):
    model.train()

    # general case (PDE loss)
    x_gen = torch.rand((num_gen, 1))
    y_gen = torch.rand((num_gen, 1))
    t_gen = torch.rand((num_gen, 1))
    in_gen = torch.cat([x_gen, y_gen, t_gen], dim=1).to(device)
    loss_pde = utils2D.pdeLoss(in_gen, model, 1.0)

    # free tensors used in PDE loss
    del x_gen
    del y_gen
    del t_gen
    del in_gen

    # initial condition
    x_ic = torch.rand((num_ic, 1))
    y_ic = torch.rand((num_ic, 1))
    t_ic = torch.zeros_like(x_ic) # time t = 0
    inputs_ic = torch.cat([x_ic, y_ic, t_ic], dim=1).to(device)
    u_ic_expected = (torch.sin(torch.pi * x_ic) + torch.sin(torch.pi * y_ic)).to(device)
    u_ic_actual = model(inputs_ic)
    loss_ic = nn.MSELoss()(u_ic_expected, u_ic_actual)

    # free tensors used in ic
    del x_ic
    del y_ic
    del t_ic
    del inputs_ic
    del u_ic_expected
    del u_ic_actual

    # boundry condition for x (u(0,y,t) = u(1,x,t) = 0) 
    t_bc = torch.linspace(0, 1, num_bc).view(-1, 1).to(device)
    x_bc_zeros = torch.zeros_like(t_bc).to(device)
    y_bc = torch.rand((num_bc, 1)).to(device)
    x_bc_ones = torch.zeros_like(t_bc).to(device)
    inputs_ic_zeros_x = torch.cat([x_bc_zeros, y_bc, t_bc], dim=1).to(device) 
    out_bc_zeros_x = model(inputs_ic_zeros_x) # calculate u(0,y,t)s
    inputs_ic_ones_x = torch.cat([x_bc_ones, y_bc, t_bc], dim=1).to(device) 
    out_bc_ones_x = model(inputs_ic_ones_x) # calculte u(1,y,t)s
    loss_bc_x = nn.MSELoss()(out_bc_zeros_x, torch.zeros_like(out_bc_zeros_x)) + nn.MSELoss()(out_bc_ones_x, torch.zeros_like(out_bc_ones_x))

    # free tensors used in bc_x
    del t_bc
    del x_bc_zeros
    del y_bc
    del x_bc_ones
    del inputs_ic_zeros_x
    del out_bc_zeros_x
    del inputs_ic_ones_x
    del out_bc_ones_x

    # boundry condition for y (u(x,0,t) = u(x,1,t) = 0) 
    t_bc = torch.linspace(0, 1, num_bc).view(-1, 1).to(device)
    y_bc_zeros = torch.zeros_like(t_bc).to(device)
    x_bc = torch.rand((num_bc, 1)).to(device)
    y_bc_ones = torch.zeros_like(t_bc).to(device)
    inputs_ic_zeros_y = torch.cat([x_bc, y_bc_zeros, t_bc], dim=1).to(device) 
    out_bc_zeros_y = model(inputs_ic_zeros_y) # calculate u(x,0,t)s
    inputs_ic_ones_y = torch.cat([x_bc, y_bc_ones, t_bc], dim=1).to(device) 
    out_bc_ones_y = model(inputs_ic_ones_y) # calculate u(x,1,t)s
    loss_bc_y = nn.MSELoss()(out_bc_zeros_y, torch.zeros_like(out_bc_zeros_y)) + nn.MSELoss()(out_bc_ones_y, torch.zeros_like(out_bc_ones_y))

    # free tensors used in bc_y
    del t_bc
    del y_bc_zeros
    del x_bc
    del y_bc_ones
    del inputs_ic_zeros_y
    del out_bc_zeros_y
    del inputs_ic_ones_y
    del out_bc_ones_y

    # force garbage collector
    gc.collect()

    # empty cache
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()
    # not necessary for CPU since gc.collect() handles that

    # Total loss
    loss = loss_pde + loss_ic + loss_bc_x + loss_bc_y
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    lossVal = float(loss)

    del loss
    del loss_pde
    del loss_ic
    del loss_bc_x
    del loss_bc_y

    if ((print_info.get('epoch')+1) % print_info.get('print_frequency') == 0):
        # force garbage collector
        gc.collect()

        # empty cache
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
        # not necessary for CPU since gc.collect() handles that

        # print stuff!
        elapsed = time.time() - print_info.get('start_time')
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        print(
            f"Epoch {print_info.get('epoch')+1} / {print_info.get('min_epochs')} : Time = {mins}m {secs:2.0f}s Loss = {lossVal:.8f}"
        )

    return lossVal
    
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
start_time = time.time()
while (i<min_epochs) or (losses[-1] > natural_minimum):
    losses.append(train_loop(
        model=model, 
        optimizer=optimizer, 
        num_gen=10000, 
        num_ic=5000, 
        num_bc=5000,
        print_info = {
            'epoch' : i,
            'print_frequency' : 100,
            'min_epochs' : min_epochs,
            'start_time' : start_time
        })
    )
    i += 1
    if (i == min_epochs):
        natural_minimum = min(losses)

torch.save(model, '2DPINN_2.pth')

