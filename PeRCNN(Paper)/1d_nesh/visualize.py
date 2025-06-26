import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.io as scio
import time
import os
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import train_1dheat_redo

# setup gpu / npu
torch.set_default_dtype(torch.float32)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# constants for net
time_steps = 2500   # 200->400->800 multi-stage training works best, then 2500 for inference.
dt = 0.5
dx = 1.0/100
dy = 1.0/100

# load low res data matrix
data = sp.io.loadmat('./PeRCNN(Paper)/1d_heat/1x1001x15_heat_eq_data.mat')['tensor']
datamat = torch.from_numpy(np.transpose(data, (0, 1, 2)).astype(np.float32))             # 1x1001x15

IC = datamat[:, 0:1, :] # first timestep is initial condition
U0_low = IC[0, 0, ::4] # lower resolution? frankly i don't know why we're doing that
init_state_low = torch.tensor((U0_low)).to(device)

# this is terrible. i NEED to rename these, but this is mostly just straight from their code
time_batch_size = time_steps
steps = time_batch_size + 1
effective_step = list(range(0, steps))

# make model
model = train_1dheat_redo.RCNN(
    input_channels = 2, 
    hidden_channels = 8,
    init_state_low = init_state_low,
    input_kernel_size = 5,
    step = steps, 
    effective_step = effective_step
).to(device)

# load model
checkpoint = torch.load('./PeRCNN(Paper)/1d_heat/model/checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = optim.Adam(model.parameters(), lr=0.0)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.98)

# get model output
data, _ = model()
data_tensor = torch.concat(data).squeeze(1).cpu().detach()

# double check size is right
print(data_tensor.shape)

# graphing time
x = np.linspace(0, 1, 52)

fig, ax = plt.subplots()

init_data = data_tensor[0]

line, = ax.plot(x, init_data) # comma is python magic to unpack a list :)

def animate(step):
    t = step / 250
    u = data_tensor[step]
    line.set_ydata(u)
    ax.set_title(f"Time t={t}")
    return line,

ani = animation.FuncAnimation(
    fig, animate, interval=10, blit=False, save_count=50)

plt.show()