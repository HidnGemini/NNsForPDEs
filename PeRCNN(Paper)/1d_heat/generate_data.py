import numpy as np
import math
import torch
import scipy as sp

def heatEquationSolution(x, t):
    """
    returns the well defined solution to the 1d heat equation at time t and position x
    """
    return (1.5*math.sin(math.pi*x)*math.e**(-1*(math.pi**2)*t)) - (0.5*math.sin(3*math.pi*x)*math.e**(-9*(math.pi**2)*t))

if __name__ == "__main__":
    num_t_steps = 1000
    dt = 1/num_t_steps
    num_x_steps = 14
    dx = 1/num_x_steps

    all_states = []
    for t in range(num_t_steps+1):
        state_list = []
        for x in range(num_x_steps+1):
            state_list.append(heatEquationSolution(x*dx,t*dt))
        all_states.append(state_list)
    data_matrix = torch.tensor([all_states])

    sp.io.savemat(f'PeRCNN(Paper)/1d_heat/1x{num_t_steps+1}x{num_x_steps+1}_heat_eq_data.mat', {'tensor': data_matrix})
