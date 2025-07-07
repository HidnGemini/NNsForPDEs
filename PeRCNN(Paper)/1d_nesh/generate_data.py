import numpy as np
import math
import torch
import scipy as sp

import reference_solution as rs

if __name__ == "__main__":
    num_t_steps = 1000
    dt = 1/num_t_steps

    soln = rs.generate_reference_solution(runtime = 200, num_steps = num_t_steps, verbose=False, with_diffusion=True)
    
    data_matrix = soln[:1, :, ::22]

    print(data_matrix.shape)

    sp.io.savemat(f'PeRCNN(Paper)/1d_nesh/1x{num_t_steps}x{15}_n_tot_data.mat', {'tensor': data_matrix})
