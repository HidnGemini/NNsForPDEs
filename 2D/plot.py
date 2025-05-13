import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
import torch
from torch import nn

from utils2D import NeuralNetwork
import utils2D

if __name__ == "__main__":
    model_file = '1D/heatEqPINN(extraEpochs1).pth'

    # model = torch.load(model_file, weights_only=False, map_location=torch.device("mps"))
    # fxn = utils2D.modelToFxn(model)
    fxn = lambda x,y,t : np.sin(x*t) * np.cos(y*t)

    utils2D.graphAnimated3D(fxn)