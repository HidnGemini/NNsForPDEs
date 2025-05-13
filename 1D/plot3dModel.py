import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
import torch
from torch import nn

from utils1D import NeuralNetwork
import utils1D

if __name__ == "__main__":
    model_file = '1D/heatEqPINN(extraEpochs1).pth'

    model = torch.load(model_file, weights_only=False, map_location=torch.device("mps"))
    fxn = utils1D.modelToFxn(model)
    # utils1D.graph3D(fxn)

    # utils1D.graph2D(fxn)
    # utils1D.graph2D(utils1D.heatEquationSolution)

    utils1D.graphAnimated2D(fxn)
