import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
import torch
from torch import nn

from utils2D import NeuralNetwork
import utils2D

if __name__ == "__main__":
    model_file = '2D/2DPINN2.pth'

    model = torch.load(model_file, weights_only=False, map_location=torch.device("mps"))
    fxn = utils2D.modelToFxn(model)

    utils2D.graphAnimated3D(fxn)