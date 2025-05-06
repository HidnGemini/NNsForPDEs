import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
import torch
from torch import nn

from utils import NeuralNetwork
import utils

if __name__ == "__main__":
    model = torch.load('heatEqPINN2.pth', weights_only=False)
    fxn = utils.modelToFxn(model)
    utils.graph3D(fxn)
