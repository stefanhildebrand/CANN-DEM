import torch
from torch.autograd import grad
import numpy as np
import numpy.random as npr
from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib as mpl
import time
from mpl_toolkits.mplot3d import Axes3D
from pyevtk.hl import gridToVTK
import numpy.matlib as ml

device = torch.device('cpu')
if torch.cuda.is_available():# and False: # Jonathan
    print("CUDA is available, running on GPU")
    device = torch.device('cuda')
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("CUDA not available, running on CPU")
#mpl.rcParams['figure.dpi'] = 120 # Jonathan #350
# fix random seeds
npr.seed(2019)
torch.manual_seed(2019)