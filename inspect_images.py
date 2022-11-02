import matplotlib.pyplot as plt
import numpy as np
import torch.nn
from PIL import Image
from torch.utils.data import DataLoader
from torchsummary import summary

from source.data_processing.dataset import WearingDataset

# data_processing
from source.models.resnet18 import ResnetRegression
from source.trainingmanager.TrainingManager import EarlyStopping, fit, validate

from torchvision import transforms
TRAINING_DATA = "data/dresses_train.csv"


xy = np.loadtxt(TRAINING_DATA, dtype=str, delimiter=",", skiprows=1)
x = xy[:, 0]

index = 0
x= Image.open(x[index]).convert('RGB')
x = transforms.Resize(236)(x)
x = transforms.CenterCrop(224)(x)
x.show()
