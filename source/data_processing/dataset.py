import torchvision
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from source.utils import normalize_year

SIZE = (224, 224)



class WearingDataset(Dataset):
    def __init__(self, path_csv: str, transform):
        xy = np.loadtxt(path_csv, dtype=str, delimiter=",", skiprows=1)
        self.x = xy[:, 0]
        self.y = torch.from_numpy(
            normalize_year(xy[:, [1]].astype(np.float32)).astype(np.float32))
        self.n_samples: int = xy.shape[0]

        self.transform: bool = transform

    def __getitem__(self, index):
        x, y = Image.open("../" + self.x[index]).convert('RGB'), self.y[index]
        x = transforms.Resize(255)(x)
        if self.transform:
            x = x
        x = transforms.CenterCrop(224)(x)
        x = transforms.ToTensor()(x)
        x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)

        return x, y

    def __len__(self):
        return self.n_samples
