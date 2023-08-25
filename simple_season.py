from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import torch
import random

class simpleDS(Dataset):
    def __init__(self, imgtransform, anntransform, length):
        self.imgtransform = imgtransform
        self.anntransform = anntransform
        self.length = length
        self.ds = ImageFolder("/home/paddy/git/input/simple_season", transform = imgtransform)
        
    def __getitem__(self, idx):
        idx = random.randint(0,3)
        return (self.ds[idx][0], self.anntransform(idx))

    def __len__(self):
        return self.length

def simple_label_transform(desired_colour):
    """
    Pass in a the names of all the attributes that you want
    """

    file = open('/home/paddy/git/input/simple_season/simple_season.csv').read().split()
    attr_names = file[0].split(',')
    file = file[1:]
    
    def transform(idx):
        attr = torch.tensor([int(entry) for entry in file[idx].split(',')[1:]])
        mask = [attr_names[1:][i] in desired_colour for i in range(len(attr))]
        masked = attr[mask]
        return torch.relu(masked).float()
    return transform

def dummy_label_transform():
    def transform(idx):
        return torch.tensor([0, 0])
    return transform

file = open('/home/paddy/git/input/simple_season/simple_season.csv').read().split()
attr_names = file[0].split(',')
