import os
import numpy as np
import pathlib
import rasterio
import cv2
import torch
from torch.utils.data import Dataset

from PIL import Image
from typing import Tuple, Dict, List

def sen12_label_transform(source_file, desired_season):
    """
    Pass in a the names of all the attributes that you want
    """

    file = open(source_file).read().split()
    attr_names = file[0].split(',')
    file = file[1:]
    print(file)
    def transform(idx):
        attr = torch.tensor([int(entry) for entry in file[idx].split(',')[1:]])
        mask = [attr_names[1:][i] in desired_season for i in range(len(attr))]
        masked = attr[mask]
        return torch.relu(masked).float()
    return transform

class SEN12MS(Dataset):
    """Custom dataset for SEN12MS data"""

    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, imgtransform=None, anntransform=None, bands="rgb") -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*/*.tif")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.imgtransform = imgtransform
        self.anntransform = anntransform
        self.bands = bands
        # Create classes and class_to_idx attributes


    # 4. Make function to load images
    def load_image(self, index: int):# -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        if self.bands == "rgb":
            image = rasterio.open(str(image_path)).read([2,3,4])
        else:
            image = rasterio.open(str(image_path)).read()

        image_norm = cv2.normalize(image.astype(np.float32), dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        return np.transpose(image_norm,(1,2,0))
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        # Transform if necessary
        if self.imgtransform:
            return self.imgtransform(img), self.anntransform(index) # return data, label (X, y)
        else:
            return img, self.anntransform(index) # return data, label (X, y)

def sen12_overfit_label_transform(source_file, desired_season):
    """
    Pass in a the names of all the attributes that you want
    """

    file = open(source_file).read().split()
    attr_names = file[0].split(',')
    file = file[1:]
    
    def transform(idx):
        attr = torch.tensor([int(entry) for entry in file[0].split(',')[1:]])
        mask = [attr_names[1:][i] in desired_season for i in range(len(attr))]
        masked = attr[mask]
        return torch.relu(masked).float()
    return transform

class SEN12MS_overfit(Dataset):
    def __init__(self, image_path, imgtransform=None, anntransform=None, bands="rgb", length=1000):
        self.image_path = image_path
        self.imgtransform = imgtransform
        self.anntransform = anntransform
        self.bands = bands
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, x):
        # open image here as PIL / numpy
        if self.bands == "rgb":
            image = rasterio.open(str(self.image_path)).read([2,3,4])
        else:
            image = rasterio.open(str(self.image_path)).read()
        image_norm = cv2.normalize(image.astype(np.float32), dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        image = np.transpose(image_norm,(1,2,0))
        if self.imgtransform:
            return self.imgtransform(image), self.anntransform(x) # return data, label (X, y)
        else:
            return image, self.anntransform(x) # return data, label (X, y)