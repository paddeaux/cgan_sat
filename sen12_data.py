import os
import numpy as np
import pathlib
import rasterio
import cv2
import torch
from torch.utils.data import Dataset

from PIL import Image
from typing import Tuple, Dict, List

def sen12_label_transform(desired_season):
    """
    Pass in a the names of all the attributes that you want
    """

    file = open('C:/Users/Paddy/CRT/Github/input/SEN12MS/sen12_seasons.csv').read().split()
    season_names = file[0].split(',')
    file = file[1:]
    
    def transform(idx):
        season = torch.tensor([int(entry) for entry in file[idx].split(',')[1:]])
        mask = [season_names[1:][i] in desired_season for i in range(len(season))]
        masked = season[mask]
        return torch.relu(masked).float()
    return transform


# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Crearte a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class SEN12MS_RGB(Dataset):
    """Custom dataset for SEN12MS data"""

    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.tif")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int):# -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        image = rasterio.open(str(image_path)).read([2,3,4])
        image_norm = cv2.normalize(image.astype(np.float32), dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        #norm_image = (image - image.min()) / (image.max() - image.min())
        return np.transpose(image_norm,(1,2,0))
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)

class SEN12MS_FULL(Dataset):
    """Custom dataset for SEN12MS data"""

    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*/*.tif")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        self.label = "Spring"
        # Create classes and class_to_idx attributes
        #self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int):# -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        image = rasterio.open(str(image_path)).read()
        image_norm = cv2.normalize(image.astype(np.float32), dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        #norm_image = (image - image.min()) / (image.max() - image.min())
        return np.transpose(image_norm,(1,2,0))
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        #class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        #class_idx = self.class_to_idx[class_name]
        # removing the need for the class_idx for the full dataset
        class_idx = 0
        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)

class Sen12DS(Dataset):
    def __init__(self, imgtransform, anntransform):
        self.imgtransform = imgtransform
        self.anntransform = anntransform
        self.ds = ImageFolder("C:/Users/Paddy/CRT/Github/input/CelebA/", transform = imgtransform)
        
    def __getitem__(self, idx):
        return (self.ds[idx][0], self.anntransform(idx))

    def __len__(self):
        return len(self.ds)

def sen12_label_transform(desired_attr):
    """
    Pass in a the names of all the attributes that you want
    """
    file = open(os.path.join(os.path.dirname(os.getcwd()), "Input/SEN12MS/seasons.csv".read().split()))
    attr_names = file[0].split(',')
    file = file[1:]
    
    def transform(idx):
        attr = torch.tensor([int(entry) for entry in file[idx].split(',')[1:]])
        mask = [attr_names[1:][i] in desired_attr for i in range(len(attr))]
        masked = attr[mask]
        return torch.relu(masked).float()
    return transform

def dummy_label_transform():
    def transform(idx):
        return torch.tensor([0, 0])
    return transform

file = open(os.path.join(os.path.dirname(os.getcwd()), "Input/SEN12MS/seasons.csv".read().split()))
attr_names = file[0].split(',')
