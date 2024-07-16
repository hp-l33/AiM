import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import DatasetFolder, ImageFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from typing import *


class FilterableImageFolder(ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            valid_classes: List = None
    ):
        self.valid_classes = valid_classes
        super(FilterableImageFolder, self).__init__(root, transform, target_transform, loader, is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        # class filter
        if self.valid_classes is not None:
            classes = [valid_class for valid_class in classes if valid_class in self.valid_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGB')
    return img
    

def build_dataset(data_path: str, final_reso: int = 256, hflip=False, mid_reso=1.125,
                  valid_classes=None, train_split='train', val_split=None):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(),
    ]
    valid_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(),
    ]
    if hflip:
        train_aug.insert(0, transforms.RandomHorizontalFlip())
        
    train_aug = transforms.Compose(train_aug)
    valid_aug = transforms.Compose(valid_aug)
    
    # build dataset
    train_set = FilterableImageFolder(root=os.path.join(data_path, train_split),
                                      loader=pil_loader,
                                      transform=train_aug,
                                      valid_classes=valid_classes)
    print(f'[Dataset] {len(train_set)=}', end=' ')
    
    if val_split is not None:
        valid_set = FilterableImageFolder(root=os.path.join(data_path, val_split),
                                          loader=pil_loader,
                                          transform=valid_aug,
                                          valid_classes=valid_classes)
        print(f'{len(valid_set)=}')
    else:
        valid_set = None
        print('valid_set is None')

    return train_set, valid_set