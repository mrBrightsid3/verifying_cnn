from torch.utils.data import Dataset
import torch
import pathlib
from typing import Tuple
from torchvision.transforms import ToTensor
from PIL import Image
import pandas as pd

def find_classes(dir):
    dir_csv = dir + '/main_dict.csv'
    df = pd.read_csv(dir_csv)
    #classes = df['Name'].unique().tolist()
    classes_to_idx = dict(zip(df['Name'], df['Key']))
    sorted_dict_classes_to_idx = dict(sorted(classes_to_idx.items(), key=lambda item: item[1]))
    #classes.sort()
    classes = list(sorted_dict_classes_to_idx.keys())
    img_name_idx = dict(zip(df['ImageId'], df['Key']))

    return classes, sorted_dict_classes_to_idx, img_name_idx

class CustomDataset(Dataset):
    def __init__(self, dir:str, transform=None):
        super().__init__()
        self.image_paths = list(pathlib.Path(dir).glob("val/*.JPEG"))
        self.transform = transform
        self.classes, self.classes_to_idx, self.img_name_to_idx = find_classes(dir)
        

    def load_image(self, index) -> Image.Image:
        path = self.image_paths[index]
        name = self.image_paths[index].stem
        return Image.open(path), name
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        img, img_name = self.load_image(index)
        class_idx = self.img_name_to_idx[img_name]
        if self.transform:
            return self.transform(img), class_idx
        else:
            return ToTensor()(img), class_idx
        
