# Source code for dataset creation and manipulations

import os
import cv2
import torch
import albumentations as A
from classifier.config import cfg
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.root_dir = cfg["root_dir"]
        self.df = df
        self.file_names = df['file_name'].values
        self.labels = df['label'].values

        self.transform = transform or A.Compose([
                              A.Resize(cfg["image_size"], cfg["image_size"]),
                              ToTensorV2(),
                           ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = os.path.join(self.root_dir, self.file_names[idx])

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        image = augmented['image']

        image = image / 255.0

        return image, label
