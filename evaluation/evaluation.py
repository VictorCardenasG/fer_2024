# Source code for evaluation stage

import os
import torch
import cv2
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from classifier.config import cfg


class CustomDataset(Dataset):
    def __init__(self, cfg, image_path, label=None, transform=None):
        self.image_path = image_path
        self.label = label

        self.transform = transform or A.Compose([
                              A.Resize(cfg["image_size"], cfg["image_size"]), 
                              ToTensorV2(),
                           ])
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        file_path = os.path.normpath(self.image_path)
        
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image from: {file_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        image = augmented['image']
        image = image / 255

        return image

def predict_emotion(cfg, model, image_path):
    dataset = CustomDataset(cfg, image_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for step, image in enumerate(dataloader):
        X = image.to(cfg["device"])

        with torch.no_grad():
            y_pred = model(X)
            y_pred_argmax = np.argmax(y_pred.cpu().numpy(), axis=1)[0]

    emotions = ["Happy", "Sad", "Surprise"]
    return emotions[y_pred_argmax]


# Traverse the folder structure and predict emotions for each image
def traverse_and_predict(main_folder, model):
    emotions = ["Happy", "Sad", "Surprise"]
    results = []

    for emotion in emotions:
        folder_path = os.path.join(main_folder, emotion)
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, file_name)
                predicted_emotion = predict_emotion(cfg, model, image_path)
                results.append((image_path, emotion, predicted_emotion))
                print(f"Image: {image_path}, True Emotion: {emotion}, Predicted Emotion: {predicted_emotion}")

    return results