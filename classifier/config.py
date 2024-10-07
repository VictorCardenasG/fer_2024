# Source code for configuration dictionary

import torch

cfg = {
    "root_dir": "C:/Users/Victor Cardenas/Documents/msc/semestre-3/idi_iii/fer_2024/data/interim/data_final_1500_evaluation",
    "image_size": 256,
    "batch_size": 32,
    "n_classes": 3,
    "backbone": 'resnet18',
    "learning_rate": 1e-3,
    "lr_min": 1e-5,
    "epochs": 20,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "seed": 42
}

