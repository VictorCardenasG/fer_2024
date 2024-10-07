# Source code for pre-trained model import and loading
import timm
import torch

def get_model(cfg):
    model = timm.create_model(cfg["backbone"], pretrained=True, num_classes=cfg["n_classes"]).to(cfg["device"])
    return model

def load_model(cfg, path):
    model = get_model(cfg)
    model.load_state_dict(torch.load(path))
    return model
