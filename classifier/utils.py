import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize_history(acc, loss, val_acc, val_loss):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(range(len(loss)), loss, color='darkgrey', label='train')
    ax[0].plot(range(len(val_loss)), val_loss, color='cornflowerblue', label='valid')
    ax[0].set_title('Loss')

    ax[1].plot(range(len(acc)), acc, color='darkgrey', label='train')
    ax[1].plot(range(len(val_acc)), val_acc, color='cornflowerblue', label='valid')
    ax[1].set_title('Metric (Accuracy)')

    for i in range(2):
        ax[i].set_xlabel('Epochs')
        ax[i].legend(loc="upper right")
    plt.show()
