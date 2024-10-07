import torch
from classifier.config import cfg
from classifier.model import get_model
from classifier.train import train_one_epoch, validate_one_epoch
from classifier.dataset import CustomDataset
from classifier.utils import set_seed, visualize_history
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Load the dataset
root_dir = cfg["root_dir"]
sub_folders = ["Happy", "Sad", "Surprise"]
labels = [0, 1, 2]

data = []
for s, l in zip(sub_folders, labels):
    for r, d, f in os.walk(os.path.join(root_dir, s)):
        for file in f:
            if '.jpg' in file:
                data.append([os.path.join(s, file), l])

df = pd.DataFrame(data, columns=['file_name', 'label'])

# Split the dataset
df_train, df_valid = train_test_split(df, test_size=0.2, random_state=cfg["seed"])

train_dataset = CustomDataset(cfg, df_train)
valid_dataset = CustomDataset(cfg, df_valid)

train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=cfg["batch_size"], shuffle=False)

# Model and optimizer
set_seed(cfg["seed"])
model = get_model(cfg)
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["lr_min"])

# Training
train_acc_history, train_loss_history = [], []
val_acc_history, val_loss_history = [], []

for epoch in range(cfg["epochs"]):
    print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")

    train_metric, train_loss = train_one_epoch(train_loader, model, optimizer, scheduler, criterion, cfg)
    val_metric, val_loss = validate_one_epoch(valid_loader, model, criterion, cfg)

    train_acc_history.append(train_metric)
    val_acc_history.append(val_metric)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    # Print train and validation accuracy
    print(f"Train Accuracy: {train_metric:.4f}, Validation Accuracy: {val_metric:.4f}")

visualize_history(train_acc_history, train_loss_history, val_acc_history, val_loss_history)

save_path = "C:/Users/Victor Cardenas/Documents/msc/semestre-3/idi_iii/fer_2024/models/model.pth"  # Adjust the path as needed

# Save the model state dictionary
torch.save(model.state_dict(), save_path)