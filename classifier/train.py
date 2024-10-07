import torch
import numpy as np
from tqdm import tqdm
from classifier.metrics import calculate_metric

def train_one_epoch(dataloader, model, optimizer, scheduler, criterion, cfg):
    model.train()
    final_y = []
    final_y_pred = []
    final_loss = []

    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = batch[0].to(cfg["device"])
        y = batch[1].to(cfg["device"])

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            y_pred = model(X)
            loss = criterion(y_pred, y)

            y = y.detach().cpu().numpy().tolist()
            y_pred = y_pred.detach().cpu().numpy().tolist()

            final_y.extend(y)
            final_y_pred.extend(y_pred)
            final_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        scheduler.step()

    loss = np.mean(final_loss)
    final_y_pred = np.argmax(final_y_pred, axis=1)
    metric = calculate_metric(final_y, final_y_pred)

    return metric, loss

def validate_one_epoch(dataloader, model, criterion, cfg):
    model.eval()
    final_y = []
    final_y_pred = []
    final_loss = []

    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = batch[0].to(cfg["device"])
        y = batch[1].to(cfg["device"])

        with torch.no_grad():
            y_pred = model(X)
            loss = criterion(y_pred, y)

            y = y.detach().cpu().numpy().tolist()
            y_pred = y_pred.detach().cpu().numpy().tolist()

            final_y.extend(y)
            final_y_pred.extend(y_pred)
            final_loss.append(loss.item())

    loss = np.mean(final_loss)
    final_y_pred = np.argmax(final_y_pred, axis=1)
    metric = calculate_metric(final_y, final_y_pred)

    return metric, loss
