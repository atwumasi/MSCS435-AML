import time

import torch
import torch.nn.functional as F


def compute_epoch_metrics(model, data_loader, device):
    """Average per-pixel cross-entropy loss and pixel accuracy over a data loader."""
    model.eval()
    total_loss, correct_pixels, total_pixels = 0.0, 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            total_loss += loss.item()
            total_pixels += targets.numel()
            predicted = torch.argmax(logits, dim=1)
            correct_pixels += (predicted == targets).sum().item()
    avg_loss = total_loss / total_pixels
    accuracy = 100.0 * correct_pixels / total_pixels
    return avg_loss, accuracy


def train_model(model, train_loader, valid_loader, optimizer, device, num_epochs=20):
    """
    Trains `model` for `num_epochs`, logging per-epoch train/validation loss and pixel
    accuracy. Returns (train_loss_list, train_acc_list, valid_loss_list, valid_acc_list).
    """
    train_loss_list, train_acc_list = [], []
    valid_loss_list, valid_acc_list = [], []

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            ### FORWARD AND BACK PROP
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

        ### LOGGING
        train_loss, train_acc = compute_epoch_metrics(model, train_loader, device)
        valid_loss, valid_acc = compute_epoch_metrics(model, valid_loader, device)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

        print(f'Epoch {epoch + 1:03d}/{num_epochs:03d} '
              f'| loss: {train_loss:.4f} - accuracy: {train_acc:.4f} '
              f'| val_loss: {valid_loss:.4f} - val_accuracy: {valid_acc:.4f}')

    print(f'Total Training Time: {(time.time() - start_time) / 60:.2f} min')
    return train_loss_list, train_acc_list, valid_loss_list, valid_acc_list
