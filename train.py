import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import config
from data_loader import get_dataloaders
from model import DenseNet14
from utils import compute_auroc


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Traning on: {}".format(device))
    train_loader, val_loader = get_dataloaders()
    model = DenseNet14().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    best_val_auroc = 0.0
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        print("Epoch {}/{}".format(epoch+1, config.NUM_EPOCHS))
        progress_bar = tqdm(train_loader, desc="training")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_outputs.append(outputs)
                all_labels.append(labels)
        avg_val_loss = val_loss / len(val_loader)

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        val_auroc = compute_auroc(all_outputs, all_labels)
        print("train loss: {:.4f} | val loss: {:.4f} | val macro AUROC: {:.4f}".format(avg_train_loss, avg_val_loss, val_auroc))

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/model.pth")
            print("saved best model")
            
if __name__ == "__main__": train()
