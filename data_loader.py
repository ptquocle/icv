import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

class MIMICCXRDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.image_dir = config.IMAGE_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Path: files/pXX/pXXXXXXXX/sXXXXXXXX/XXXX.jpg
        subject_id_str = f"p{row['subject_id']}"
        folder_group = subject_id_str[:3]
        study_id_str = f"s{row['study_id']}"
        dicom_id = f"{row['dicom_id']}.jpg"

        img_path = os.path.join(self.image_dir, folder_group, subject_id_str, study_id_str, dicom_id)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # For missing/corrupt images
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE))

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[config.CLASSES].values.astype(float), dtype=torch.float32)
        return image, labels

def get_dataloaders():
    print("Loading pre-split datasets...")

    # Read from the files created by split_files.py
    train_df = pd.read_csv('mimic_train.csv')
    val_df = pd.read_csv('mimic_val.csv')

    for sub_df in [train_df, val_df]:
        sub_df[config.CLASSES] = sub_df[config.CLASSES].fillna(0.0)
        if config.UNCERTAINTY_STRATEGY == "u-ones":
            sub_df[config.CLASSES] = sub_df[config.CLASSES].replace(-1.0, 1.0)
        else:
            sub_df[config.CLASSES] = sub_df[config.CLASSES].replace(-1.0, 0.0)

    print(f"Dataset ready: {len(train_df)} train, {len(val_df)} val.")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    val_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        normalize
        ])

    train_dataset = MIMICCXRDataset(train_df, transform=train_transform)
    val_dataset = MIMICCXRDataset(val_df, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,             
        pin_memory=False,           
        persistent_workers=True,   
        prefetch_factor=2       
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,             
        pin_memory=False
    )

    return train_loader, val_loader
