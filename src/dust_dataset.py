import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

import pickle
import os

# Load the dataset from the specified file
data_path = '/home/daisysong/2024-HL-Virtual-Dosimeter/analysis/experiments_info_tables/exp_type-classification/exp-1/metadata.csv'
df = pd.read_csv(data_path)

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a manageable size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

transform = get_transform()

# Function to load and process images
def load_and_process_images(image_paths, transform):
    images = []
    for path in image_paths:
        try:
            image = Image.open(path).convert('RGB')
            if transform:
                image = transform(image)
            images.append(image)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    return images

# Extract video data and labels
def extract_video_and_labels(df):
    video_data = []
    labels = []

    for _, row in df.iterrows():
        image_paths = eval(row['input_sdoml_data_filepaths'])  # Convert string representation of list to actual list
        images = load_and_process_images(image_paths, transform)
        if images:
            video_data.append(images)
            labels.append(row['target_window_num_rad_dose_obs_avail'])  # Use the total number of radiation dose observations as the label

    return video_data, labels
    
class DustDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_paths = eval(row['input_sdoml_data_filepaths'])
        images = self.load_and_process_images(image_paths)
        target = row['target_window_num_rad_dose_obs_avail']
        return torch.stack(images), target

    def load_and_process_images(self, image_paths):
        images = []
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
        return images

def create_data_loaders(train_df, val_df, batch_size=32, num_workers=4):
    transform = get_transform()
    train_dataset = DustDataset(train_df, transform=transform)
    val_dataset = DustDataset(val_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
