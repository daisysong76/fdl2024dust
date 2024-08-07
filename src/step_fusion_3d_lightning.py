# https://www.youtube.com/watch?v=e6Nw01v2X4s
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/pytorch_lightning/2.%20LightningModule/simple_fc.py
import pytorch_lightning as pl#TODO: Check if this is necessary beneficial for multi-GPU training #pip install pytorch-lightning
import os
import pandas as pd
import numpy as np
import torch
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from models.video_swin_transformer import SwinTransformer3D
import math
import logging
import gc
import wandb
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from torch.nn import MSELoss
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from torchvision import transforms#used to preprocess images before feeding them into a model.

# Define transformations for SDO images #TODO: Check if this is necessary
sdo_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Resize((512, 512)),  # Resize the image to 512x512
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image
])


# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, sdo_data, labels):
        self.sdo_data = sdo_data
        self.labels = labels

    def __len__(self):
        return len(self.sdo_data)

    def __getitem__(self, idx):
        sdo = torch.tensor(self.sdo_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sdo, label

# Function to load SDO images (placeholder)
def load_sdo_images(input_window_start, input_window_end, sdo_data_dir):
    # List to store loaded images
    sdo_images = []
    # Iterate over the files in the specified directory
    for root, _, files in os.walk(sdo_data_dir):
        for file in files:
            parts = file.split('_')
            if len(parts) < 2:
                continue
            # Extract the timestamp from the filename
            timestamp_str = parts[1]
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M')
            except ValueError:
                continue
            # Check if the timestamp is within the input window
            if input_window_start <= timestamp <= input_window_end:
                # Load the image (placeholder for actual image loading)
                image = np.random.randn(224, 224)  # Replace with actual image loading
                sdo_images.append(image)
    return np.stack(sdo_images) if sdo_images else np.array([])

# Function to load radiation dose data (placeholder)
def load_radiation_data(target_window_start, target_window_end, rad_dose_filepath):
    rad_data = pd.read_csv(rad_dose_filepath)
    rad_data['timestamp_utc'] = pd.to_datetime(rad_data['timestamp_utc'])
    mask = (rad_data['timestamp_utc'] >= target_window_start) & (rad_data['timestamp_utc'] <= target_window_end)
    return rad_data.loc[mask, 'absorbed_dose_rate'].values

# Load metadata CSV
metadata_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/analysis/experiments_info_tables/exp_type-classification/exp-3/metadata.csv"
metadata_df = pd.read_csv(metadata_path)

# Directories
sdo_data_dir = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/google_cloud_buckets/sdoml"
rad_dose_filepath = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/google_cloud_buckets/radlab-private/data_tables/readings_table/per_instrument_padded/CRaTER-D1D2_readings_padded.csv"

# Lists to store all input data and labels
all_sdo_data = []
all_labels = []

# Iterate over each sample in the metadata
for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Loading SDO and radiation data"):
    input_window_start = pd.to_datetime(row['input_window_start'])
    input_window_end = pd.to_datetime(row['input_window_end'])
    target_window_start = pd.to_datetime(row['target_window_start'])
    target_window_end = pd.to_datetime(row['target_window_end'])

    # Load SDO data
    sdo_data = load_sdo_images(input_window_start, input_window_end, sdo_data_dir)
    all_sdo_data.append(sdo_data)

    # Load radiation dose data
    label_data = load_radiation_data(target_window_start, target_window_end, rad_dose_filepath)
    all_labels.append(label_data)

# Convert lists to numpy arrays
all_sdo_data = np.array(all_sdo_data, dtype=object)  # Use dtype=object to handle arrays of different shapes
all_labels = np.array(all_labels, dtype=object)  # Use dtype=object to handle arrays of different shapes

# Flatten all_labels to avoid multi-dimensional issues in train_test_split
all_labels_flat = np.array([np.mean(label) for label in all_labels])

# Split data into training, validation, and test sets
train_sdo, temp_sdo, train_label, temp_label = train_test_split(all_sdo_data, all_labels_flat, test_size=0.4, random_state=42, stratify=all_labels_flat)
val_sdo, test_sdo, val_label, test_label = train_test_split(temp_sdo, temp_label, test_size=0.5, random_state=42, stratify=temp_label)

# Convert split data back to numpy arrays (not object arrays)
train_sdo = np.array(train_sdo)
val_sdo = np.array(val_sdo)
test_sdo = np.array(test_sdo)
train_label = np.array(train_label)
val_label = np.array(val_label)
test_label = np.array(test_label)

# Save input SDO data to pickle files
train_sdo_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/train_sdo.pkl"
val_sdo_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/val_sdo.pkl"
test_sdo_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/test_sdo.pkl"

with open(train_sdo_path, 'wb') as file:
    pickle.dump(train_sdo, file)
with open(val_sdo_path, 'wb') as file:
    pickle.dump(val_sdo, file)
with open(test_sdo_path, 'wb') as file:
    pickle.dump(test_sdo, file)

# Save label data to pickle files
train_label_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/train_label.pkl"
val_label_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/val_label.pkl"
test_label_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/test_label.pkl"

with open(train_label_path, 'wb') as file:
    pickle.dump(train_label, file)
with open(val_label_path, 'wb') as file:
    pickle.dump(val_label, file)
with open(test_label_path, 'wb') as file:
    pickle.dump(test_label, file)

print(f"Train SDO data saved to {train_sdo_path}")
print(f"Validation SDO data saved to {val_sdo_path}")
print(f"Test SDO data saved to {test_sdo_path}")
print(f"Train labels saved to {train_label_path}")
print(f"Validation labels saved to {val_label_path}")
print(f"Test labels saved to {test_label_path}")

# Load data from pickle files
def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Load the datasets
train_sdo = load_data_from_pickle(train_sdo_path)
val_sdo = load_data_from_pickle(val_sdo_path)
train_label = load_data_from_pickle(train_label_path)
val_label = load_data_from_pickle(val_label_path)

# Reshape the data if necessary
train_sdo = np.array([sdo if sdo.size > 0 else np.zeros((10, 9, 512, 512)) for sdo in train_sdo])
val_sdo = np.array([sdo if sdo.size > 0 else np.zeros((10, 9, 512, 512)) for sdo in val_sdo])
train_sdo = torch.tensor(train_sdo, dtype=torch.float32)
val_sdo = torch.tensor(val_sdo, dtype=torch.float32)
train_label = torch.tensor(train_label, dtype=torch.float32).view(-1, 1)
val_label = torch.tensor(val_label, dtype=torch.float32).view(-1, 1)

# Create datasets and dataloaders
train_dataset = CustomDataset(train_sdo, train_label)
val_dataset = CustomDataset(val_sdo, val_label)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinTransformer3D(num_classes=1,  # regression
                          embed_dim=96,
                          mlp_ratio=4.0,
                          patch_norm=True,
                          patch_size=(2, 4, 4),
                          drop_path_rate=0.001,
                          num_heads=[4, 8, 16, 32],
                          pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_base_patch4_window7_512.pth',
                          pretrained2d=True,
                          window_size=(8, 7, 7))

model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # Use 4 GPUs
model = model.to(device)

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

model.apply(init_weights)

# Set up WandB
wandb.init(project='dust', entity='trilliumtechnologies')
wandb.config = {
    "batch_size": 8,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "num_frames": 10,
    "height": 512,
    "width": 512,
    "seed": 93728645
}

criterion = MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.02)
scheduler = CombinedLRScheduler(optimizer, num_epochs=100, end_of_linear=2.5)
scaler = GradScaler()

# Training and evaluation
logging.basicConfig(filename='./logs/step_3d.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

torch.manual_seed(93728645)
if torch.cuda.is_available():
    torch.cuda.manual_seed(93728645)
    torch.cuda.manual_seed_all(93728645)
np.random.seed(93728645)

outpath = './saved_models/step_3d/'
logpath = './logs/'
os.makedirs(outpath, exist_ok=True)
os.makedirs(logpath, exist_ok=True)

def evaluate_test_metrics(model, test_loader, criterion, epoch, outpath):
    model.eval()
    running_test_loss = 0.0
    running_mse = 0.0
    with torch.no_grad():
        for inputs_3d, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs_3d, labels = inputs_3d.float().to(device), labels.float().to(device)

            outputs = model(inputs_3d)
            test_loss = criterion(outputs, labels)
            running_test_loss += test_loss.item()
            
            mse = torch.mean((outputs - labels) ** 2).item()
            running_mse += mse

    average_test_loss = running_test_loss / len(test_loader)
    average_mse = running_mse / len(test_loader)
    average_rmse = math.sqrt(average_mse)

    torch.save(model.state_dict(), os.path.join(outpath, f'e{epoch}_{np.round(average_test_loss, 4)}.pth'))
    return average_test_loss, average_mse, average_rmse

# Training loop
logging.info('start training')
test_loss_list = []
test_mse_list = []
test_rmse_list = []
epoch_list = []

for epoch in range(100):
    logging.info(f'----- Epoch training start {epoch + 1}/100 -----')
    model.train()

    running_loss = 0.0
    early_stop_threshold = 0.02
    patience = 5
    best_loss = float('inf')
    counter = 0

    for i, (inputs_3d, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False)):
        if i % 50 == 0:
            logging.info('---- '+str(i)+' th batch -----')
        inputs_3d, labels = inputs_3d.float().to(device), labels.float().to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs_3d)

            if torch.isnan(outputs).any():
                print(f"NaN detected in outputs at epoch {epoch + 1}, batch {i}")
            if torch.isnan(labels).any():
                print(f"NaN detected in labels at epoch {epoch + 1}, batch {i}")

            loss = criterion(outputs, labels)
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

            if torch.isnan(loss):
                print(f"NaN detected in loss at epoch {epoch + 1}, batch {i}")
                print(f"Outputs: {outputs}")
                print(f"Labels: {labels}")

            if loss.item() < early_stop_threshold:
                print("Early stopping triggered.")
                break

            if loss.item() < best_loss:
                best_loss = loss.item()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered due to no improvement.")
                    break

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    epoch_list.append(epoch + 1)
    test_loss_item, test_mse_item, test_rmse_item = evaluate_test_metrics(model, val_loader, criterion, epoch + 1, outpath)
    logging.info(f"Epoch: {epoch + 1}/100, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}, Val MSE: {test_mse_item}, Val RMSE: {test_rmse_item}")
    print(f"Epoch: {epoch + 1}/100, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}, Val MSE: {test_mse_item}, Val RMSE: {test_rmse_item}")
    test_loss_list.append(test_loss_item)
    test_mse_list.append(test_mse_item)
    test_rmse_list.append(test_rmse_item)

    wandb.log({
        "train_loss": running_loss / len(train_loader),
        "val_loss": test_loss_item,
        "val_mse": test_mse_item,
        "val_rmse": test_rmse_item,
        "epoch": epoch + 1
    })

    scheduler.step()
    torch.cuda.empty_cache()
    gc.collect()

logging.info("Training completed!")
print("Training completed!")

torch.save(model.state_dict(), os.path.join(outpath, 'final_3d.pth'))
np.save(os.path.join(outpath, 'val_loss.npy'), np.array(test_loss_list))
np.save(os.path.join(outpath, 'val_mse.npy'), np.array(test_mse_list))
np.save(os.path.join(outpath, 'val_rmse.npy'), np.array(test_rmse_list))
logging.info(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
print(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
