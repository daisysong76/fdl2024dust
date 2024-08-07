import os
import numpy as np
import pandas as pd
import torch
import pickle
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from models.video_swin_transformer import SwinTransformer3D
import logging
import gc
import wandb
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from torch.nn import MSELoss
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import math

class SwinTransformer3DFineTuner:
    def __init__(self, model, fine_tune_first_layer=True, fine_tune_last_layer=True):
        self.model = model
        self.fine_tune_first_layer = fine_tune_first_layer
        self.fine_tune_last_layer = fine_tune_last_layer
        self.norm_layer = nn.InstanceNorm3d(9)
        self.freeze_all_layers()
        self.unfreeze_selected_layers()

    def freeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_selected_layers(self):
        if self.fine_tune_first_layer:
            for param in self.model.patch_embed.parameters():
                param.requires_grad = True
        if self.fine_tune_last_layer:
            for param in self.model.head.parameters():
                param.requires_grad = True

    def get_parameters(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def get_optimizer(self, lr=1e-4):
        return torch.optim.Adam(self.get_parameters(), lr=lr)

class CombinedLRScheduler(_LRScheduler):
    def __init__(self, optimizer, num_epochs, end_of_linear=2.5, eta_min=0, last_epoch=-1):
        self.num_epochs = num_epochs
        self.end_of_linear = end_of_linear
        self.eta_min = eta_min
        super(CombinedLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.end_of_linear:
            factor = 1 - self.last_epoch / self.end_of_linear
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            T_max = self.num_epochs - self.end_of_linear
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * (self.last_epoch - self.end_of_linear) / T_max)) / 2
                    for base_lr in self.base_lrs]

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

def load_data_from_pickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def evaluate_test_metrics(model, test_loader, criterion, epoch, outpath, device):
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

def main():
    # Clear GPU cache
    torch.cuda.empty_cache()

    # Initialize wandb
    wandb.init(project='dust', entity='trilliumtechnologies')

    # Training settings
    batch_size = 1  # Reduced batch size to fit into memory
    num_epochs = 100
    lr = 1e-4  # Reduced learning rate
    seed = 93728645

    # Model
    num_frames = 7200  # Number of frames (time points) in each video clip 30*24*4
    height = 512  # Height of each frame after resizing
    width = 512  # Width of each frame after resizing

    train_sdo_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/train_sdo.pkl"
    val_sdo_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/val_sdo.pkl"
    test_sdo_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/test_sdo.pkl"
    train_label_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/train_label.pkl"
    val_label_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/val_label.pkl"
    test_label_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/test_label.pkl"

    train_sdo = (4, 10, 9, 512, 512)  # Adjust this shape according to your actual data
    val_label_shape = (4, 12)  # Adjust this shape according to your actual data

    # torch.from_numpy????
    train_sdo = load_data_from_pickle(train_sdo_path)
    val_sdo = load_data_from_pickle(val_sdo_path)
    test_sdo = load_data_from_pickle(test_sdo_path)
    train_label = load_data_from_pickle(train_label_path)
    val_label = load_data_from_pickle(val_label_path)
    test_label = load_data_from_pickle(test_label_path)
    #
    train_sdo = np.array([sdo if sdo.size > 0 else np.zeros((10, 9, 512, 512)) for sdo in train_sdo])
    val_sdo = np.array([sdo if sdo.size > 0 else np.zeros((10, 9, 512, 512)) for sdo in val_sdo])
    test_sdo = np.array([sdo if sdo.size > 0 else np.zeros((10, 9, 512, 512)) for sdo in test_sdo])

    train_sdo = torch.tensor(train_sdo, dtype=torch.float32)
    val_sdo = torch.tensor(val_sdo, dtype=torch.float32)
    test_sdo = torch.tensor(test_sdo, dtype=torch.float32)
    train_label = torch.tensor(train_label, dtype=torch.float32).view(-1, 1)
    val_label = torch.tensor(val_label, dtype=torch.float32).view(-1, 1)
    test_label = torch.tensor(test_label, dtype=torch.float32).view(-1, 1)

    # Normalize the labels to [0, 1]
    min_label = train_label.min(dim=0)[0]
    max_label = train_label.max(dim=0)[0]
    label_train_norm = (train_label - min_label) / (max_label - min_label)
    label_val_norm = (val_label - min_label) / (max_label - min_label)
    label_test_norm = (test_label - min_label) / (max_label - min_label)

    # Standardize the labels to have mean 0 and standard deviation 1
    mean_label = train_label.mean(dim=0)
    std_label = train_label.std(dim=0)
    label_train_std = (train_label - mean_label) / std_label
    label_val_std = (val_label - mean_label) / std_label
    label_test_std = (test_label - mean_label) / std_label

    # Use normalized or standardized labels
    use_standardization = True  # Set to False to use min-max normalization

    if use_standardization:
        train_label = label_train_std
        val_label = label_val_std
        test_label = label_test_std
    else:
        train_label = label_train_norm
        val_label = label_val_norm
        test_label = label_test_norm

    train_dataset = CustomDataset(train_sdo, train_label)
    val_dataset = CustomDataset(val_sdo, val_label)
    test_dataset = CustomDataset(test_sdo, test_label)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinTransformer3D(num_classes=1,
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

    logging.basicConfig(filename='./logs/step_3d_1m.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    torch.manual_seed(93728645)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(93728645)
        torch.cuda.manual_seed_all(93728645)
    np.random.seed(93728645)

    outpath = './saved_models/step_3d/'
    logpath = './logs/'
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(logpath, exist_ok=True)

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
        test_loss_item, test_mse_item, test_rmse_item = evaluate_test_metrics(model, val_loader, criterion, epoch + 1, outpath, device)
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

    torch.save(model.state_dict(), os.path.join(outpath, 'final_3d_1m.pth'))
    np.save(os.path.join(outpath, 'val_loss_1m.npy'), np.array(test_loss_list))
    np.save(os.path.join(outpath, 'val_mse_1m.npy'), np.array(test_mse_list))
    np.save(os.path.join(outpath, 'val_rmse_1m.npy'), np.array(test_rmse_list))
    logging.info(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
    print(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')

if __name__ == "__main__":
    main()

