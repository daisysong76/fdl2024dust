import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.nn import MSELoss
import logging
from torch.optim.lr_scheduler import _LRScheduler
import math
import os
import wandb
import pickle
from torch.cuda.amp import GradScaler, autocast
import gc

# Example SlowFast instantiation
model = torchvision.models.video.slowfast_r50(pretrained=True)

class CombinedLRScheduler(_LRScheduler):
    def __init__(self, optimizer, num_epochs, end_of_linear=2.5, eta_min=0, last_epoch=-1):
        self.num_epochs = num_epochs
        self.end_of_linear = end_of_linear
        self.eta_min = eta_min
        super(CombinedLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.end_of_linear:
            # Linear decay
            factor = 1 - self.last_epoch / self.end_of_linear
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            T_max = self.num_epochs - self.end_of_linear
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * (self.last_epoch - self.end_of_linear) / T_max)) / 2
                    for base_lr in self.base_lrs]

def evaluate_test_metrics(model, test_loader, criterion, epoch, outpath):
    model.eval()
    running_test_loss = 0.0
    running_mse = 0.0
    with torch.no_grad():
        for inputs_3d, labels in test_loader:
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

def load_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    # Add debug statements to inspect the structure of data
    print(f"Data type: {type(data)}")
    if isinstance(data, list):
        print(f"List length: {len(data)}")
        print(f"First element type: {type(data[0])}")
        if isinstance(data[0], torch.Tensor):
            print(f"First element shape: {data[0].shape}")
        return torch.cat(data, dim=0)  # Concatenate the list of tensors along the first dimension
    elif isinstance(data, np.ndarray):
        print(f"Array shape: {data.shape}")
    return torch.tensor(data)  # Ensure data is returned as tensor

if __name__ == "__main__":
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

    # Inputs
    path_video_train = '/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/train_image.pkl'
    path_video_val = '/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/val_image.pkl'
    path_label_train = '/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/train_label.pkl'
    path_label_val = '/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/val_label.pkl'

    video_shape = (4, 3, 9, 512, 512)  # Adjust this shape according to your actual data
    label_shape = (4, 12)  # Adjust this shape according to your actual data

    # Load the data
    video_train = torch.from_numpy(pd.read_pickle(path_video_train))
    video_val = torch.from_numpy(pd.read_pickle(path_video_val))
    label_train = torch.from_numpy(pd.read_pickle(path_label_train))
    label_val = torch.from_numpy(pd.read_pickle(path_label_val))

    # Reshape the data
    video_train = video_train.view(-1, 3, 9, 512, 512)
    video_val = video_val.view(-1, 3, 9, 512, 512)
    label_train = label_train.view(-1, 12)
    label_val = label_val.view(-1, 12)

    print(f"Reshaped video_train: {video_train.shape}")
    print(f"Reshaped label_train: {label_train.shape}")
    print(f"Reshaped video_val: {video_val.shape}")
    print(f"Reshaped label_val: {label_val.shape}")

    assert not torch.isnan(video_train).any(), "Training data contains NaN values"
    assert not torch.isinf(video_train).any(), "Training data contains Inf values"
    assert not torch.isnan(label_train).any(), "Training labels contain NaN values"
    assert not torch.isinf(label_train).any(), "Training labels contain Inf values"

    # Check if sizes match
    assert video_train.size(0) == label_train.size(0), "Size mismatch between video_train and label_train"
    assert video_val.size(0) == label_val.size(0), "Size mismatch between video_val and label_val"

   # Compute the min and max of the training labels
    min_label = label_train.min(dim=0)[0]
    max_label = label_train.max(dim=0)[0]

    # Normalize the labels to [0, 1]
    label_train_norm = (label_train - min_label) / (max_label - min_label)
    label_val_norm = (label_val - min_label) / (max_label - min_label)

    train_dataset = TensorDataset(video_train, label_train_norm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = TensorDataset(video_val, label_val_norm)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

      # Check for NaN in inputs
    for i, (inputs_3d, labels) in enumerate(train_loader):
        if torch.isnan(inputs_3d).any():
            print(f"NaN detected in inputs_3d at epoch {epoch + 1}, batch {i}")
        if torch.isnan(labels).any():
            print(f"NaN detected in inputs_3d at epoch {epoch + 1}, batch {i}")

    print(f"No NaN detected")
    #import pdb; pdb.set_trace()
    # Outputs
    outpath = './saved_models/step_3d/'  # Where to save models
    logpath = './logs/'

    # Ensure the directories exist
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(logpath, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure logging
    logging.basicConfig(filename=os.path.join(logpath, 'step_3d.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    logging.info('----- Define Data Loader -----')
    print('----- Define Data Loader -----')

    logging.info(f'batch size {batch_size}')
    print(f'batch size {batch_size}')
    
    logging.info(f'----- data path {path_video_train}')
    print(f'----- data path {path_video_train}')

    logging.info('----- Define 3D SlowFast -----')
    print('----- Define 3D SlowFast -----')

    model = torchvision.models.video.slowfast_r50(pretrained=True)
    
    # Modify the final layer to match the number of output classes
    model.blocks[6] = nn.Linear(model.blocks[6].in_features, 12)

    # Wrap the model with DataParallel to use multiple GPUs
    model = nn.DataParallel(model)
    model = model.to(device)

    logging.info(f'learning rate = {lr}')
    print(f'learning rate = {lr}')

    # Initialize weight initialization
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):#
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    model.apply(init_weights)

    # Log hyperparameters to wandb
    wandb.config = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "seed": seed
    }
    
    # Set loss function and optimizer
    criterion = MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.02)
    scheduler = CombinedLRScheduler(optimizer, num_epochs=num_epochs, end_of_linear=2.5)
    scaler = GradScaler()

    logging.info('start training')
    test_loss_list = []
    test_mse_list = []
    test_rmse_list = []
    epoch_list = []

    for epoch in range(num_epochs):
        logging.info(f'----- Epoch training start{epoch + 1}/{num_epochs} -----')
        model.train()

        running_loss = 0.0
        # Early stopping parameters
        early_stop_threshold = 0.02
        patience = 5
        best_loss = float('inf')
        counter = 0

        for i, (inputs_3d, labels) in enumerate(train_loader, 0):
            #print({inputs_3d.shape})
            print(inputs_3d[:5])  # Print the first 5 elements
            #print({labels.shape})
            #print(labels[:5])  # Print the first 5 elements
            if torch.isnan(inputs_3d).any():
                print(f"NaN detected in inputs")
            if torch.isnan(labels).any():
                print(f"NaN detected in labels")

            if i % 50 == 0:
                logging.info('---- '+str(i)+' th batch -----')
            inputs_3d, labels = inputs_3d.float().to(device), labels.float().to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs_3d)

                # Debugging prints for inputs, outputs, and loss
                if torch.isnan(outputs).any():
                    print(f"NaN detected in outputs at epoch {epoch + 1}, batch {i}")
                if torch.isnan(labels).any():
                    print(f"NaN detected in labels at epoch {epoch + 1}, batch {i}")

                loss = criterion(outputs, labels)
                #print(f"Loss: {loss}")
                # Print the loss for monitoring
                print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

                # Debugging print for loss
                if torch.isnan(loss):
                    print(f"NaN detected in loss at epoch {epoch + 1}, batch {i}")
                    print(f"Outputs: {outputs}")
                    print(f"Labels: {labels}")
                
                # Early stopping condition
                if loss.item() < early_stop_threshold:
                    print("Early stopping triggered.")
                    break

                # Track the best loss and implement patience
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping triggered due to no improvement.")
                        break

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # Evaluate and print metrics every epoch
        epoch_list.append(epoch + 1)
        test_loss_item, test_mse_item, test_rmse_item = evaluate_test_metrics(model, val_loader, criterion, epoch + 1, outpath)
        logging.info(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}, Val MSE: {test_mse_item}, Val RMSE: {test_rmse_item}")
        print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}, Val MSE: {test_mse_item}, Val RMSE: {test_rmse_item}")
        test_loss_list.append(test_loss_item)
        test_mse_list.append(test_mse_item)
        test_rmse_list.append(test_rmse_item)

        # Log metrics to wandb
        wandb.log({
            "train_loss": running_loss / len(train_loader),
            "val_loss": test_loss_item,
            "val_mse": test_mse_item,
            "val_rmse": test_rmse_item,
            "epoch": epoch + 1
        })

        scheduler.step()

        # Empty the cache
        torch.cuda.empty_cache()
        # Collect garbage
        gc.collect()
    
    logging.info("Training completed!")
    print("Training completed!")
    
    torch.save(model.state_dict(), os.path.join(outpath, 'final_3d.pth'))
    np.save(os.path.join(outpath, 'val_loss.npy'), np.array(test_loss_list))
    np.save(os.path.join(outpath, 'val_mse.npy'), np.array(test_mse_list))
    np.save(os.path.join(outpath, 'val_rmse.npy'), np.array(test_rmse_list))
    logging.info(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
    print(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
