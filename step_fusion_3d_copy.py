#   File "/home/daisysong/2024-HL-Virtual-Dosimeter/src/models/video_swin_transformer.py", line 456, in forward
#     x = self.proj(x)  # B C D Wh Ww
#   File "/home/daisysong/fdl2024project/myenv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/home/daisysong/fdl2024project/myenv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/home/daisysong/fdl2024project/myenv/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 608, in forward
#     return self._conv_forward(input, self.weight, self.bias)
#   File "/home/daisysong/fdl2024project/myenv/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 603, in _conv_forward
#     return F.conv3d(
# RuntimeError: Given groups=1, weight of size [96, 3, 2, 4, 4], expected input[1, 10, 10, 512, 512] to have 3 channels, but got 10 channels instead

import torch
import torch.nn as nn
from models.video_swin_transformer import SwinTransformer3D
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import logging
import math
import gc
import timm
import wandb
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from torch.nn import MSELoss
from torch.optim.lr_scheduler import _LRScheduler
import pandas as pd

class SwinTransformer3DFineTuner:
    def __init__(self, num_classes, embed_dim, mlp_ratio, patch_norm, patch_size, drop_path_rate, num_heads, pretrained, pretrained2d, window_size, in_channels=10):
        super(SwinTransformer3D, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_norm = patch_norm
        self.patch_size = patch_size
        self.drop_path_rate = drop_path_rate
        self.num_heads = num_heads
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.window_size = window_size
        
        self.patch_embed = PatchEmbed3D(
            patch_size=self.patch_size,
            in_channels=in_channels,  # Update this to in_channels
            embed_dim=self.embed_dim,
            norm_layer=nn.LayerNorm if self.patch_norm else None
        )

        self.freeze_all_layers()
        self.unfreeze_selected_layers()

    def freeze_all_layers(self):
        """Freeze all layers in the model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_selected_layers(self):
        """Unfreeze the first and last layers."""
        if self.fine_tune_first_layer:
            for param in self.model.patch_embed.parameters():
                param.requires_grad = True

        if self.fine_tune_last_layer:
            for param in self.model.head.parameters():
                param.requires_grad = True

    def get_parameters(self):
        """Get model parameters that require gradients."""
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def get_optimizer(self, lr=1e-4):
        """Get an optimizer for the fine-tuned layers."""
        return torch.optim.Adam(self.get_parameters(), lr=lr)

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

def load_tensor_from_pickle(file_path):
    """Load a .kpl file and return it as a tensor."""
    with open(file_path, 'rb') as f:
        array = np.frombuffer(f.read(), dtype=np.float32)
    
    tensor = torch.from_numpy(array.copy())
    print(tensor.shape)
    print("Reshaped tensor (first 5 elements):")
    print(tensor[:5])
    return tensor

def load_data(directory, max_files=3):
    """Load up to max_files pickle files from a directory and concatenate them into a single tensor."""
    tensors = []
    files_loaded = 0

    for file_name in sorted(os.listdir(directory)):
        if files_loaded >= max_files:
            break
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path) or file_name.endswith('.pkl'):
            tensor = load_tensor_from_pickle(file_path)
            tensors.append(tensor)
            files_loaded += 1

    if tensors:
        concatenated_tensor = torch.cat(tensors, dim=0)  # Concatenate along the batch dimension
        print(f"Loaded {files_loaded} files.")
        return concatenated_tensor
    else:
        print("No files loaded.")
        return None

if __name__ == "__main__":
    # Clear GPU cache
    torch.cuda.empty_cache()

    # Initialize wandb
    wandb.init(project='dust', entity='trilliumtechnologies')

    # Training settings
    batch_size = 1  # Based on your input size example
    num_epochs = 100
    lr = 1e-4  # Reduced learning rate
    seed = 93728645

    # Model
    num_frames = 7200  # Number of frames in each video clip
    #num_channels = 9  # Number of channels in each frame
    height = 512  # Height of each frame
    width = 512  # Width of each frame

    # Paths
    path_video_train = '/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/train_batches_X_sdoml'
    path_label_train = '/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/train_batches_y'
    path_video_val = '/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/val_batches_X_sdoml'
    path_label_val = '/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/val_batches_y'

    # Define expected shapes
    video_shape = (4, 10, 9, 512, 512)
    label_shape = (4)  # Adjust this shape according to your actual data

    # Load the data
    video_train = load_data(path_video_train)
    video_val = load_data(path_video_val)
    label_train = load_data(path_label_train)
    label_val = load_data(path_label_val)

    print(f"Loaded video_train: {video_train.shape}")
    print(f"Loaded label_train: {label_train.shape}")
    print(f"Loaded video_val: {video_val.shape}")
    print(f"Loaded label_val: {label_val.shape}")

    # Reshape the data
    video_train = video_train.view(-1, 10, 9, 512, 512)
    video_val = video_val.view(-1, 10, 9, 512, 512)
    label_train = label_train.view(-1, 10)
    label_val = label_val.view(-1, 10)

    print(f"Reshaped video_train: {video_train.shape}")
    print(f"Reshaped label_train: {label_train.shape}")
    print(f"Reshaped video_val: {video_val.shape}")
    print(f"Reshaped label_val: {label_val.shape}")

    print("Reshaped label_val tensor (first 5 elements):")
    print(label_val[:5])

    assert not torch.isnan(video_train).any(), "Training data contains NaN values"
    assert not torch.isinf(video_train).any(), "Training data contains Inf values"
    assert not torch.isnan(label_train).any(), "Training labels contain NaN values"
    assert not torch.isinf(label_train).any(), "Training labels contain Inf values"

    # Normalize the labels
    min_label = label_train.min()
    max_label = label_train.max()

    label_train_norm = (label_train - min_label) / (max_label - min_label)
    label_val_norm = (label_val - min_label) / (max_label - min_label)

    print(f"video_train shape: {video_train.shape}")
    print(f"label_train_norm shape: {label_train_norm.shape}")
    print(f"video_val shape: {video_val.shape}")
    print(f"label_val_norm shape: {label_val_norm.shape}")

    # assert video_train.shape[0] == label_train_norm.shape[0], "Mismatch in video_train and label_train_norm dimensions"
    # assert video_val.shape[0] == label_val_norm.shape[0], "Mismatch in video_val and label_val_norm dimensions"

    train_dataset = TensorDataset(video_train, label_train_norm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = TensorDataset(video_val, label_val_norm)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    outpath = './saved_models/step_3d/'
    logpath = './logs/'

    os.makedirs(outpath, exist_ok=True)
    os.makedirs(logpath, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(filename=os.path.join(logpath, 'step_3d.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = SwinTransformer3D(num_classes=1, # regression
                              embed_dim=96,
                              mlp_ratio=4.0,
                              patch_norm=True,
                              patch_size=(2,4,4),
                              drop_path_rate=0.001,
                              num_heads=[4,8,16,32],
                              pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_base_patch4_window7_512.pth',
                              pretrained2d=True,
                              window_size=(8,7,7))

    model = nn.DataParallel(model)
    model = model.to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    model.apply(init_weights)

    wandb.config = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "seed": seed
    }

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
        logging.info(f'----- Epoch training start {epoch + 1}/{num_epochs} -----')
        model.train()

        running_loss = 0.0
        early_stop_threshold = 0.02
        patience = 5
        best_loss = float('inf')
        counter = 0

        for i, (inputs_3d, labels) in enumerate(train_loader, 0):
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
        logging.info(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}, Val MSE: {test_mse_item}, Val RMSE: {test_rmse_item}")
        print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}, Val MSE: {test_mse_item}, Val RMSE: {test_rmse_item}")
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


# def load_kpl_file(file_path):
#     """Load a .kpl file and return it as a reshaped tensor."""
#     with open(file_path, 'rb') as f:
#         array = np.frombuffer(f.read(), dtype=np.float32)
    
#     # # Ensure the total number of elements matches the expected shape
#     # if array.size != np.prod(expected_shape):
#     #     raise ValueError(f"Shape {expected_shape} is invalid for input of size {array.size}")

#     tensor = torch.from_numpy(array.copy())
#     #tensor = tensor.view(expected_shape)
#     #tensor = tensor.permute(0, 1, 4, 2, 3)  # Reorder dimensions to (batch, time, channels, height, width)
#     print(tensor.shape)
#     return tensor

# def load_data(directory, max_files=3):
#     """Load all .kpl files from a directory and concatenate them into a single tensor."""
#     tensors = []
#     for file_name in sorted(os.listdir(directory)):
#         file_path = os.path.join(directory, file_name)
#         if os.path.isfile(file_path) and file_name.endswith('.kpl'):
#             tensor = load_kpl_file(file_path)
#             tensors.append(tensor)
#     tensor = torch.cat(tensors, dim=0) # Concatenate along the batch dimension
#     return tensor  

# def load_tensor_from_pickle(file_path):
#     """Load a tensor from a pickle file."""
#     df = pd.read_pickle(file_path)
#     tensor = torch.tensor(df.values).float()
#     return tensor