import torch
import torch.nn as nn
from models.video_swin_transformer import SwinTransformer3D
from torch.utils.data import Dataset, DataLoader, TensorDataset

import cv2
import numpy as np
import pandas as pd
import random

import torch.optim as optim
from torch.nn import CrossEntropyLoss

import logging
import pickle

def get_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for i, (video_tensor, label) in enumerate(loader):
            # debug here
            video_tensor = video_tensor.view(-1, 3, 9, 512, 512)
            video_tensor = video_tensor.to(device)
            print(f"Video tensor shape: {video_tensor.shape}")
            output = model.get_vid_feature(video_tensor)
            print(f"Output shape: {output.shape}")
            features.append(output.cpu().numpy())
            print(f"Features shape: {np.concatenate(features).shape}")
            
            if (i + 1) % 5 == 0:
                logging.info(f'Processed {i + 1} batches')
    return np.concatenate(features)

def load_state_dict_with_mismatch(model, state_dict):
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
    for k, v in filtered_state_dict.items():
        model_state_dict[k] = v
    model.load_state_dict(model_state_dict, strict=False)

if __name__ == "__main__":
    
    # Hard-coded settings
    batch_size = 1
    seed = 93728645
    
    # Model settings
    path_3d_model = '/home/daisysong/saved_models/step_3d/final_3d.pth'
    num_frames = 7200 # TODO change it based on the actual number of frames later on# # Assuming 9 frames per video segment???
    height = 512
    width = 512

    # Input paths
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
    label_train_norm = (label_train - min_label) / (max_label - min_label)# this is the label normalization based one this one month data
    label_val_norm = (label_val - min_label) / (max_label - min_label)

    train_dataset = TensorDataset(video_train, label_train_norm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) # overfit to these local patterns.
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

    # Output paths
    outpath = '/home/daisysong/2024-HL-Virtual-Dosimeter/video_features'
    logpath = './logs/'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Configure logging
    logging.basicConfig(filename=logpath+'get_3d_features.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    logging.info('----- Define Data Loader -----')
    print('----- Define Data Loader -----')

    bsize = batch_size
    logging.info(f'batch size {bsize}')
    print(f'batch size {bsize}')
    
    logging.info(f'----- data path {path_video_train}')
    print(f'----- data path {path_video_train}')
  
    video_train = torch.from_numpy(pd.read_pickle(path_video_train).astype('float32'))
    video_val = torch.from_numpy(pd.read_pickle(path_video_val).astype('float32'))
    
    label_train = torch.from_numpy(pd.read_pickle(path_label_train).astype('long'))
    label_val = torch.from_numpy(pd.read_pickle(path_label_val).astype('long'))
    
    train_dataset = TensorDataset(video_train, label_train)
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)

    val_dataset = TensorDataset(video_val, label_val)
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False)

    logging.info('---- Define 3D Swin ----')
    print('---- Define 3D Swin ----')

    model = SwinTransformer3D(num_classes=1,
                              embed_dim=96,
                              drop_path_rate=0.1,
                              mlp_ratio=4.0,
                              patch_norm=True,
                              patch_size=(2, 4, 4),
                              pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_512.pth',
                              pretrained2d=True,
                              window_size=(8, 7, 7))

    #model.load_state_dict(torch.load(path_3d_model, map_location=device))
    state_dict = torch.load(path_3d_model, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    # Load state_dict with strict=False to ignore non-matching keys
    #model.load_state_dict(state_dict, strict=False)
    load_state_dict_with_mismatch(model, state_dict)
    model = model.to(device)
    
    logging.info('train loader')
    print('train loader')

    train_features = get_features(model, train_loader, device)
    with open(outpath+'train_3d_features.pkl', 'wb') as f:
        pickle.dump(train_features, f)
    logging.info('saved train features')
    print('saved train features')

    logging.info('val loader')
    print('val loader')

    val_features = get_features(model, val_loader, device)
    with open(outpath+'val_3d_features.pkl', 'wb') as f:
        pickle.dump(val_features, f)
    logging.info('saved val features')
    print('saved val features')

    
# Adjust the model definition:
# Ensure that the model architecture parameters (e.g., embed_dim, num_heads, patch_size, etc.) match those used during the checkpoint training.

# Manually load state_dict:
# Manually filter out the parameters that cause mismatches and load the rest.

# RuntimeError: Error(s) in loading state_dict for SwinTransformer3D:
#         size mismatch for layers.0.blocks.0.attn.relative_position_bias_table: copying a param with shape torch.Size([2535, 4]) from checkpoint, the shape in current model is torch.Size([2535, 3]).
#         size mismatch for layers.0.blocks.1.attn.relative_position_bias_table: copying a param with shape torch.Size([2535, 4]) from checkpoint, the shape in current model is torch.Size([2535, 3]).
#         size mismatch for layers.1.blocks.0.attn.relative_position_bias_table: copying a param with shape torch.Size([2535, 8]) from checkpoint, the shape in current model is torch.Size([2535, 6]).
#         size mismatch for layers.1.blocks.1.attn.relative_position_bias_table: copying a param with shape torch.Size([2535, 8]) from checkpoint, the shape in current model is torch.Size([2535, 6]).
#         size mismatch for layers.2.blocks.0.attn.relative_position_bias_table: copying a param with shape torch.Size([2535, 16]) from checkpoint, the shape in current model is torch.Size([2535, 12]).
#         size mismatch for layers.2.blocks.1.attn.relative_position_bias_table: copying a param with shape torch.Size([2535, 16]) from checkpoint, the shape in current model is torch.Size([2535, 12]).
#         size mismatch for layers.2.blocks.2.attn.relative_position_bias_table: copying a param with shape torch.Size([2535, 16]) from checkpoint, the shape in current model is torch.Size([2535, 12]).
#         size mismatch for layers.2.blocks.3.attn.relative_position_bias_table: copying a param with shape torch.Size([2535, 16]) from checkpoint, the shape in current model is torch.Size([2535, 12]).
#         size mismatch for layers.2.blocks.4.attn.relative_position_bias_table: copying a param with shape torch.Size([2535, 16]) from checkpoint, the shape in current model is torch.Size([2535, 12]).
#         size mismatch for layers.2.blocks.5.attn.relative_position_bias_table: copying a param with shape torch.Size([2535, 16]) from checkpoint, the shape in current model is torch.Size([2535, 12]).
#         size mismatch for layers.3.blocks.0.attn.relative_position_bias_table: copying a param with shape torch.Size([2535, 32]) from checkpoint, the shape in current model is torch.Size([2535, 24]).
#         size mismatch for layers.3.blocks.1.attn.relative_position_bias_table: copying a param with shape torch.Size([2535, 32]) from checkpoint, the shape in current model is torch.Size([2535, 24]).
# (myenv) (base) daisysong@deeplearning-xiaomei-gpu-vm:~$ 
