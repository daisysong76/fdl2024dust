import torch
import torch.nn as nn
from models.SwinTransformer_cls import SwinTransformer_1D
from models.SwinTransformer_fusion import SwinTransformer_1D as SwinTransformer_1D_fusion
from torch.utils.data import Dataset, DataLoader, TensorDataset

import cv2
import numpy as np
import pandas as pd
import random

import torch.optim as optim
from torch.nn import CrossEntropyLoss

import logging

from torch.optim.lr_scheduler import _LRScheduler
import math
import argparse


def evaluate_test_loss(model, test_loader, criterion, epoch, outpath):
    model.eval()
    running_test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs_1d, labels, vid_feat in test_loader:
            inputs_1d, labels, vid_feat = inputs_1d.to(device), labels.to(device), vid_feat.to(device)

            outputs = model(inputs_1d, vid_feat)
            test_loss = criterion(outputs, labels)
            running_test_loss += test_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_test_loss = running_test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    torch.save(model.state_dict(), outpath+'e'+str(epoch)+'_'+str(np.round(test_accuracy,4))+'.pth')
    return average_test_loss, test_accuracy
    
if __name__ == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser(description='Train 1d Swin Transformer')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='input batch size for training (default: 2)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--seed', type=int, default=93728645,
                        help='random seed (default: 93728645)')
    parser.add_argument('--print_every', type=int, default=1,
                        help='epoch interval for validation and model saving (e.g., 1 for every 1 epoch).')

    # Model
    parser.add_argument('--path_1d_model', type=str, default='/home/daisysong/2024-HL-Virtual-Dosimeter/saved_models/step_1d/final_1d.pth',
                        help='Path to pre-trained swin 3d model (default: /home/daisysong/2024-HL-Virtual-Dosimeter/saved_models/step_1d/final_1d.pth')
    parser.add_argument('--num_frames', type=int, default=4,
                        help='number of frames (time points)')
    parser.add_argument('--num_channels', type=int, default=1,
                        help='number of channels (default: 3)')

    # Inputs
    # radiation dose data input in the input region, TODO Q: are they same as kine data in step_fusion_1d.py? should use new data?
    parser.add_argument('--path_kine_train', type=str, default='/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/train_image.pkl',
                        help='Path to the kine training data (default: /home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/train_image.pkl')
    parser.add_argument('--path_kine_val', type=str, default='/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/val_image.pkl',
                        help='Path to the kine validation data (default: /home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/val_image.pkl')
    # feature data saved from 3D swin model (video features)
    parser.add_argument('--path_vidfeat_train', type=str, default='/home/daisysong/2024-HL-Virtual-Dosimeter/video_features/video_featurestrain_3d_features.pkl',
                        help='Path to the training video feature (default: /home/daisysong/2024-HL-Virtual-Dosimeter/video_features/video_featurestrain_3d_features.pkl')
    parser.add_argument('--path_vidfeat_val', type=str, default='/home/daisysong/2024-HL-Virtual-Dosimeter/video_features/video_featuresval_3d.py',
                        help='Path to the validation video feature (default: /home/daisysong/2024-HL-Virtual-Dosimeter/video_features/video_featuresval_3d.py')
    # label data radiation dose in the target region
    parser.add_argument('--path_label_train', type=str, default='/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/train_label.pkl',
                        help='Path to training label (default: /home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/train_label.pkl')
    parser.add_argument('--path_label_val', type=str, default='/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/val_label.pkl',
                        help='Path to validation label (default: /home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/val_label.pkl')
    
    # Outputs
    parser.add_argument('--outpath', type=str, default='./saved_models/step_fusion/',
                        help='where to save models')
    parser.add_argument('--logpath', type=str, default='./logs/',
                        help='where to save logs')

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outpath = args.outpath
    
    
    # Configure logging
    logging.basicConfig(filename=args.logpath+'step_fusion.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    logging.info('----- Define Data Loader -----')
    print('----- Define Data Loader -----')

    bsize=args.batch_size
    seq_len = args.num_frames
    num_chl = args.num_channels
    logging.info(f'batch size {bsize}')
    print(f'batch size {bsize}')
      
    X_train = torch.from_numpy(pd.read_pickle(args.path_kine_train).astype('float32'))
    X_val = torch.from_numpy(pd.read_pickle(args.path_kine_val).astype('float32'))
    
    X_add_train = torch.from_numpy(pd.read_pickle(args.path_vidfeat_train).astype('float32'))
    X_add_val = torch.from_numpy(pd.read_pickle(args.path_vidfeat_val).astype('float32'))

    label_train = torch.from_numpy(pd.read_pickle(args.path_label_train))
    label_val = torch.from_numpy(pd.read_pickle(args.path_label_val))
    
    train_dataset = TensorDataset(X_train, label_train, X_add_train)
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)

    val_dataset = TensorDataset(X_val, label_val, X_add_val)
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False)
    

    logging.info('----- Define 1D swin -----')
    print('----- Define 1D swin -----')

    model = SwinTransformer_1D_fusion(seq_len=seq_len, 
                                        in_chans=num_chl, 
                                        num_classes=1,
                                        window_size=8, 
                                        # drop_rate=0.3, 
                                        patch_size=1,
                                        num_heads=[16, 16, 16, 16], 
                                        depths=[2, 2, 6, 2],
                                        embed_dim=128)

    loaded_state_dict = torch.load(args.path_1d_model)
    partial_state_dict = {key: value for key, value in loaded_state_dict.items() if not key.startswith('head')}
    
    model.load_state_dict(partial_state_dict, strict=False)
    
    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_epochs = args.num_epochs
    print_every = args.print_every
    lr = args.lr
    logging.info(f'learning rate = {lr}')
    print(f'learning rate = {lr}')
    
    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999), weight_decay=0.02)    
    
    logging.info('start training')
    print('start training')

    test_loss_list = []
    test_acc_list = []
    epoch_list = []
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0

        for i, (inputs_1d, labels, vid_feat) in enumerate(train_loader, 0):

            inputs_1d, labels, vid_feat = inputs_1d.to(device), labels.to(device), vid_feat.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs_1d, vid_feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % print_every == 0:
            epoch_list.append(epoch+1)
            test_loss_item, test_acc_item = evaluate_test_loss(model, val_loader, criterion, epoch+1, outpath)
            logging.info(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}, Val Acc: {test_acc_item}")
            print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}, Val Acc: {test_acc_item}")
            test_loss_list.append(test_loss_item)
            test_acc_list.append(test_acc_item)


    logging.info("Training completed!")
    print("Training completed!")
    
    torch.save(model.state_dict(), outpath+'final_fusion.pth')
    np.save(outpath+'val_loss.npy', np.array(test_loss_list))
    np.save(outpath+'val_acc.npy', np.array(test_acc_list))
    logging.info(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
    print(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
    
    
