from transformers import get_scheduler
from torch.optim import AdamW
from lora import LoRALayer  # Import your LoRA implementation
from video_swin_transformer import SwinTransformer3D  # Import your Swin Transformer model


class TimeSeriesModel(nn.Module):
    def __init__(self, ...):
        super(TimeSeriesModel, self).__init__()
        self.swin_transformer = SwinTransformer1D(...)  # Replace with your actual model
        self.lora_layer = LoRALayer(in_features=256, out_features=256)  # Adjust features as per your model

    def forward(self, x):
        x = self.swin_transformer(x)
        x = self.lora_layer(x)  # Apply LoRA Layer
        return x

optimizer = AdamW(model.parameters(), lr=args.lr)
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.num_training_steps,
)



import torch  
import torch.nn as nn 
from models.SwinTransformer_cls import SwinTransformer_1D  
from torch.utils.data import Dataset, DataLoader, TensorDataset  

import cv2  
import numpy as np 
import pandas as pd  
import random 

import torch.optim as optim  
from torch.nn import CrossEntropyLoss  

import logging 

from torch.optim.lr_scheduler import _LRScheduler  # Import the learning rate scheduler from PyTorch
import math  
import argparse  


def evaluate_test_loss(model, test_loader, criterion, epoch, outpath):
    model.eval()  # Set the model to evaluation mode
    running_test_loss = 0.0  # Initialize the running test loss
    correct = 0  # Initialize the correct predictions count
    total = 0  
    with torch.no_grad():  # Disable gradient computation
        for inputs_1d, labels in test_loader:  # Iterate over the test loader
            inputs_1d, labels = inputs_1d.to(device), labels.to(device)  # Move inputs and labels to the device

            outputs = model(inputs_1d) 
            test_loss = criterion(outputs, labels) 
            running_test_loss += test_loss.item()  # Accumulate the test loss

            _, predicted = torch.max(outputs.data, 1)  # Get the predicted labels
            total += labels.size(0)  # Update the total number of samples
            correct += (predicted == labels).sum().item()  # Update the correct predictions count

    average_test_loss = running_test_loss / len(test_loader)  
    test_accuracy = 100 * correct / total  
    torch.save(model.state_dict(), outpath+'e'+str(epoch)+'_'+str(np.round(test_accuracy,4))+'.pth')  # Save the model state
    return average_test_loss, test_accuracy  # Return the average test loss and test accuracy
    
if __name__ == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser(description='Train 1d Swin Transformer') 
    parser.add_argument('--batch_size', type=int, default=2000,  # Add batch size 
                        help='input batch size for training (default: 2000)')
    parser.add_argument('--num_epochs', type=int, default=200,  # Add number of epochs argument
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0003,  # Add learning 
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--seed', type=int, default=93728645,  # Add random seed 
                        help='random seed (default: 93728645)')
    parser.add_argument('--print_every', type=int, default=1,  # Add print frequency 
                        help='epoch interval for validation and model saving (e.g., 1 for every 1 epoch).')

    # Model
    parser.add_argument('--num_frames', type=int, default=51,  # Add number of frames argument
                        help='number of frames (time points)')
    # TODO: change the number of channels argument
    parser.add_argument('--num_channels', type=int, default=3,   
                        help='number of channels (default: 3)')

    # Inputs
    parser.add_argument('--path_kine_train', type=str, default='./data/kine_train.pkl',  # Add path to training data
                        help='Path to the kine training data (default: ./data/kine_train.pkl')
    parser.add_argument('--path_kine_val', type=str, default='./data/kine_val.pkl',  # Add path to validation data
                        help='Path to the kine validation data (default: ./data/kine_val.pkl')
    
    parser.add_argument('--path_label_train', type=str, default='./data/label_train.pkl',  # Add path to training labels
                        help='Path to training label (default: ./data/label_train.pkl')
    parser.add_argument('--path_label_val', type=str, default='./data/label_val.pkl',  # Add path to validation labels
                        help='Path to validation label (default: ./data/label_val.pkl')
    
    # Outputs
    parser.add_argument('--outpath', type=str, default='./saved_models/step_1d/',  # Add path to save models
                        help='where to save models')
    parser.add_argument('--logpath', type=str, default='./logs/',  # Add path to save logs
                        help='where to save logs')

    args = parser.parse_args()  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    outpath = args.outpath  # Get output path 
    
   
    logging.basicConfig(filename=args.logpath+'step_1d.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    seed = args.seed  
    torch.manual_seed(seed)  # Set PyTorch seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set CUDA seed if GPU is available
        torch.cuda.manual_seed_all(seed)  # Set CUDA seed for all devices if GPU is available
    np.random.seed(seed)  # Set NumPy seed
    
    logging.info('----- Define Data Loader -----')  # Log defining data loader
    print('----- Define Data Loader -----')  # Print defining data loader
    bsize = args.batch_size  # Get batch size from arguments
    seq_len = args.num_frames  # Get number of frames from arguments
    num_chl = args.num_channels  # Get number of channels from arguments
    logging.info(f'batch size {bsize}')  # Log batch size
    print(f'batch size {bsize}')  # Print batch size

    X_train = torch.from_numpy(pd.read_pickle(args.path_kine_train).astype('float32'))  # Load training data
    X_val = torch.from_numpy(pd.read_pickle(args.path_kine_val).astype('float32'))  # Load validation data
    label_train = torch.from_numpy(pd.read_pickle(args.path_label_train))  # Load training labels
    label_val = torch.from_numpy(pd.read_pickle(args.path_label_val))  # Load validation labels
    
    train_dataset = TensorDataset(X_train, label_train)  # Create training dataset
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)  # Create training data loader

    val_dataset = TensorDataset(X_val, label_val)  # Create validation dataset
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False)  # Create validation data loader
    
    logging.info('----- Define 1D swin -----')  # Log defining 1D Swin Transformer
    print('----- Define 1D swin -----')  # Print defining 1D Swin Transformer

    model = SwinTransformer_1D(seq_len=seq_len,  # Create Swin Transformer model
                                 in_chans=num_chl, 
                                 num_classes=4,
                                 window_size=8, 
                                 drop_path_rate=0.1,  # stochastic depth rate in the paper
                                 patch_size=1,
                                 num_heads=[16, 16, 16, 16], 
                                 depths=[2, 2, 6, 2],
                                 embed_dim=256)
    
    device_id = 0  # Set device ID
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")  # Set device to GPU if available
    model = model.to(device)  # Move model to device

    num_epochs = args.num_epochs  # Get number of epochs from arguments
    print_every = args.print_every  # Get print frequency from arguments
    lr = args.lr  # Get learning rate from arguments
    logging.info(f'learning rate = {lr}')  # Log learning rate
    print(f'learning rate = {lr}')  # Print learning rate

    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Define the loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999), weight_decay=0.02)  # Define the optimizer    
    
    logging.info('start training')  # Log start of training
    print('start training')  # Print start of training

    test_loss_list = []  # Initialize list to store test losses
    test_acc_list = []  # Initialize list to store test accuracies
    epoch_list = []  # Initialize list to store epochs
    for epoch in range(num_epochs):  # Loop over epochs
        model.train()  # Set model to training mode

        running_loss = 0.0  # Initialize running loss

        for i, (inputs_1d, labels) in enumerate(train_loader, 0):  # Loop over training data

            inputs_1d, labels = inputs_1d.to(device), labels.to(device)  # Move inputs and labels to device
            
            optimizer.zero_grad()  # Zero the gradients

            outputs = model(inputs_1d)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()  # Accumulate running loss
        
        if (epoch + 1) == 100:  # Change learning rate after 100 epochs
            optimizer = optim.Adam(model.parameters(), lr=3e-5, betas=(0.9,0.999), weight_decay=0.02)
            print("Changed learning rate to 3e-5")
        
        if (epoch + 1) % print_every == 0:  # Print and log information every specified number of epochs
            epoch_list.append(epoch+1)
            test_loss_item, test_acc_item = evaluate_test_loss(model, val_loader, criterion, epoch+1, outpath)
            logging.info(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}, Val Acc: {test_acc_item}")
            print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}, Val Acc: {test_acc_item}")
            test_loss_list.append(test_loss_item)
            test_acc_list.append(test_acc_item)


    logging.info("Training completed!")  # Log completion of training
    print("Training completed!")  # Print completion of training
    
    torch.save(model.state_dict(), outpath+'final_1d.pth')  # Save final model state
    np.save(outpath+'val_loss.npy', np.array(test_loss_list))  # Save validation loss array
    np.save(outpath+'val_acc.npy', np.array(test_acc_list))  # Save validation accuracy array
    logging.info(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')  # Log epoch with minimum validation loss
    print(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')  # Print epoch with minimum validation loss
