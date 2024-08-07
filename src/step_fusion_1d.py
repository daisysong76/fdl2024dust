# # ----- Define Data Loader -----
# X_train shape: (148, 3)
# label_train shape: (148, 4)
# X_val shape: (50, 3)
# label_val shape: (50, 4)
# X_test shape: (50, 3)
# label_test shape: (50, 4)
# Converted X_train shape: torch.Size([148, 3])
# Converted label_train shape: torch.Size([148, 4])
# Converted X_val shape: torch.Size([50, 3])
# Converted label_val shape: torch.Size([50, 4])
# Converted X_test shape: torch.Size([50, 3])
# Converted label_test shape: torch.Size([50, 4])
# Converted X_train shape after padding: torch.Size([148, 4])
# Converted label_train shape: torch.Size([148, 4])
# Converted X_val shape after padding: torch.Size([50, 4])
# Converted label_val shape: torch.Size([50, 4])
# Converted X_test shape after padding: torch.Size([50, 4])
# Converted label_test shape: torch.Size([50, 4])
# ----- Define 1D swin -----
# learning rate = 0.0003
# start training
# /home/daisysong/fdl2024project/myenv/lib/python3.10/site-packages/torch/nn/parallel/parallel_apply.py:79: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
#   with torch.cuda.device(device), torch.cuda.stream(stream), autocast(enabled=autocast_enabled):
# /home/daisysong/fdl2024project/myenv/lib/python3.10/site-packages/torch/nn/modules/loss.py:538: UserWarning: Using a target size (torch.Size([2, 4])) that is different to the input size (torch.Size([2, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.

# Your model is outputting a tensor of size [batch_size, 1], but your labels are of size [batch_size, 4]. This discrepancy needs to be resolved.
# Given that this is a regression task and the num_classes is set to 1 in the SwinTransformer model, it seems the model is expected to output a single value per sample. If the target is supposed to be a single value, you need to ensure the labels are also a single value per sample.
# Adjust the Label Shape: If your task is to predict a single value for each sample, make sure your labels reflect that. Modify the label tensors to match the output shape of the model.
# Model Adjustment: If your task indeed requires predicting multiple values (e.g., num_classes=4), then you need to adjust the model output dimensions accordingly.

# import torch
# import torch.nn as nn
# from models.SwinTransformer_cls import SwinTransformer_1D
# from torch.utils.data import Dataset, DataLoader, TensorDataset

# import numpy as np
# import pandas as pd

# import torch.optim as optim
# import torch.nn.functional as F
# import logging
# from sklearn.preprocessing import StandardScaler
# import wandb
# import gc
# import os
# #import torch.lightning as pl

# from torch.utils.tensorboard import SummaryWriter
# lightning_log_dir = '/home/daisysong/2024-HL-Virtual-Dosimeter/logs/lightning_logs'

# def evaluate_test_loss(model, test_loader, criterion, epoch, outpath):
#     model.eval()
#     running_test_loss = 0.0
#     with torch.no_grad():
#         for inputs_1d, labels in test_loader:
#             inputs_1d, labels = inputs_1d.to(device), labels.to(device)
            
#             if len(inputs_1d.shape) == 2:
#                 inputs_1d = inputs_1d.unsqueeze(1)
#             elif len(inputs_1d.shape) != 3:
#                 raise ValueError(f"Expected input with 3 dimensions, but got {inputs_1d.shape}")

#             outputs = model(inputs_1d)
#             test_loss = criterion(outputs, labels)
#             running_test_loss += test_loss.item()

#     average_test_loss = running_test_loss / len(test_loader)
#     torch.save(model.state_dict(), outpath + 'e' + str(epoch) + '_' + str(np.round(average_test_loss, 4)) + '.pth')
#     return average_test_loss

# if __name__ == "__main__":
#     # Clear GPU cache
#     torch.cuda.empty_cache()

#     # Initialize wandb
#     wandb.init(project='dust', entity='trilliumtechnologies')

#     batch_size = 2
#     num_epochs = 100
#     lr = 0.0003
#     seed = 93728645
#     print_every = 1
#     #num_frames = 1#TODO 4???
#     num_frames = 4 # Set to 4 as per the input window size
#     #target_window = 4 # Set to 4 as per the target window size
#     num_channels = 1

#     path_kine_train = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/train_input_data_1m_1d.pkl'
#     path_kine_val = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/val_input_data_1m_1d.pkl'
#     path_kine_test = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/test_input_data_1m_1d.pkl'
#     path_label_train = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/train_labels_data_1m_1d.pkl'
#     path_label_val = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/val_labels_data_1m_1d.pkl'
#     path_label_test = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/test_labels_data_1m_1d.pkl'

#     outpath = '/home/daisysong/2024-HL-Virtual-Dosimeter/saved_models/step_1d/'
#     logpath = '/home/daisysong/2024-HL-Virtual-Dosimeter/logs/'

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

#     logging.basicConfig(filename=logpath+'step_1d.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#     torch.manual_seed(seed)  
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)  
#         torch.cuda.manual_seed_all(seed)  
#     np.random.seed(seed)  
    
#     logging.info('----- Define Data Loader -----')  
#     print('----- Define Data Loader -----')  
    
#     X_train = pd.read_pickle(path_kine_train)
#     X_val = pd.read_pickle(path_kine_val)
#     X_test = pd.read_pickle(path_kine_test)
#     label_train = pd.read_pickle(path_label_train)
#     label_val = pd.read_pickle(path_label_val)
#     label_test = pd.read_pickle(path_label_test)
    
#     print(f"X_train shape: {X_train.shape}")
#     print(f"label_train shape: {label_train.shape}")
#     print(f"X_val shape: {X_val.shape}")
#     print(f"label_val shape: {label_val.shape}")
#     print(f"X_test shape: {X_test.shape}")
#     print(f"label_test shape: {label_test.shape}")

#     logging.info(f"X_train shape: {X_train.shape}")
#     logging.info(f"label_train shape: {label_train.shape}")
#     logging.info(f"X_val shape: {X_val.shape}")
#     logging.info(f"label_val shape: {label_val.shape}")
#     logging.info(f"X_test shape: {X_test.shape}")
#     logging.info(f"label_test shape: {label_test.shape}")
    
#     def clean_and_normalize_data(train_data, val_data, test_data):
#         scaler = StandardScaler()
#         train_data = scaler.fit_transform(train_data)
#         val_data = scaler.transform(val_data)
#         test_data = scaler.transform(test_data)
#         return train_data, val_data, test_data

#     # def clean_labels(*labels):
#     #     cleaned_labels = []
#     #     for label in labels:
#     #         label = np.array(label, dtype=np.float32)
#     #         cleaned_labels.append(label)
#     #     return cleaned_labels

#     def clean_and_normalize_labels(train_labels, val_labels, test_labels):
#         scaler = StandardScaler()
#         train_labels = scaler.fit_transform(train_labels)
#         val_labels = scaler.transform(val_labels)
#         test_labels = scaler.transform(test_labels)
#         return train_labels, val_labels, test_labels
#     try:
#         X_train, X_val, X_test = clean_and_normalize_data(X_train, X_val, X_test)
#         label_train, label_val, label_test = clean_and_normalize_labels(label_train, label_val, label_test)
        
#         X_train = torch.tensor(X_train, dtype=torch.float32)
#         X_val = torch.tensor(X_val, dtype=torch.float32)
#         X_test = torch.tensor(X_test, dtype=torch.float32)
#         label_train = torch.tensor(label_train, dtype=torch.float32).view(-1, 4)
#         label_val = torch.tensor(label_val, dtype=torch.float32).view(-1, 4)
#         label_test = torch.tensor(label_test, dtype=torch.float32).view(-1, 4)
#     except ValueError as e:
#         print(f"Data conversion error: {e}")
#         logging.error(f"Data conversion error: {e}")
#         exit(1)

#     print(f"Converted X_train shape: {X_train.shape}")
#     print(f"Converted label_train shape: {label_train.shape}")
#     print(f"Converted X_val shape: {X_val.shape}")
#     print(f"Converted label_val shape: {label_val.shape}")
#     print(f"Converted X_test shape: {X_test.shape}")
#     print(f"Converted label_test shape: {label_test.shape}")

#     X_train = F.pad(X_train, (0, 1), "constant", 0)
#     X_val = F.pad(X_val, (0, 1), "constant", 0)
#     X_test = F.pad(X_test, (0, 1), "constant", 0)
    
#     print(f"Converted X_train shape after padding: {X_train.shape}")
#     print(f"Converted label_train shape: {label_train.shape}")
#     print(f"Converted X_val shape after padding: {X_val.shape}")
#     print(f"Converted label_val shape: {label_val.shape}")
#     print(f"Converted X_test shape after padding: {X_test.shape}")
#     print(f"Converted label_test shape: {label_test.shape}")

#     train_dataset = TensorDataset(X_train, label_train)  
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  

#     val_dataset = TensorDataset(X_val, label_val)  
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  

#     test_dataset = TensorDataset(X_test, label_test)  
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  
    
#     logging.info('----- Define 1D swin -----')  
#     print('----- Define 1D swin -----')  

#     model = SwinTransformer_1D(seq_len=4,  
#                                  in_chans=1, 
#                                  num_classes=4,  # For regression, we have 1 output
#                                  # TODO: Change to 4, # Predicting 4 values, sequence prediction, 4 values for each sequence, input sequence length is 4, output sequence length is 4
#                                  window_size=4, 
#                                  drop_path_rate=0.1,  
#                                  patch_size=1,
#                                  num_heads=[16, 16, 16, 16], 
#                                  depths=[2, 2, 6, 2],
#                                  embed_dim=256)
    
#     model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
#     model = model.to(device)  

#     logging.info(f'learning rate = {lr}')  
#     print(f'learning rate = {lr}')  

#     # Log hyperparameters to wandb
#     wandb.config = {
#         "batch_size": batch_size,
#         "num_epochs": num_epochs,
#         "learning_rate": lr,
#         "num_frames": num_frames,
#         "seed": seed
#     }

#     criterion = nn.MSELoss()  
#     optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999), weight_decay=0.02)     
    
#     logging.info('start training')  
#     print('start training') 

#     test_loss_list = []  
#     epoch_list = [] 
#     for epoch in range(num_epochs):  
#         model.train() 
#         running_loss = 0.0  

#         for i, (inputs_1d, labels) in enumerate(train_loader, 0):  
#             inputs_1d, labels = inputs_1d.to(device), labels.to(device) 

#             if torch.isnan(inputs_1d).any():
#                 print(f"NaN detected in inputs")
#             if torch.isnan(labels).any():
#                 print(f"NaN detected in labels") 

#             if len(inputs_1d.shape) == 2:
#                 inputs_1d = inputs_1d.unsqueeze(1)
#             elif len(inputs_1d.shape) != 3:
#                 raise ValueError(f"Expected input with 3 dimensions, but got {inputs_1d.shape}")  

#             optimizer.zero_grad()  
#             outputs = model(inputs_1d)  
#             loss = criterion(outputs, labels)  
#             loss.backward()  
#             optimizer.step()  
#             running_loss += loss.item() 
        
#         if (epoch + 1) == 100:  
#             optimizer = optim.Adam(model.parameters(), lr=3e-5, betas=(0.9,0.999), weight_decay=0.02)
#             print("Changed learning rate to 3e-5")
        
#         if (epoch + 1) % print_every == 0:  
#             epoch_list.append(epoch+1)
#             test_loss_item = evaluate_test_loss(model, val_loader, criterion, epoch+1, outpath)
#             logging.info(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}")
#             print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}")
#             test_loss_list.append(test_loss_item)

#     logging.info("Training completed!")  
#     print("Training completed!")  
    
#     torch.save(model.state_dict(), outpath+'final_1d.pth')  
#     np.save(outpath+'val_loss.npy', np.array(test_loss_list))  
#     logging.info(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)') 
#     print(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)') 

import torch
import torch.nn as nn
from models.SwinTransformer_cls import SwinTransformer_1D
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import logging
from sklearn.preprocessing import StandardScaler
import wandb
import gc
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

lightning_log_dir = '/home/daisysong/2024-HL-Virtual-Dosimeter/logs/lightning_logs'

def evaluate_test_loss(model, test_loader, criterion, epoch, outpath):
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for inputs_1d, labels in test_loader:
            inputs_1d, labels = inputs_1d.to(device), labels.to(device)
            
            if len(inputs_1d.shape) == 2:
                inputs_1d = inputs_1d.unsqueeze(1)
            elif len(inputs_1d.shape) != 3:
                raise ValueError(f"Expected input with 3 dimensions, but got {inputs_1d.shape}")

            outputs = model(inputs_1d)
            test_loss = criterion(outputs, labels)
            running_test_loss += test_loss.item()

    average_test_loss = running_test_loss / len(test_loader)
    torch.save(model.state_dict(), outpath + 'e' + str(epoch) + '_' + str(np.round(average_test_loss, 4)) + '.pth')
    return average_test_loss

if __name__ == "__main__":
    # Clear GPU cache
    torch.cuda.empty_cache()

    # Initialize wandb
    wandb.init(project='dust', entity='trilliumtechnologies')

    batch_size = 2
    num_epochs = 100
    lr = 0.0003
    seed = 93728645
    print_every = 1
    num_frames = 4
    num_channels = 1

    path_kine_train = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/train_input_data_1m_1d.pkl'
    path_kine_val = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/val_input_data_1m_1d.pkl'
    path_kine_test = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/test_input_data_1m_1d.pkl'
    path_label_train = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/train_labels_data_1m_1d.pkl'
    path_label_val = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/val_labels_data_1m_1d.pkl'
    path_label_test = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/test_labels_data_1m_1d.pkl'

    outpath = '/home/daisysong/2024-HL-Virtual-Dosimeter/saved_models/step_1d/'
    logpath = '/home/daisysong/2024-HL-Virtual-Dosimeter/logs/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(filename=logpath+'step_1d.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    logging.info('----- Define Data Loader -----')
    print('----- Define Data Loader -----')

    X_train = pd.read_pickle(path_kine_train)
    X_val = pd.read_pickle(path_kine_val)
    X_test = pd.read_pickle(path_kine_test)
    label_train = pd.read_pickle(path_label_train)
    label_val = pd.read_pickle(path_label_val)
    label_test = pd.read_pickle(path_label_test)

    print(f"X_train shape: {X_train.shape}")
    print(f"label_train shape: {label_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"label_val shape: {label_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"label_test shape: {label_test.shape}")

    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"label_train shape: {label_train.shape}")
    logging.info(f"X_val shape: {X_val.shape}")
    logging.info(f"label_val shape: {label_val.shape}")
    logging.info(f"X_test shape: {X_test.shape}")
    logging.info(f"label_test shape: {label_test.shape}")

    def clean_and_normalize_data(train_data, val_data, test_data):
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        val_data = scaler.transform(val_data)
        test_data = scaler.transform(test_data)
        return train_data, val_data, test_data

    def clean_and_normalize_labels(train_labels, val_labels, test_labels):
        scaler = StandardScaler()
        train_labels = scaler.fit_transform(train_labels)
        val_labels = scaler.transform(val_labels)
        test_labels = scaler.transform(test_labels)
        return train_labels, val_labels, test_labels

    try:
        X_train, X_val, X_test = clean_and_normalize_data(X_train, X_val, X_test)
        label_train, label_val, label_test = clean_and_normalize_labels(label_train, label_val, label_test)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        label_train = torch.tensor(label_train, dtype=torch.float32).view(-1, 4)
        label_val = torch.tensor(label_val, dtype=torch.float32).view(-1, 4)
        label_test = torch.tensor(label_test, dtype=torch.float32).view(-1, 4)
    except ValueError as e:
        print(f"Data conversion error: {e}")
        logging.error(f"Data conversion error: {e}")
        exit(1)

    print(f"Converted X_train shape: {X_train.shape}")
    print(f"Converted label_train shape: {label_train.shape}")
    print(f"Converted X_val shape: {X_val.shape}")
    print(f"Converted label_val shape: {label_val.shape}")
    print(f"Converted X_test shape: {X_test.shape}")
    print(f"Converted label_test shape: {label_test.shape}")

    logging.info(f"Converted X_train shape: {X_train.shape}")
    logging.info(f"Converted label_train shape: {label_train.shape}")
    logging.info(f"Converted X_val shape: {X_val.shape}")
    logging.info(f"Converted label_val shape: {label_val.shape}")
    logging.info(f"Converted X_test shape: {X_test.shape}")
    logging.info(f"Converted label_test shape: {label_test.shape}")

    X_train = F.pad(X_train, (0, 1), "constant", 0)
    X_val = F.pad(X_val, (0, 1), "constant", 0)
    X_test = F.pad(X_test, (0, 1), "constant", 0)

    print(f"Converted X_train shape after padding: {X_train.shape}")
    print(f"Converted X_val shape after padding: {X_val.shape}")
    print(f"Converted X_test shape after padding: {X_test.shape}")

    logging.info(f"Converted X_train shape after padding: {X_train.shape}")
    logging.info(f"Converted X_val shape after padding: {X_val.shape}")
    logging.info(f"Converted X_test shape after padding: {X_test.shape}")

    train_dataset = TensorDataset(X_train, label_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val, label_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(X_test, label_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logging.info('----- Define 1D swin -----')
    print('----- Define 1D swin -----')

    model = SwinTransformer_1D(seq_len=4,
                               in_chans=1,
                               num_classes=4,  # For regression, we have 1 output
                               window_size=4,
                               drop_path_rate=0.1,
                               patch_size=1,
                               num_heads=[16, 16, 16, 16],
                               depths=[2, 2, 6, 2],
                               embed_dim=256)

    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)

    logging.info(f'learning rate = {lr}')
    print(f'learning rate = {lr}')

    # Log hyperparameters to wandb
    wandb.config = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "num_frames": num_frames,
        "seed": seed
    }

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999), weight_decay=0.02)

    logging.info('start training')
    print('start training')

    test_loss_list = []
    epoch_list = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs_1d, labels) in enumerate(train_loader, 0):
            inputs_1d, labels = inputs_1d.to(device), labels.to(device)

            if torch.isnan(inputs_1d).any():
                print(f"NaN detected in inputs")
                logging.warning(f"NaN detected in inputs at batch {i} of epoch {epoch}")
            if torch.isnan(labels).any():
                print(f"NaN detected in labels")
                logging.warning(f"NaN detected in labels at batch {i} of epoch {epoch}")

            if len(inputs_1d.shape) == 2:
                inputs_1d = inputs_1d.unsqueeze(1)
            elif len(inputs_1d.shape) != 3:
                raise ValueError(f"Expected input with 3 dimensions, but got {inputs_1d.shape}")

            optimizer.zero_grad()
            outputs = model(inputs_1d)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) == 100:
            optimizer = optim.Adam(model.parameters(), lr=3e-5, betas=(0.9,0.999), weight_decay=0.02)
            print("Changed learning rate to 3e-5")
            logging.info("Changed learning rate to 3e-5")

        if (epoch + 1) % print_every == 0:
            epoch_list.append(epoch+1)
            test_loss_item = evaluate_test_loss(model, val_loader, criterion, epoch+1, outpath)
            logging.info(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}")
            print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}")
            test_loss_list.append(test_loss_item)

    logging.info("Training completed!")
    print("Training completed!")

    torch.save(model.state_dict(), outpath+'final_1d.pth')
    np.save(outpath+'val_loss.npy', np.array(test_loss_list))
    logging.info(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
    print(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')

    # Plot the validation loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, test_loss_list, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss over Epochs')
    plt.grid(True)
    plt.savefig(outpath + 'validation_loss_plot.png')
    plt.show()


