import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load test data and labels
path_kine_test = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/test_input_data_1m_1d.pkl'
path_label_test = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/test_labels_data_1m_1d.pkl'

X_test = pd.read_pickle(path_kine_test)
label_test = pd.read_pickle(path_label_test)

# Function to create sequences
def create_sequences(data, labels, input_window):
    X, y = [], []
    for i in range(len(data) - input_window):
        X.append(data[i:i+input_window])
        y.append(labels[i+input_window])
    return np.array(X), np.array(y)

# Parameters
num_frames = 4  # Set to 4 as per the input window size
num_channels = 1
batch_size = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# Clean and normalize data
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)
label_test = scaler.fit_transform(label_test)

# Create sequences from test data
X_test, y_test = create_sequences(X_test, label_test, num_frames)

# Print shapes for debugging
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Check tensor sizes before creating dataset
print(f"Shape of X_test tensor: {X_test.shape}")
print(f"Shape of y_test tensor: {y_test.shape}")

# Create DataLoader for test data
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model architecture
class SwinTransformer_1D(nn.Module):
    def __init__(self, seq_len, in_chans, num_classes, window_size, drop_path_rate, patch_size, num_heads, depths, embed_dim):
        super(SwinTransformer_1D, self).__init__()
        # Define your model layers here
        self.seq_len = seq_len
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.window_size = window_size
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depths = depths
        self.embed_dim = embed_dim
        # Example layer: replace this with your actual model definition
        self.fc = nn.Linear(seq_len * in_chans, num_classes)

    def forward(self, x):
        # Example forward pass: replace this with your actual forward logic
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

# Function to load model and make predictions
def load_model_and_predict(model_path, test_loader):
    model = SwinTransformer_1D(seq_len=num_frames,  
                               in_chans=num_channels, 
                               num_classes=1,  # Predicting a single value
                               window_size=4, 
                               drop_path_rate=0.1,  
                               patch_size=1,
                               num_heads=[16, 16, 16, 16], 
                               depths=[2, 2, 6, 2],
                               embed_dim=256)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs_1d, labels in test_loader:
            inputs_1d, labels = inputs_1d.to(device), labels.to(device)
            if len(inputs_1d.shape) == 2:
                inputs_1d = inputs_1d.unsqueeze(1)
            elif len(inputs_1d.shape) != 3:
                raise ValueError(f"Expected input with 3 dimensions, but got {inputs_1d.shape}")
            
            outputs = model(inputs_1d)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(actuals)

# Function to plot predictions vs actual values
def plot_predictions(predictions, actuals, title='Predictions vs Actual Values'):
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label='Actual', marker='o', linestyle='-')
    plt.plot(predictions, label='Predicted', marker='x', linestyle='--')
    plt.title(title)
    plt.xlabel('Data Point Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Paths and parameters
test_model_path = '/home/daisysong/2024-HL-Virtual-Dosimeter/saved_models/step_1d/final_1d.pth'  # Adjust the path accordingly

# Load model and get predictions
predictions, actuals = load_model_and_predict(test_model_path, test_loader)

# Plot the results
plot_predictions(predictions, actuals)




