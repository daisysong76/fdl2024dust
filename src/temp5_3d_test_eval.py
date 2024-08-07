# import torch
# import torch.nn as nn
# from models.video_swin_transformer import SwinTransformer3D
# from torch.utils.data import DataLoader, TensorDataset
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, r2_score

# # Commenting out wandb initialization for testing
# import wandb
# wandb.init(project='dust', entity='trilliumtechnologies')

# def load_state_dict_with_mismatch(model, state_dict):
#     model_state_dict = model.state_dict()
#     filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
#     for k, v in filtered_state_dict.items():
#         model_state_dict[k] = v
#     model.load_state_dict(model_state_dict, strict=False)

# def evaluate(model, loader, device):
#     total_loss = 0
#     total_samples = 0
#     criterion = nn.MSELoss()
#     all_predictions = []
#     all_targets = []

#     model.eval()
#     with torch.no_grad():
#         for i, (data, target) in enumerate(loader):
#             data, target = data.to(device).float(), target.to(device)
#             output = model(data)
#             loss = criterion(output, target)
#             total_loss += loss.item() * data.size(0)
#             total_samples += data.size(0)
#             all_predictions.append(output.cpu().numpy())
#             all_targets.append(target.cpu().numpy())

#             # Log the loss for this batch
#             wandb.log({"batch_loss": loss.item(), "batch": i})

#     avg_loss = total_loss / total_samples
#     all_predictions = np.concatenate(all_predictions, axis=0)
#     all_targets = np.concatenate(all_targets, axis=0)
#     return avg_loss, all_predictions, all_targets

# if __name__ == "__main__":
#     # Paths and settings
#     batch_size = 1
#     path_3d_model = '/home/daisysong/saved_models/step_3d/final_3d.pth'
#     path_video_test = '/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/test_image.pkl'
#     path_label_test = '/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/test_label.pkl'

#     # Load the test data
#     video_test = torch.from_numpy(pd.read_pickle(path_video_test))
#     label_test = torch.from_numpy(pd.read_pickle(path_label_test))

#     # Print additional information about the data
#     print(f"Loaded video_test with shape: {video_test.shape}")
#     print(f"Loaded label_test with shape: {label_test.shape}")

#     # Reshape the data
#     video_test = video_test.view(-1, 3, 9, 512, 512)
#     label_test = label_test.view(-1, 12)  # Adjust this if the number of regression targets is different

#     print(f"Reshaped video_test: {video_test.shape}")
#     print(f"Reshaped label_test: {label_test.shape}")

#     assert not torch.isnan(video_test).any(), "Test data contains NaN values"
#     assert not torch.isinf(video_test).any(), "Test data contains Inf values"
#     assert not torch.isnan(label_test).any(), "Test labels contain NaN values"
#     assert not torch.isinf(label_test).any(), "Test labels contain Inf values"

#     # Check if sizes match
#     assert video_test.size(0) == label_test.size(0), "Size mismatch between video_test and label_test"
#     print(f"Number of samples in the test set: {video_test.size(0)}")

#     # Normalize the labels to [0, 1] using train normalization values (use min_label and max_label from training)
#     min_label = label_test.min(dim=0)[0]
#     max_label = label_test.max(dim=0)[0]
#     label_test_norm = (label_test - min_label) / (max_label - min_label)

#     test_dataset = TensorDataset(video_test, label_test_norm)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # Define the model
#     num_regression_targets = label_test.shape[1]
#     model = SwinTransformer3D(num_classes=1,
#                               embed_dim=96,
#                               drop_path_rate=0.1,
#                               mlp_ratio=4.0,
#                               patch_norm=True,
#                               patch_size=(2, 4, 4),
#                               pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_512.pth',
#                               pretrained2d=True,
#                               window_size=(8, 7, 7))

#     # Load the saved model state
#     state_dict = torch.load(path_3d_model, map_location=device)
#     if list(state_dict.keys())[0].startswith('module.'):
#         state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

#     load_state_dict_with_mismatch(model, state_dict)
#     model = model.to(device)

#     # Evaluate the model
#     test_loss, test_predictions, test_ground_truths = evaluate(model, test_loader, device)

#     # Print the evaluation results
#     print(f"Test Loss: {test_loss}")
#     print(f"Mean Absolute Error: {mean_absolute_error(test_ground_truths, test_predictions)}")
#     print(f"R² Score: {r2_score(test_ground_truths, test_predictions)}")

#     # Plotting
#     sns.set(style="whitegrid")
    
#     # Predictions vs Ground Truth
#     plt.figure(figsize=(10, 6))
#     for i in range(num_regression_targets):
#         plt.scatter(test_ground_truths[:, i], test_predictions[:, i], label=f'Target {i+1}')
#     plt.plot([test_ground_truths.min(), test_ground_truths.max()], [test_ground_truths.min(), test_ground_truths.max()], 'k--', lw=2)
#     plt.xlabel('Ground Truth')
#     plt.ylabel('Predictions')
#     plt.title('Predictions vs Ground Truth')
#     plt.legend()
#     plt.savefig('./saved_models/predictions_vs_ground_truth.png')
#     plt.show()


import torch
import torch.nn as nn
from models.video_swin_transformer import SwinTransformer3D
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import logging
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import wandb

# Initialize wandb
wandb.init(project='dust', entity='trilliumtechnologies')

def load_state_dict_with_mismatch(model, state_dict):
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
    for k, v in filtered_state_dict.items():
        model_state_dict[k] = v
    model.load_state_dict(model_state_dict, strict=False)

def evaluate(model, loader, device):
    total_loss = 0
    total_samples = 0
    criterion = nn.MSELoss()
    all_predictions = []
    all_targets = []
    step_losses = []

    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device).float(), target.to(device)
            output = model(data)
            output = output.view(target.size())  # Ensure the output has the same shape as the target
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            step_losses.append(loss.item())
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
            # Log the loss for this batch
            wandb.log({"batch_loss": loss.item(), "batch": i})

    avg_loss = total_loss / total_samples
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return avg_loss, all_predictions, all_targets, step_losses

if __name__ == "__main__":
    # Hard-coded settings
    batch_size = 1
    seed = 93728645

    # Model settings
    path_3d_model = '/home/daisysong/saved_models/step_3d/final_3d.pth'
    num_frames = 7200
    height = 512
    width = 512

    # Input paths
    path_video_test = '/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/test_image.pkl'
    path_label_test = '/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/test_label.pkl'

    # Load the data
    video_test = torch.from_numpy(pd.read_pickle(path_video_test)).float()
    label_test = torch.from_numpy(pd.read_pickle(path_label_test))

    # Print additional information about the data
    print(f"Loaded video_test with shape: {video_test.shape}")
    print(f"Loaded label_test with shape: {label_test.shape}")

    # Reshape the data
    # Assuming each sample in video_test corresponds to a certain number of frames
    num_samples = video_test.size(0) * video_test.size(1)  # Total number of time steps
    video_test = video_test.view(num_samples, 3, 9, 512, 512)
    label_test = label_test.view(num_samples, 12)  # Adjust this if the number of regression targets is different

    print(f"Reshaped video_test: {video_test.shape}")
    print(f"Reshaped label_test: {label_test.shape}")

    assert not torch.isnan(video_test).any(), "Test data contains NaN values"
    assert not torch.isinf(video_test).any(), "Test data contains Inf values"
    assert not torch.isnan(label_test).any(), "Test labels contain NaN values"
    assert not torch.isinf(label_test).any(), "Test labels contain Inf values"

    # Check if sizes match
    assert video_test.size(0) == label_test.size(0), "Size mismatch between video_test and label_test"
    print(f"Number of samples in the test set: {video_test.size(0)}")

    # Normalize the labels to [0, 1] using train normalization values (use min_label and max_label from training)
    min_label = label_test.min(dim=0)[0]
    max_label = label_test.max(dim=0)[0]
    label_test_norm = (label_test - min_label) / (max_label - min_label)

    test_dataset = TensorDataset(video_test, label_test_norm)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Output paths
    outpath = './saved_models/'
    logpath = './logs/'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure logging
    logging.basicConfig(filename=logpath+'test_3d_features.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    logging.info(f'----- data path {path_video_test}')
    print(f'----- data path {path_video_test}')

    logging.info('---- Define 3D Swin ----')
    print('---- Define 3D Swin ----')

    # Get the number of regression targets
    num_regression_targets = label_test.shape[1]

    model = SwinTransformer3D(num_classes=num_regression_targets,  # Set this to match the number of regression targets
                              embed_dim=96,
                              drop_path_rate=0.1,
                              mlp_ratio=4.0,
                              patch_norm=True,
                              patch_size=(2, 4, 4),
                              pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_512.pth',
                              pretrained2d=True,
                              window_size=(8, 7, 7))

    state_dict = torch.load(path_3d_model, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    load_state_dict_with_mismatch(model, state_dict)
    model = model.to(device)

    logging.info('test loader')
    print('test loader')

    # Evaluate on the test set
    test_loss, test_predictions, test_ground_truths, step_losses = evaluate(model, test_loader, device)

    # Save the model
    torch.save(model.state_dict(), outpath + 'test_model.pth')

    # Log the test results
    wandb.log({
        "test_loss": test_loss
    })

    # Finish the wandb run
    wandb.finish()

    print(f"Test Loss: {test_loss}")
    print(f"Mean Absolute Error: {mean_absolute_error(test_ground_truths, test_predictions)}")
    print(f"R² Score: {r2_score(test_ground_truths, test_predictions)}")

    # Plotting
    sns.set(style="whitegrid")
    
    # Predictions vs Ground Truth over time
    time_steps = np.arange(len(test_predictions))

    plt.figure(figsize=(15, 10))
    for i in range(num_regression_targets):
        plt.plot(time_steps, test_ground_truths[:, i], label=f'Ground Truth {i+1}')
        plt.plot(time_steps, test_predictions[:, i], label=f'Prediction {i+1}', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Values')
    plt.title('Predictions vs Ground Truth Over Time')
    plt.legend()
    plt.savefig(outpath + 'predictions_vs_ground_truth_over_time.png')
    plt.show()

    # Plot the test loss convergence
    plt.figure(figsize=(10, 6))
    plt.plot(step_losses, label='Test Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Test Loss Convergence')
    plt.legend()
    plt.savefig(outpath + 'test_loss_convergence.png')
