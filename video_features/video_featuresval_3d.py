# import pickle
# import numpy as np

# file_path = '/home/daisysong/2024-HL-Virtual-Dosimeter/video_features/video_featuresval_3d_features.pkl'

# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # Verify the type of data
# print(f"Data type: {type(data)}")

# # If the data is a numpy array, inspect its shape and a few elements
# if isinstance(data, np.ndarray):
#     print(f"Shape of the array: {data.shape}")
#     print(f"Data type of the array elements: {data.dtype}")

#     # Print the number of data points in the array
#     print(f"Number of data points in the array: {data.size}")
    
#     # Print the first few elements of the array
#     print("First few elements of the array:")
#     print(data[:5])

#     # Optionally, print more detailed statistics or visualizations
#     print(f"Array statistics:\nMean: {np.mean(data)}\nStandard Deviation: {np.std(data)}")

# import matplotlib.pyplot as plt

# # Assuming extracted_features is a 4D tensor of shape (batch_size, channels, height, width)
# # Select a sample to visualize
# sample_features = extracted_features[0]

# # If the features are 3D (e.g., (channels, height, width)), you can plot them
# if len(sample_features.shape) == 3:
#     channels = sample_features.shape[0]
#     fig, axs = plt.subplots(1, channels, figsize=(15, 5))
#     for i in range(channels):
#         axs[i].imshow(sample_features[i], cmap='gray')
#         axs[i].set_title(f'Channel {i}')
#     plt.show()

# # If the features are 4D (e.g., (batch_size, channels, height, width))
# # You can visualize a specific batch and channel
# if len(sample_features.shape) == 4:
#     batch = 0  # Select the first batch
#     channels = sample_features.shape[1]
#     fig, axs = plt.subplots(1, channels, figsize=(15, 5))
#     for i in range(channels):
#         axs[i].imshow(sample_features[batch, i], cmap='gray')
#         axs[i].set_title(f'Channel {i}')
#     plt.show()


#     Data type: <class 'numpy.ndarray'>
# Shape of the array: (12, 768)
# Data type of the array elements: float32
# Number of data points in the array: 9216
# First few elements of the array:
# [[-0.235736    0.17120068  1.9012018  ... -0.00685726 -0.5758095
#   -0.56273353]
#  [-0.23576014  0.17104064  1.9012616  ... -0.00681437 -0.5757894
#   -0.56277305]
#  [-0.23576915  0.17109165  1.901218   ... -0.00683831 -0.5758073
#   -0.56272143]
#  [-0.23576961  0.17111821  1.9012079  ... -0.00681788 -0.57581025
#   -0.5627408 ]
#  [-0.23573218  0.17116988  1.9012278  ... -0.00684086 -0.57582897
#   -0.5627288 ]]
# Array statistics:
# Mean: 1.847361090767663e-05
#Standard Deviation: 0.9975384473800659

import pickle
import numpy as np
import matplotlib.pyplot as plt

file_path = '/home/daisysong/2024-HL-Virtual-Dosimeter/video_features/video_featuresval_3d_features.pkl'

# Load the data from the pickle file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Verify the type of data
print(f"Data type: {type(data)}")

# If the data is a numpy array, inspect its shape and a few elements
if isinstance(data, np.ndarray):
    print(f"Shape of the array: {data.shape}")
    print(f"Data type of the array elements: {data.dtype}")

    # Print the number of data points in the array
    print(f"Number of data points in the array: {data.size}")
    
    # Print the first few elements of the array
    print("First few elements of the array:")
    print(data[:5])

    # Optionally, print more detailed statistics or visualizations
    print(f"Array statistics:\nMean: {np.mean(data)}\nStandard Deviation: {np.std(data)}")
else:
    print("Data is not a numpy array")

# Visualization part
if isinstance(data, np.ndarray):
    # Select a sample to visualize
    sample_features = data

    # If the features are 3D (e.g., (channels, height, width)), you can plot them
    if len(sample_features.shape) == 3:
        channels = sample_features.shape[0]
        fig, axs = plt.subplots(1, channels, figsize=(15, 5))
        for i in range(channels):
            axs[i].imshow(sample_features[i], cmap='gray')
            axs[i].set_title(f'Channel {i}')
        plt.show()

    # If the features are 4D (e.g., (batch_size, channels, height, width))
    # You can visualize a specific batch and channel
    elif len(sample_features.shape) == 4:
        batch = 0  # Select the first batch
        channels = sample_features.shape[1]
        fig, axs = plt.subplots(1, channels, figsize=(15, 5))
        for i in range(channels):
            axs[i].imshow(sample_features[batch, i], cmap='gray')
            axs[i].set_title(f'Channel {i}')
        plt.show()
    else:
        print("The data has an unsupported number of dimensions for visualization")
