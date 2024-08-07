import pickle
import numpy as np
import os
import torch

# Define the path to the pickle file
file_path = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/kine_train.pkl'
file_path= '/home/daisysong/2024-HL-Virtual-Dosimeter/data/label_train.pkl'

# Load and inspect the data
def inspect_and_convert_to_tensor(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        print(f"File: {file_path}")
        print(f"Type: {type(data)}")
        try:
            print(f"Shape: {data.shape}")
        except AttributeError:
            print("Shape: Not applicable")
        print(f"Size: {os.path.getsize(file_path)} bytes")
        print("-" * 50)

        # Convert to tensor
        tensor_data = torch.tensor(data)
        print(f"Converted to tensor: {tensor_data.shape}")

        # Print each element of the tensor
        print("Tensor elements:")
        print(tensor_data)

        return tensor_data

# Run the inspection and conversion
tensor_data = inspect_and_convert_to_tensor(file_path)


data = np.random.randn(30,3,77,224,224)   # 30 videos, 3 channels, 77 frames, 224x224 pixels. #TODO: 1, 9, 12*10, 512, 512
with open('./data/video_train.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randn(10,3,77,224,224)  # 10 videos, 3 channels, 77 frames, 224x224 pixels
with open('./data/video_val.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randn(30,3,51)        # 30 videos, 3 channels, 51 kinematic features
with open('./data/kine_train.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randn(10,3,51)     # 10 videos, 3 channels, 51 kinematic features
with open('./data/kine_val.pkl', 'wb') as file:
    pickle.dump(data, file)

data = np.random.randint(0, 4, size=30)   #. astype(np.float32)  # 30 labels
with open('./data/label_train.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randint(0, 4, size=10)      # astype(np.float32)  # 10 labels
with open('./data/label_val.pkl', 'wb') as file:
    pickle.dump(data, file)


# File: ./data/video_train.pkl
# Type: <class 'numpy.ndarray'>
# Shape: (30, 3, 77, 224, 224)     # 96, 10, 9, 512, 512
# Size: <file_size> bytes
# --------------------------------------------------
# File: ./data/video_val.pkl
# Type: <class 'numpy.ndarray'>
# Shape: (10, 3, 77, 224, 224)      # 12, 10,9,512,12
# Size: <file_size> bytes
# --------------------------------------------------
# File: ./data/kine_train.pkl
# Type: <class 'numpy.ndarray'>
# Shape: (30, 3, 51)                #96, 10, 51
# Size: <file_size> bytes
# --------------------------------------------------
# File: ./data/kine_val.pkl
# Type: <class 'numpy.ndarray'>     #12,10,51
# Shape: (10, 3, 51)
# Size: <file_size> bytes
# --------------------------------------------------
# File: ./data/label_train.pkl
# Type: <class 'numpy.ndarray'>    #30, 10
# Shape: (30,)
# Size: <file_size> bytes
# --------------------------------------------------
# File: ./data/label_val.pkl
# Type: <class 'numpy.ndarray'>   #10, 10
# Shape: (10,)
# Size: <file_size> bytes
# --------------------------------------------------


# (myenv) (base) daisysong@deeplearning-xiaomei-gpu-vm:~$ python3 /home/daisysong/2024-HL-Virtual-Dosimeter/src/dust_gen_data.py
# File: /home/daisysong/2024-HL-Virtual-Dosimeter/data/kine_train.pkl
# Type: <class 'numpy.ndarray'>
# Shape: (30, 3, 51)
# Size: 36875 bytes
# --------------------------------------------------
# Converted to tensor: torch.Size([30, 3, 51])
# Tensor elements:
# tensor([[[-0.3028, -1.1277,  1.0764,  ...,  0.0824,  0.7165,  0.3713],
#          [ 1.3466,  0.7104, -0.8315,  ..., -0.2936,  0.4977, -0.7212],
#          [ 0.6806, -0.7080, -0.3811,  ..., -0.4596,  1.4136,  0.3660]],

#         [[ 0.9330, -0.1114, -0.1837,  ..., -0.6794,  0.9509, -0.6121],
#          [-0.5610,  0.2032,  0.1641,  ..., -0.0440,  0.8940,  0.5298],
#          [ 1.0610, -1.0471,  2.1270,  ...,  0.3468,  0.7728, -0.7673]],

#         [[ 0.5420,  0.9294, -0.1973,  ..., -0.5273, -0.3438,  0.2226],
#          [-0.2597,  0.5619,  2.3201,  ...,  0.5740,  1.0309, -0.9839],
#          [ 1.3708,  1.4090,  0.2219,  ..., -0.1589,  1.1745, -0.3439]],

#         ...,

#         [[-0.7116,  1.2032, -0.1327,  ..., -0.3480,  0.4439,  2.3149],
#          [ 0.8051,  1.2006,  0.0312,  ...,  0.6951,  0.9807, -0.5393],
#          [ 0.3425,  1.9075,  0.4224,  ...,  1.0481,  0.3360, -1.2506]],

#         [[-0.0839, -1.4022,  0.4010,  ..., -1.3191,  1.8072,  0.4016],
#          [-1.1828, -1.1642, -1.5545,  ...,  0.9449, -0.1235,  1.0101],
#          [ 0.7823, -0.8361,  0.0783,  ...,  1.1444,  0.3645, -1.1402]],

#         [[ 0.0431,  0.6963, -1.3160,  ...,  2.8985, -0.3685,  1.3443],
#          [-1.1784,  0.7746, -1.1372,  ..., -1.3792,  1.2671, -0.6345],
#          [ 0.7173,  0.2717, -1.0427,  ..., -0.9979,  0.6952,  0.2131]]],
#        dtype=torch.float64)
# (myenv) (base) daisysong@deeplearning-xiaomei-gpu-vm:~$ 


# (myenv) (base) daisysong@deeplearning-xiaomei-gpu-vm:~$ python3 /home/daisysong/2024-HL-Virtual-Dosimeter/src/dust_gen_data.py
# File: /home/daisysong/2024-HL-Virtual-Dosimeter/data/label_train.pkl
# Type: <class 'numpy.ndarray'>
# Shape: (30,)
# Size: 388 bytes
# --------------------------------------------------
# Converted to tensor: torch.Size([30])
# Tensor elements:
# tensor([2, 0, 3, 1, 1, 3, 0, 0, 2, 3, 3, 1, 1, 0, 1, 3, 0, 0, 3, 0, 1, 1, 0, 3,
#         2, 0, 2, 0, 1, 3])
# (myenv) (base) daisysong@deeplearning-xiaomei-gpu-vm:~$ 