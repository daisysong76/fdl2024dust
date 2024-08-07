import pickle
import numpy as np

# Load the data from the .pkl file
file_path = '/home/daisysong/2024-HL-Virtual-Dosimeter/video_features/video_featurestrain_3d_features.pkl'
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


# (myenv) (base) daisysong@deeplearning-xiaomei-gpu-vm:~$ python3 /home/daisysong/2024-HL-Virtual-Dosimeter/video_features/video_featurestrain_3d.py
# Data type: <class 'numpy.ndarray'>
# Shape of the array: (96, 768)
# Data type of the array elements: float32
# Number of data points in the array: 73728  # 96+12+12
# First few elements of the array:
# [[-0.23574458  0.17114656  1.9012388  ... -0.00685013 -0.5757994
#   -0.56275505]
#  [-0.23576665  0.17103672  1.901245   ... -0.00677709 -0.575834
#   -0.56277674]
#  [-0.23575087  0.1710956   1.9011993  ... -0.00684119 -0.5758047
#   -0.5627401 ]
#  [-0.23573375  0.17115788  1.901191   ... -0.00680656 -0.5758235
#   -0.56274134]
#  [-0.23576014  0.17107204  1.9012264  ... -0.00680894 -0.5758349
#   -0.5627694 ]]
# Array statistics:
# Mean: 1.847374005592428e-05
# Standard Deviation: 0.9975384473800659
# (myenv) (base) daisysong@deeplearning-xiaomei-gpu-vm:~$ 
