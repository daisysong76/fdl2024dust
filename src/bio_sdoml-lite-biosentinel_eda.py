import numpy as np

file_path = '/home/daisysong/2024-HL-Virtual-Dosimeter/data/google_cloud_buckets/sdoml-lite-biosentinel/2022/11/01/0000.aia_0094.npy'
data = np.load(file_path)

print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(data)
