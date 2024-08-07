

#/home/daisysong/2024-HL-Virtual-Dosimeter/src/toy_aia_data.py
# 
# import zarr

# # Path to the Zarr directory
# zarr_path = '/home/daisysong/AIA.zarr'

# # Open the Zarr group
# zarr_group = zarr.open(zarr_path, mode='r')

# # List all arrays in the Zarr group
# def list_zarr_arrays(group, prefix=''):
#     for key, item in group.items():
#         if isinstance(item, zarr.Group):
#             list_zarr_arrays(item, prefix + key + '/')
#         elif isinstance(item, zarr.Array):
#             print(prefix + key, item.shape, item.dtype)

# # List all arrays in the Zarr file
# list_zarr_arrays(zarr_group)

import zarr
import os

# Function to recursively print structure and metadata
def explore_zarr(group, prefix=''):
    # Print metadata for the current group/array
    print(f"Group/Array: {prefix}")
    if group.attrs:
        print(f"Metadata for {prefix}:")
        for key, value in group.attrs.items():
            print(f"  {key}: {value}")
    else:
        print(f"No metadata for {prefix}")
    print()
    
    # Recursively explore child groups/arrays
    for key, item in group.items():
        if isinstance(item, zarr.Group):
            explore_zarr(item, prefix + key + '/')
        elif isinstance(item, zarr.Array):
            print(f"Array: {prefix + key}")
            if item.attrs:
                print(f"Metadata for {prefix + key}:")
                for attr_key, attr_value in item.attrs.items():
                    print(f"  {attr_key}: {attr_value}")
            else:
                print(f"No metadata for {prefix + key}")
            print(f"Shape: {item.shape}")
            print(f"Dtype: {item.dtype}")
            print()

# Path to the Zarr directory
zarr_path = '/home/daisysong/AIA.zarr'

# Open the Zarr group
zarr_group = zarr.open(zarr_path, mode='r')

# Explore the Zarr file structure and metadata
explore_zarr(zarr_group)


# import zarr
# import os

# # Function to recursively print metadata
# def print_metadata(group, prefix=''):
#     # Print metadata for the current group/array
#     print(f"Metadata for {prefix}:")
#     for key, value in group.attrs.items():
#         print(f"  {key}: {value}")
#     print()
    
#     # Recursively print metadata for child groups/arrays
#     for key, item in group.items():
#         if isinstance(item, zarr.Group):
#             print_metadata(item, prefix + key + '/')
#         elif isinstance(item, zarr.Array):
#             print(f"Metadata for {prefix + key}:")
#             for attr_key, attr_value in item.attrs.items():
#                 print(f"  {attr_key}: {attr_value}")
#             print()

# # Path to the Zarr directory
# zarr_path = '/home/daisysong/AIA.zarr'

# # Open the Zarr group
# zarr_group = zarr.open(zarr_path, mode='r')

# # Print metadata for the root group
# print_metadata(zarr_group)

