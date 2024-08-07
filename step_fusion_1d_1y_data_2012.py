#/home/daisysong/2024-HL-Virtual-Dosimeter/data/google_cloud_buckets/saved_experiments/configs_and_metadata/exp-3/metadata.csv
# 2012-01-01 03:00:00+00:00,2013-01-01 03:00:00+00:00



import os
import pandas as pd
import numpy as np
import torch
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import concurrent.futures

# Load metadata CSV
metadata_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/google_cloud_buckets/saved_experiments/configs_and_metadata/exp-3/metadata.csv"
metadata_df = pd.read_csv(metadata_path)

# Function to load radiation dose data for the target window
def load_radiation_data_target(target_window_start, target_window_end, rad_dose_filepath):
    rad_data = pd.read_csv(rad_dose_filepath)
    rad_data['timestamp_utc'] = pd.to_datetime(rad_data['timestamp_utc'])
    mask = (rad_data['timestamp_utc'] >= target_window_start) & (rad_data['timestamp_utc'] <= target_window_end)
    return rad_data.loc[mask, 'absorbed_dose_rate'].values

# Function to load radiation dose data for the input window
def load_radiation_data_input(input_window_start, input_window_end, rad_dose_filepath):
    rad_data = pd.read_csv(rad_dose_filepath)
    rad_data['timestamp_utc'] = pd.to_datetime(rad_data['timestamp_utc'])
    mask = (rad_data['timestamp_utc'] >= input_window_start) & (rad_data['timestamp_utc'] <= input_window_end)
    return rad_data.loc[mask, 'absorbed_dose_rate'].values

# Directories
rad_dose_filepath = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/google_cloud_buckets/radlab-private/data_tables/readings_table/per_instrument_padded/CRaTER-D1D2_readings_padded.csv"

# Lists to store all input data and labels
all_input_data = []
all_labels = []

# Function to process each row of the metadata
def process_row(row):
    input_window_start = pd.to_datetime(row['input_window_start'])
    input_window_end = pd.to_datetime(row['input_window_end'])
    target_window_start = pd.to_datetime(row['target_window_start'])
    target_window_end = pd.to_datetime(row['target_window_end'])

    # Load radiation dose data for input and target windows
    input_data = load_radiation_data_input(input_window_start, input_window_end, rad_dose_filepath)
    label_data = load_radiation_data_target(target_window_start, target_window_end, rad_dose_filepath)
    
    return input_data, label_data

# Iterate over each sample in the metadata using ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_row, [row for _, row in metadata_df.iterrows()]), total=len(metadata_df), desc="Loading radiation data"))

# Unpack the results
all_input_data, all_labels = zip(*results)

# Convert lists to numpy arrays
all_input_data = np.array(all_input_data, dtype=object)
all_labels = np.array(all_labels, dtype=object)

# Split the data into train, validation, and test sets
train_input_data, temp_input_data, train_labels, temp_labels = train_test_split(all_input_data, all_labels, test_size=0.4, random_state=42)
val_input_data, test_input_data, val_labels, test_labels = train_test_split(temp_input_data, temp_labels, test_size=0.5, random_state=42)

# Save input data and labels to pickle files
def save_data_to_pickle(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)

save_data_to_pickle(train_input_data, "/home/daisysong/2024-HL-Virtual-Dosimeter/data/train_input_data_1y_1d.pkl")
save_data_to_pickle(val_input_data, "/home/daisysong/2024-HL-Virtual-Dosimeter/data/val_input_data_1y_1d.pkl")
save_data_to_pickle(test_input_data, "/home/daisysong/2024-HL-Virtual-Dosimeter/data/test_input_data_1y_1d.pkl")

save_data_to_pickle(train_labels, "/home/daisysong/2024-HL-Virtual-Dosimeter/data/train_labels_data_1y_1d.pkl")
save_data_to_pickle(val_labels, "/home/daisysong/2024-HL-Virtual-Dosimeter/data/val_labels_data_1y_1d.pkl")
save_data_to_pickle(test_labels, "/home/daisysong/2024-HL-Virtual-Dosimeter/data/test_labels_data_1y_1d.pkl")

print("Data has been saved successfully.")

