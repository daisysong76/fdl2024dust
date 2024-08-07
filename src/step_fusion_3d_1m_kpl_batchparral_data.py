import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import concurrent.futures

# Function to load SDO images (placeholder)
def load_sdo_images(input_window_start, input_window_end, sdo_data_dir):
    sdo_images = []
    for root, _, files in os.walk(sdo_data_dir):
        for file in files:
            parts = file.split('_')
            if len(parts) < 2:
                continue
            timestamp_str = parts[1]
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M')
            except ValueError:
                continue
            if input_window_start <= timestamp <= input_window_end:
                image = np.random.randn(512, 512)  # Replace with actual image loading
                sdo_images.append(image)
    return np.stack(sdo_images) if sdo_images else np.array([])

# Function to load radiation dose data (placeholder)
def load_radiation_data(target_window_start, target_window_end, rad_dose_filepath):
    rad_data = pd.read_csv(rad_dose_filepath)
    rad_data['timestamp_utc'] = pd.to_datetime(rad_data['timestamp_utc'])
    mask = (rad_data['timestamp_utc'] >= target_window_start) & (rad_data['timestamp_utc'] <= target_window_end)
    return rad_data.loc[mask, 'absorbed_dose_rate'].values

# Parallel loading of SDO and radiation data
def parallel_load_data(batch, sdo_data_dir, rad_dose_filepath):
    sdo_data_batch = []
    label_data_batch = []
    
    for _, row in batch.iterrows():
        input_window_start = pd.to_datetime(row['input_window_start'])
        input_window_end = pd.to_datetime(row['input_window_end'])
        target_window_start = pd.to_datetime(row['target_window_start'])
        target_window_end = pd.to_datetime(row['target_window_end'])

        sdo_data = load_sdo_images(input_window_start, input_window_end, sdo_data_dir)
        label_data = load_radiation_data(target_window_start, target_window_end, rad_dose_filepath)
        
        sdo_data_batch.append(sdo_data)
        label_data_batch.append(label_data)
    
    return sdo_data_batch, label_data_batch

# Load metadata CSV
metadata_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/google_cloud_buckets/saved_experiments/configs_and_metadata/exp-4/metadata.csv"
metadata_df = pd.read_csv(metadata_path)

# Directories
sdo_data_dir = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/google_cloud_buckets/sdoml"
rad_dose_filepath = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/google_cloud_buckets/radlab-private/data_tables/readings_table/per_instrument_padded/CRaTER-D1D2_readings_padded.csv"

# Define batch size
batch_size = 100

all_sdo_data = []
all_labels = []

# Split the metadata into batches
metadata_batches = [metadata_df.iloc[i:i + batch_size] for i in range(0, len(metadata_df), batch_size)]

# Parallelize the data loading process
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [
        executor.submit(parallel_load_data, batch, sdo_data_dir, rad_dose_filepath)
        for batch in metadata_batches
    ]

    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Loading SDO and radiation data"):
        sdo_data_batch, label_data_batch = future.result()
        all_sdo_data.extend(sdo_data_batch)
        all_labels.extend(label_data_batch)

all_sdo_data = np.array(all_sdo_data, dtype=object)  
all_labels = np.array(all_labels, dtype=object)  

all_labels_flat = np.array([np.mean(label) for label in all_labels])

train_sdo, temp_sdo, train_label, temp_label = train_test_split(all_sdo_data, all_labels_flat, test_size=0.4, random_state=42, stratify=all_labels_flat)
val_sdo, test_sdo, val_label, test_label = train_test_split(temp_sdo, temp_label, test_size=0.5, random_state=42, stratify=temp_label)

train_sdo = np.array(train_sdo)
val_sdo = np.array(val_sdo)
test_sdo = np.array(test_sdo)
train_label = np.array(train_label)
val_label = np.array(val_label)
test_label = np.array(test_label)

def save_data_to_pickle(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)

train_sdo_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/train_sdo.pkl"
val_sdo_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/val_sdo.pkl"
test_sdo_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/test_sdo.pkl"
train_label_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/train_label.pkl"
val_label_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/val_label.pkl"
test_label_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/test_label.pkl"

save_data_to_pickle(train_sdo, train_sdo_path)
save_data_to_pickle(val_sdo, val_sdo_path)
save_data_to_pickle(test_sdo, test_sdo_path)
save_data_to_pickle(train_label, train_label_path)
save_data_to_pickle(val_label, val_label_path)
save_data_to_pickle(test_label, test_label_path)

print(f"Train SDO data saved to {train_sdo_path}")
print(f"Validation SDO data saved to {val_sdo_path}")
print(f"Test SDO data saved to {test_sdo_path}")
print(f"Train labels saved to {train_label_path}")
print(f"Validation labels saved to {val_label_path}")
print(f"Test labels saved to {test_label_path}")
