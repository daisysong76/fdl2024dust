# 2015-05-01 03:00:00+00:00
# 2015-06-01 03:00:00+00:00
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import concurrent.futures
import dask.dataframe as dd
import dask.array as da
import dask.bag as db
from dask.diagnostics import ProgressBar
import logging
import asyncio
from joblib import Memory

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create a memory cache directory
cache_dir = "./cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

memory = Memory(cache_dir, verbose=0)

# Function to load SDO images (replace placeholder logic with actual image loading)
@memory.cache
def load_sdo_images_cached(input_window_start, input_window_end, sdo_data_dir):
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
                # Replace this with actual image loading logic
                image = np.random.randn(512, 512)
                sdo_images.append(image)
    return np.stack(sdo_images) if sdo_images else np.array([])

# Function to load radiation dose data
@memory.cache
def load_radiation_data_cached(target_window_start, target_window_end, rad_dose_filepath):
    rad_data = dd.read_csv(rad_dose_filepath)
    rad_data['timestamp_utc'] = dd.to_datetime(rad_data['timestamp_utc'])
    mask = (rad_data['timestamp_utc'] >= target_window_start) & (rad_data['timestamp_utc'] <= target_window_end)
    return rad_data.loc[mask, 'absorbed_dose_rate'].compute().values

# Asynchronous wrapper for parallel_load_data
async def parallel_load_data(row, sdo_data_dir, rad_dose_filepath):
    input_window_start = pd.to_datetime(row['input_window_start'])
    input_window_end = pd.to_datetime(row['input_window_end'])
    target_window_start = pd.to_datetime(row['target_window_start'])
    target_window_end = pd.to_datetime(row['target_window_end'])

    loop = asyncio.get_event_loop()
    sdo_data = await loop.run_in_executor(None, load_sdo_images_cached, input_window_start, input_window_end, sdo_data_dir)
    label_data = await loop.run_in_executor(None, load_radiation_data_cached, target_window_start, target_window_end, rad_dose_filepath)
    
    return sdo_data, label_data

# Load metadata CSV
metadata_path = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/google_cloud_buckets/saved_experiments/configs_and_metadata/exp-4/metadata.csv"
metadata_df = pd.read_csv(metadata_path)

# Directories
sdo_data_dir = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/google_cloud_buckets/sdoml"
rad_dose_filepath = "/home/daisysong/2024-HL-Virtual-Dosimeter/data/google_cloud_buckets/radlab-private/data_tables/readings_table/per_instrument_padded/CRaTER-D1D2_readings_padded.csv"

async def load_data():
    all_sdo_data = []
    all_labels = []

    tasks = [
        parallel_load_data(row, sdo_data_dir, rad_dose_filepath)
        for _, row in metadata_df.iterrows()
    ]

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Loading SDO and radiation data"):
        sdo_data, label_data = await future
        all_sdo_data.append(sdo_data)
        all_labels.append(label_data)

    return np.array(all_sdo_data, dtype=object), np.array(all_labels, dtype=object)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    all_sdo_data, all_labels = loop.run_until_complete(load_data())

    all_labels_flat = np.array([np.mean(label) for label in all_labels])

    train_sdo, temp_sdo, train_label, temp_label = train_test_split(
        all_sdo_data, all_labels_flat, test_size=0.4, random_state=42
    )
    val_sdo, test_sdo, val_label, test_label = train_test_split(
        temp_sdo, temp_label, test_size=0.5, random_state=42
    )

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

