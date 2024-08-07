import os
import pandas as pd

def collect_metadata(image_dir, labels, timestamps, radiation_levels):
    metadata = {
        "file_path": [],
        "label": [],
        "timestamp": [],
        "radiation_level": []
    }

    image_files = sorted(os.listdir(image_dir))  # Ensure the files are sorted to match your metadata lists

    for img_file, label, timestamp, radiation_level in zip(image_files, labels, timestamps, radiation_levels):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):  # Add other image formats if necessary
            img_path = os.path.join(image_dir, img_file)
            metadata["file_path"].append(img_path)
            metadata["label"].append(label)
            metadata["timestamp"].append(timestamp)
            metadata["radiation_level"].append(radiation_level)

    return pd.DataFrame(metadata)

# Example usage
image_dir = 'path/to/images'
labels = [1, 0, 1, 0, 1]  # List of labels corresponding to each image
timestamps = ['2024-01-01T00:00:00Z', '2024-01-01T00:30:00Z', '2024-01-01T01:00:00Z', '2024-01-01T01:30:00Z', '2024-01-01T02:00:00Z']  # List of timestamps
radiation_levels = [0.5, 0.7, 0.6, 0.8, 0.4]  # List of radiation levels

metadata_df = collect_metadata(image_dir, labels, timestamps, radiation_levels)

# Save to train.tsv
metadata_df.to_csv('train.tsv', sep='\t', index=False)

print('train.tsv file created successfully!')
