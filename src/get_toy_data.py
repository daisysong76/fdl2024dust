from google.cloud import storage
import os

# Initialize the GCS client
client = storage.Client()

# Define the bucket name and file name
bucket_name = 'us-hl-dosi-data'
source_blob_name = 'path/to/your/file.txt'
destination_file_name = '/local/path/to/file.txt'

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

# Download the file
download_blob(bucket_name, source_blob_name, destination_file_name)

# Now you can use the file locally
with open(destination_file_name, 'r') as file:
    data = file.read()
    print(data)

# If you want to process the file without downloading
def read_blob(bucket_name, source_blob_name):
    """Reads a blob from the bucket directly."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    data = blob.download_as_text()

    return data

# Read the file content directly from GCS
file_content = read_blob(bucket_name, source_blob_name)
print(file_content)
