import os
import numpy as np
import pandas as pd
from PIL import Image

class DataLoader:
    def __init__(self, image_dir, csv_file, image_size=(512, 512)):
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.image_size = image_size

    def load_images(self):
        images = []
        for img_file in os.listdir(self.image_dir):
            if img_file.endswith('.png'):  # Assuming images are in PNG format
                img_path = os.path.join(self.image_dir, img_file)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.image_size)  # Resize images if needed
                images.append(np.array(img))
        return np.array(images)

    def load_radiation_data(self):
        df = pd.read_csv(self.csv_file)
        return df['radiation'].values  # Assuming there's a 'radiation' column

    def load_data(self):
        images = self.load_images()
        radiation_data = self.load_radiation_data()
        return images, radiation_data

# Example usage
if __name__ == "__main__":
    image_dir = 'path/to/images'
    csv_file = 'path/to/radiation.csv'
    dataloader = DataLoader(image_dir, csv_file)

    images, radiation_data = dataloader.load_data()

    # Print shapes for verification
    print(f'Loaded {images.shape[0]} images with shape {images.shape[1:]}')
    print(f'Loaded radiation data with shape {radiation_data.shape}')
