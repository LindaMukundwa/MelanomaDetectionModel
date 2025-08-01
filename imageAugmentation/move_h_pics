# Moving healthy pics to our dataset
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

import os
import random
import shutil

folder_path = '/content/drive/MyDrive/AI4ALL-Group-1A-Datasets-Finalized/Dataset2Healthy/Augmented'
destination_folder = '/content/drive/MyDrive/AI4ALL-Group-1A-Datasets-Finalized/Test1-1000'

all_images = [f for f in os.listdir(folder_path) if (f.lower().endswith('.jpg') or f.lower().endswith('.jpeg'))]

# Randomly select 130 images
selected_images = random.sample(all_images, 130)

# Move the selected images to the destination folder
for image_name in selected_images:
    src = os.path.join(folder_path, image_name)
    dst = os.path.join(destination_folder, image_name)
    shutil.move(src, dst)

print(f"Moved {len(selected_images)} images to {destination_folder}")
