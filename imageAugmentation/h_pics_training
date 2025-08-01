# Moving healthy pics to training folder (change amounts and folder paths if needed)
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

import os
import random
import shutil

folder_path = '/content/drive/MyDrive/AI4ALL-Group-1A-Datasets-Finalized/Test1-1000'
destination_folder = '/content/drive/MyDrive/AI4ALL-Group-1A-Datasets-Finalized/Test1-1000/TotalTraining/Healthy-350'

all_images = [f for f in os.listdir(folder_path) if (f.lower().endswith('.jpg') or f.lower().endswith('.jpeg'))]

# Randomly select 91 images
selected_images = random.sample(all_images, 91)

# Move the selected images to the destination folder
for image_name in selected_images:
    src = os.path.join(folder_path, image_name)
    dst = os.path.join(destination_folder, image_name)
    shutil.move(src, dst)

print(f"Moved {len(selected_images)} images to {destination_folder}")
