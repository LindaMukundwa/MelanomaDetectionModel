# Colab to augment images for dataset2
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

!pip install -q albumentations opencv-python-headless

import albumentations as A
import cv2
import os
import random
from glob import glob
import matplotlib.pyplot as plt
import uuid

folder_path = '/content/drive/MyDrive/AI4ALL-Group-1A-Datasets-Finalized/Dataset2Healthy'
augmented_folder = '/content/drive/MyDrive/AI4ALL-Group-1A-Datasets-Finalized/Dataset2Healthy/Augmented'
image_paths = sorted(glob(os.path.join(folder_path, '*.jpg')) +
                     glob(os.path.join(folder_path, '*.jpeg')))

# Define transformations we want from Albumentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),                # flips horizontally, same size
    A.RandomBrightnessContrast(p=0.5),      # brightness and contrast changes
    A.GaussNoise(p=0.3),                    # adds noise, no size change
    A.Blur(blur_limit=3, p=0.2),            # blur, no size change
])

# Function augment images
def augment_images(paths, n_images=5, n_augmentations=3):
    selected_paths = random.sample(paths, min(n_images, len(paths)))
    all_augmented_data = []

    for i, img_path in enumerate(selected_paths):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented_images = []
        for _ in range(n_augmentations):
            augmented = transform(image=image)
            augmented_images.append(augmented['image'])

        all_augmented_data.append((img_path, augmented_images))

        plt.figure(figsize=(12, 3))
        plt.subplot(1, n_augmentations + 1, 1)
        plt.imshow(image)
        plt.title("Original")
        plt.axis('off')

        for j, aug_img in enumerate(augmented_images):
            plt.subplot(1, n_augmentations + 1, j + 2)
            plt.imshow(aug_img)
            plt.title(f"Aug {j+1}")
            plt.axis('off')

        plt.suptitle(f"Image {i+1}")
        plt.show()

    return all_augmented_data

# Saving images to new folder called augmented
def save_images(augmented_data, save_folder=augmented_folder):
    for original_path, aug_images in augmented_data:
        base_name = os.path.splitext(os.path.basename(original_path))[0]

        for aug_img in aug_images:
            aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            unique_suffix = str(uuid.uuid4())[:8]
            new_filename = f"{base_name}_aug_{unique_suffix}.jpg"
            save_path = os.path.join(save_folder, new_filename)
            cv2.imwrite(save_path, aug_img_bgr)
            print(f"Saved augmented image: {new_filename}")

augmented_images = augment_images(image_paths)
save_images(augmented_images)
