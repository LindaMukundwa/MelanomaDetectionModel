This folder contains all the code used to create augmentations to existing images. This is because we only had 60 images of healthy skin. For our model training, we would need at least 500 images of healthy skin. 

To clarify, our complete dataset consists of 500 healthy skin images and 500 skin with melanoma images. 

imgAlbumentation contains the code inspired by the albumentations.ai document. For our transformations, we chose to keep the image the same size as to prevent any black space in the background. The program's task was to randomly choose a combination of these transformations: horizontally flip the image, change in brightness and color constrast, add noise, or blur image. 

Our images were sorted in Google Drive so Google Colab was used to quickly sort the images into their respective folders. 

move_h_pics contains the code to move healthy skin images into our dataset folder. 
move_m_pics contains the code to move images from the skin cancer kaggle dataset (by: Farjana Kabir) into our dataset folder. 

Once in our dataset folder, we had to divide the images into their class folders for the model. 
h_pics_training contains the code used to help us move healthy skin images to the training folder.
m_pics_training contains the code used to help us move skin with melanoma images into the training folder. 
validateTestFolders contains code that instructs the program to randomly choose a specified number of images and sort them evenly into both the validate and test folders. 

This completes our expectation to use 70% of our images for training, 15% for validation, and 15% for testing. 
