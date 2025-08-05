# Melanoma AI Detection Tool 🩺

Tested and created a model based on a CNN and ABCD criteria to act as a medical tool for identifying signs of melanoma on skin, all within the AI4ALL's cutting-edge AI4ALL Ignite accelerator.

<img width="1913" height="939" alt="Screenshot 2025-08-03 at 17 50 23" src="https://github.com/user-attachments/assets/e1081552-3a15-46df-937c-03ec8e45f3f5" />

<img width="1373" height="683" alt="Screenshot 2025-08-04 at 19 11 51" src="https://github.com/user-attachments/assets/b67483e5-bcbb-45e0-931b-f121ef0d90b6" />

# Problem Statement 

Our research question is **How well can a CNN model trained on ABCD visual criteria detect early signs of melanoma?** and one of our main reasons for embarking on this project was because melanoma is one of the deadliest forms of skin cancer, however when detected early, the 5-year survival rate for melanoma is **99%**.

In order to try and come up with a tool for this, we thought of using the ABCD markers to do these because they repressent 4 things to look out for that can detect signs of melanoma. These are:

**A** is for asymmetry where half of the spot is not symmetrical to the other

**B** is for border where there’s an irregular or undefined border on the spot

**C** is for color where there are varying colors present on the spot in different areas

**D** is for diameter where most melanomas are usually 6mm in diameter or smaller

This type of research and this type of work is very helpful because it can help to advance work in this area so that people can have access to resources like this and from all types of backgrounds and skin tones. Its impacts will show the importance of this kind of work and the need for responsible and impactful AI.


# Key Results 📊
Here are some of the plots we created from the results of our project to show the performance of the model. 
<img width="1189" height="390" alt="download" src="https://github.com/user-attachments/assets/4841b0e3-d38d-4ba7-97cf-9b89723885bd" />

<img width="1489" height="589" alt="download (1)" src="https://github.com/user-attachments/assets/701298a0-d147-42bf-9d3e-2d7f605d5e50" />

<img width="1589" height="1180" alt="download (2)" src="https://github.com/user-attachments/assets/6aaf91f0-f5ed-4df4-85aa-168887638566" />

These show the results of the combined model on different metrics and in comparison to the CNN and ABCD model as well. 

In order to accomplish this, we used ResNet50 and ImageNet to train the CNN based off the Gaussian blur and Otsu's threasholding to get key differences in the imaging. Then, we did a comprehensive training for the ABCD model based off different markers all with 15 epochs to learn for each model. Finally, a fusion between the CNN and ABCD was the combined model which endeed up exceeding the performance of all prior models. We accomplished feats like 96.4% AUC score as well as other key models from out 70/15/15 split of 1000 images. 

# Data Sources 🔗
Here is our base kaggle data source that was used in our project. The other was a custom one in our Google Drive. 

Link: https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset 

EXAMPLE: Kaggle Datasets: Link to Kaggle Dataset

# Installation 📦

In order to use this model locally, make sure to start the python backend, check the requirements are all met then use streamlit to run the app. The following commands detail how to do this one by one after cloning this repository locally and opening your IDE.

```bash
source melanoma-env/bin/activate
pip install -r requirements.txt
python setup_development.py
streamlit run app.py
```

# Technologies Used 
These are a list of some of the technologies, libraries, and frameworks used in our project.

Python

pandas

Streamlit

Google Colab

Tensor Flow

Resnet50 & ImageNet

CV2

Numpy

Matplotlib

# Authors and Contributors 👩‍💻

This project was completed in collaboration with:
 
Linda Mukundwa (Linda.Mukundwa1@marist.edu)

Serena Chen (schen172@u.rochester.edu)

Erica Okeh (erica.okeh@bison.howard.edu)

Marcellino Melika (marcellinomelika.college@gmail.com)
