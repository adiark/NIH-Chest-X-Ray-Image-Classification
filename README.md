# Chest X-ray Image Classification

This repository contains the code and resources for a chest X-ray image classification project. The primary goal of this project is to develop a machine learning tool to identify specific diseases, such as atelectasis, effusion, and infiltration, from chest X-ray images. We used a Support Vector Machine (SVM) classifier with a RBF kernel (SVC) as our best-performing model and fine-tuned its hyperparameters to optimize its performance.

## Table of Contents

1. [Data](#data)
2. [Dependencies](#dependencies)
3. [Code Structure](#code-structure)
4. [Usage](#usage)
5. [Results](#results)

## Data

The dataset used in this project is the ChestX-ray8 dataset, which contains 108,948 frontal-view chest X-ray images of 32,717 unique patients. The dataset can be found [here](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community).

## Dependencies

To set up the project, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Keras
- scikit-learn
- NumPy
- pandas
- Matplotlib
- Seaborn

## Code Structure

The repository is organized into the following folders and files:

- `EDA`: Contains the code for Exploratory data analysis, data visualization.
- `Preprocessing and baseline models`: Contains the code for data preprocesing, normalization, feature extraction using Principal Component Analysis (PCA) and baseline models, Code for evaluating various classifiers, such as SVM, Decision Trees. Hyperparameter tuning and metrics calculation for SVC (final best performing model).
- `neural_network`: Contains custom CNN with hyperparameter tuning on sub set of data. Around 1000 images per group.
- `app.py`: Contains a simple user interface using streamlit for end user to upload and test the images for diseases classification. Click [here](https://adiark-nih-chest-x-ray-image-classification-app-9g3efy.streamlit.app/) to visit the application page.
- `requirements.txt`: Requirement files for streamlit application
- `trained_models`: Contains pickel files for trained SVC models and PCA for the diseases.
- `data`: Contains image and patient related information for the dataset
- `images`: Sample images to test model implementation on the streamlit application
- `img`: Images used in the report

## Usage

1. Clone the repository to your local machine.
2. Download the ChestX-ray8 dataset and place it in a folder named `images` within the project directory.
3. Install the necessary dependencies.
4. Run the code for results.
5. Test the pretrained models using streamlit application [here](https://adiark-nih-chest-x-ray-image-classification-app-9g3efy.streamlit.app/) 

## Results

Our final model, the SVM classifier with a Linear/RBF kernel (SVC), achieved classification accuracy scores of about 70%, 71%, and 75% for infiltration, atelectasis, and effusion, respectively.
