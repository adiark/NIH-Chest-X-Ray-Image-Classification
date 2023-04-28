from scipy.ndimage import gaussian_filter
from tf_explain.core.grad_cam import GradCAM
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pickle
from PIL import Image


def load_and_normalize_image(image, target_size=(512, 512)):
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    elif len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1):
        img = img_array
    else:
        raise ValueError("Invalid number of channels in the input image.")

    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    normalized_img = img_resized / 255.0
    return normalized_img


st.set_page_config(page_title="Medical Image Prediction",
                   layout="centered", initial_sidebar_state="expanded")

st.title("Medical Image Prediction")

st.sidebar.title("Input Patient Information")
age = st.sidebar.number_input(
    "Patient Age", min_value=0, max_value=120, value=30, step=1)
gender = st.sidebar.selectbox("Patient Gender", options=["M", "F"])
view_position = st.sidebar.selectbox("View Position", options=["PA", "AP"])

patient_data = {
    "Patient Age": [age],
    "Patient Gender": [gender],
    "View Position": [view_position],
}

uploaded_file = st.file_uploader("Upload an image", type=["png"])


def apply_heatmap(image, sigma=5, alpha=0.5):
    img_array = np.array(image)

    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    elif len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1):
        img_gray = img_array
    else:
        raise ValueError("Invalid number of channels in the input image.")

    img_blurred = gaussian_filter(img_gray, sigma=sigma)
    img_diff = np.abs(img_gray - img_blurred)

    # Normalize the difference image
    img_diff = (img_diff - img_diff.min()) / (img_diff.max() - img_diff.min())

    # Create a heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * img_diff), cv2.COLORMAP_JET)

    # Blend the original image with the heatmap
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        heatmap_overlay = cv2.addWeighted(
            img_array, 1 - alpha, heatmap, alpha, 0)
    else:
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        heatmap_overlay = cv2.addWeighted(
            img_rgb, 1 - alpha, heatmap, alpha, 0)

    return heatmap_overlay


# Load the pre-trained SVC models
with open("./trained_models/Infiltration_svc_model.pkl", "rb") as file:
    svc_model_disease1 = pickle.load(file)

with open("./trained_models/Effusion_svc_model.pkl", "rb") as file:
    svc_model_disease2 = pickle.load(file)

with open("./trained_models/Atelectasis_svc_model.pkl", "rb") as file:
    svc_model_disease3 = pickle.load(file)

# Load PCA objects
with open("./trained_models/Infiltration_pca.pkl", "rb") as file:
    pca_disease1 = pickle.load(file)

with open("./trained_models/Effusion_pca.pkl", "rb") as file:
    pca_disease2 = pickle.load(file)

with open("./trained_models/Atelectasis_pca.pkl", "rb") as file:
    pca_disease3 = pickle.load(file)

# Load preprocessing objects
with open("./trained_models/Infiltration_preprocessor.pkl", "rb") as file:
    preprocessor_disease1 = pickle.load(file)

with open("./trained_models/Effusion_preprocessor.pkl", "rb") as file:
    preprocessor_disease2 = pickle.load(file)

with open("./trained_models/Atelectasis_preprocessor.pkl", "rb") as file:
    preprocessor_disease3 = pickle.load(file)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image", use_column_width=True)
    img_normalized = load_and_normalize_image(image)
    prediction_table = pd.DataFrame(columns=["Disease", "Prediction"])

    patient_df = pd.DataFrame(patient_data, index=[0])

    for disease, model, preprocessor, pca in [
        ("Infiltration", svc_model_disease1, preprocessor_disease1, pca_disease1),
        ("Effusion ", svc_model_disease2, preprocessor_disease2, pca_disease2),
        ("Atelectasis ", svc_model_disease3, preprocessor_disease3, pca_disease3),
    ]:
        X_patient = preprocessor.transform(patient_df)
        X_image_flat = img_normalized.flatten().reshape(1, -1)
        X_combined = np.hstack((X_image_flat, X_patient))

        # Apply PCA transformation
        X_pca = pca.transform(X_combined)

        prediction = model.predict(X_pca)
        prediction_text = "Positive" if prediction[0] == 1 else "Negative"
        new_row = pd.DataFrame(
            {"Disease": [disease], "Prediction": [prediction_text]})
        prediction_table = pd.concat(
            [prediction_table, new_row], ignore_index=True)

        if prediction[0] == 1:
            heatmap_image = apply_heatmap(image)

            # Display the original image and the heatmap visualization
            st.image(heatmap_image, caption="Heatmap Visualization",
                     use_column_width=True)
    st.table(prediction_table)

else:
    st.warning("Please upload an image.")
