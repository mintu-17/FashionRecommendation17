import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D, Input
from tensorflow.keras.models import Model
import cv2
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import pickle
from PIL import Image
import streamlit as st
import io

# Load feature vectors and filenames
feature_list = np.array(pickle.load(open("featurevector.pkl", "rb")))
filename = pickle.load(open("filenames.pkl", "rb"))

# Define the feature extraction model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalMaxPooling2D()(x)
feature_extraction_model = Model(inputs, x)

def extract_feature(img):
    img = np.array(img)  # Convert PIL image to NumPy array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    img = cv2.resize(img, (224, 224))
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = feature_extraction_model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return np.squeeze(normalized)

def find_similar_images(input_img, threshold=0.8):
    # Extract features for the input image
    input_features = extract_feature(input_img)
    
    # Initialize Nearest Neighbors
    neighbors = NearestNeighbors(n_neighbors=len(feature_list), algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    
    # Find nearest neighbors
    distances, indices = neighbors.kneighbors([input_features])
    
    similar_images = []
    
    # Collect similar images, skipping the input image itself
    for i, distance in enumerate(distances[0]):
        if distance < threshold:
            similar_images.append((indices[0][i], distance))
    
    # Sort the results based on distance
    similar_images = sorted(similar_images, key=lambda x: x[1])
    
    # Return similar images
    return similar_images

# Streamlit interface
st.title("Image Similarity Search")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open and display the uploaded image
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Find similar images
    similar_images = find_similar_images(uploaded_image, threshold=1)
    
    # Display results
    if len(similar_images) >= 6:
        images_to_display = similar_images[1:6]  # Skip the first image
    else:
        images_to_display = similar_images[1:]  # Skip the first image if fewer than 6 images found
    
    st.write(f"Found {len(similar_images)} similar images:")
    
    # Use a set to avoid duplicates
    displayed_indices = set()
    
    # Create a list of columns dynamically based on the number of images
    num_images = len(images_to_display)
    columns = st.columns(num_images) if num_images > 0 else []
    
    # Display images in columns
    for i, (idx, dist) in enumerate(images_to_display):
        if idx not in displayed_indices:  # Check if the image has already been displayed
            if i < len(columns):
                with columns[i]:
                    st.image(filename[idx], caption=f"Distance: {dist:.2f}")
                    displayed_indices.add(idx)  # Add index to the set of displayed images
