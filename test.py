import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D, Input
from tensorflow.keras.models import Model
import cv2
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import pickle

# Load feature vectors and filenames
feature_list = np.array(pickle.load(open("featurevector.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))  # Renamed to avoid conflict

# Define the feature extraction model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalMaxPooling2D()(x)
feature_extraction_model = Model(inputs, x)

def extract_feature(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image at {img_path} could not be loaded.")
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = feature_extraction_model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return np.squeeze(normalized)

def find_similar_images(input_img_path, threshold=0.5):
    # Extract features for the input image
    input_features = extract_feature(input_img_path)
    
    # Initialize Nearest Neighbors
    neighbors = NearestNeighbors(n_neighbors=len(feature_list), algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    
    # Find nearest neighbors
    distances, indices = neighbors.kneighbors([input_features])
    
    similar_images = []
    input_image_index = -1
    
    # Identify the index of the input image in the feature list
    for i, fname in enumerate(filenames):  # Renamed for clarity
        if fname == input_img_path:
            input_image_index = i
            break
    
    # Collect similar images, skipping the input image itself
    for i, distance in enumerate(distances[0]):
        if indices[0][i] != input_image_index and distance < threshold:
            similar_images.append((indices[0][i], distance))
    
    # Sort the results based on distance
    similar_images = sorted(similar_images, key=lambda x: x[1])
    
    # Return top 5 similar images
    return similar_images[1:6]

# Example usage
input_img_path ="C:/Users/MANASWINI KARNATAKA/Downloads/2.JPG"  # Change this to your input image path
threshold = 0.9
similar_images = find_similar_images(input_img_path, threshold)

print(f"Found {len(similar_images)} similar images:")
for idx, dist in similar_images:
    print(f"Image: {filenames[idx]}, Distance: {dist:.2f}")
