import os 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the saved model for set prediction
path_for_saved_set_model = r"C:\Users\lf220\Desktop\mobilenet project\dataset_for_model\set_model.h5"
set_model = tf.keras.models.load_model(path_for_saved_set_model)

# Function to predict the set of the card
def classify_set(imageFile):
    img = Image.open(imageFile)
    img = img.resize((224, 224))
    x = np.expand_dims(image.img_to_array(img), axis=0)
    x = preprocess_input(x)

    # Predict the set
    pred = set_model.predict(x)
    set_predicted = np.argmax(pred, axis=1)[0]

    categories = os.listdir(r"C:\Users\lf220\Desktop\mobilenet project\dataset_for_model\train")
    return categories[set_predicted]

# Now predict the card ID after predicting the set
def classify_card_id(imageFile, set_name):
    # Load the card ID prediction model for the predicted set
    path_for_saved_card_id_model = r"C:\Users\lf220\Desktop\mobilenet project\dataset_for_model\card_id_model.h5"
    card_id_model = tf.keras.models.load_model(path_for_saved_card_id_model)

    img = Image.open(imageFile)
    img = img.resize((224, 224))
    x = np.expand_dims(image.img_to_array(img), axis=0)
    x = preprocess_input(x)

    # Predict the card ID within the set
    pred = card_id_model.predict(x)
    card_id = np.argmax(pred, axis=1)[0]

    # Map the card ID to actual name (you need to map it accordingly)
    card_ids_map = os.listdir(os.path.join('dataset_for_model', 'train', set_name))
    return card_ids_map[card_id]

# Test the whole process
imagePath = r"C:\Users\lf220\Desktop\mobilenet project\test_images\squirtle.jpg"

# Step 1: Predict the set
predicted_set = classify_set(imagePath)
print(f"Predicted Set: {predicted_set}")

# Step 2: Predict the card ID in the predicted set
predicted_card_id = classify_card_id(imagePath, predicted_set)
print(f"Predicted Card ID: {predicted_card_id}")

