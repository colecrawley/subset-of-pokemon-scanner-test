import os
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# List categories (sets) and sort
categories = os.listdir(r"C:\Users\Cole\Desktop\subset-of-pokemon-scanner-test\dataset_for_model\train")
categories.sort()
print(categories)

# Load the saved model
path_for_saved_model = r"C:\Users\Cole\Desktop\subset-of-pokemon-scanner-test\dataset_for_model\setModel.h5"
model = tf.keras.models.load_model(path_for_saved_model)

print(model.summary())

# Define image classification function
def classfiy_image(imageFile):
    x = []

    img = Image.open(imageFile)
    img.load()
    img = img.resize((224, 224))  # Resize image to match MobileNetV2 input size

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)  # Preprocess the image (e.g., subtract mean values)

    print(x.shape)

    pred = model.predict(x)
    categoryValue = np.argmax(pred, axis=1)

    categoryValue = categoryValue[0]
    print(categoryValue)

    result = categories[categoryValue]

    return result

# Test the image classification
imagePath = r"C:\Users\Cole\Desktop\subset-of-pokemon-scanner-test\test_images\fearow.jpg"
resultText = classfiy_image(imagePath)
print(resultText)

img = cv2.imread(imagePath)
img = cv2.putText(img, resultText, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

################################################

# Card ID and rarity prediction functions

# Function to get card IDs and rarities from the CSV file of the predicted set
def get_card_id_and_rarities_from_csv(set_name):
    # Construct the path to the CSV file for the set
    csv_path = os.path.join(r"C:\Users\Cole\Desktop\subset-of-pokemon-scanner-test\151_data", f"{set_name}_cards.csv")
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Get the card IDs (as they are) and rarities directly
    card_ids = df['id'].values  # Assuming 'id' column exists in your CSV
    rarities = df['rarity'].unique()  # Assuming a 'rarity' column exists
    
    return card_ids, rarities

# Function to load card ID model
def load_card_id_model(set_name):
    # Path for the set-specific card ID model (assuming it's stored similarly as the set model)
    card_id_model_path = os.path.join(r"C:\Users\Cole\Desktop\subset-of-pokemon-scanner-test\card_id_models", f"{set_name}_model.h5")
    card_id_model = tf.keras.models.load_model(card_id_model_path)
    
    return card_id_model

# Function to predict the card ID and rarity
def predict_card_and_rarity(imageFile):
    # Predict the set first
    predicted_set = classfiy_image(imageFile)
    
    # Load the card IDs and rarities for the predicted set from the CSV file
    card_ids, rarities = get_card_id_and_rarities_from_csv(predicted_set)
    
    # Load the card ID model for the predicted set
    card_id_model = load_card_id_model(predicted_set)
    
    # Preprocess the image for prediction
    img = Image.open(imageFile).convert("RGB")
    img = img.resize((224, 224))
    x = np.expand_dims(image.img_to_array(img), axis=0)
    x = preprocess_input(x)
    
    # For this example, we'll assume rarity is a constant or known value (this is just an example).
    # If rarity is categorical, convert it into a one-hot encoded vector or something that matches the model.
    rarity = np.array([0])  # Modify this if you have more details on how the rarity is encoded.
    
    # Ensure that both inputs are passed
    prediction = card_id_model.predict([x, rarity])  # Pass both inputs (image and rarity)

    # Get the predicted card ID (index of highest probability)
    predicted_card_id_index = np.argmax(prediction[0, :-1])  # Assuming the last output is the rarity
    predicted_card_id = card_ids[predicted_card_id_index]
    
    # Get the predicted rarity (last value of the prediction)
    predicted_rarity_index = np.argmax(prediction[0, -1])  # Assuming last value represents rarity
    
    # Decode the rarity from the possible rarities
    predicted_card_rarity = rarities[predicted_rarity_index]

    return predicted_set, predicted_card_id, predicted_card_rarity, img


# Example usage:
image_file = r"C:\Users\Cole\Desktop\subset-of-pokemon-scanner-test\test_images\fearow.jpg"  # Replace with your image file path

# Predict set, card ID, and rarity
predicted_set, predicted_card_id, predicted_card_rarity, img = predict_card_and_rarity(image_file)

# Display the image and predictions
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis('off')  # Hide axis
plt.title(f"Predicted Set: {predicted_set}\nPredicted Card ID: {predicted_card_id}")
plt.show()
