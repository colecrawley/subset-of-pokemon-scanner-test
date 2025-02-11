import os
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Raw paths for training and validation directories
train_path = r"C:\Users\lf220\Desktop\mobilenet project\dataset_for_model\train"
validate_path = r"C:\Users\lf220\Desktop\mobilenet project\dataset_for_model\validate"

# Path to the saved set prediction model
path_for_saved_set_model = r"C:\Users\lf220\Desktop\mobilenet project\dataset_for_model\set_model.h5"

# Check if the set prediction model already exists
if os.path.exists(path_for_saved_set_model):
    print("Loading pre-trained Set Prediction Model...")
    model = tf.keras.models.load_model(path_for_saved_set_model)
else:
    print("Training Set Prediction Model...")

    # Set Prediction Model 
    trainGenerator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        train_path, target_size=(224, 224), batch_size=30
    )
    validGenerator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        validate_path, target_size=(224, 224), batch_size=30
    )

    baseModel = MobileNetV2(weights='imagenet', include_top=False)
    x = baseModel.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    predictLayer = Dense(5, activation='softmax')(x)  # 5 is  number of sets, change it if needed
    model = Model(inputs=baseModel.input, outputs=predictLayer)

    # Freeze mobilenet pre-trained layers
    for layer in model.layers[:-5]:
        layer.trainable = False

    # Compile 
    epochs = 10
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    # Train set prediction model
    model.fit(trainGenerator, validation_data=validGenerator, epochs=epochs)

    # Save set prediction model
    model.save(path_for_saved_set_model)
    print(f"Set Prediction Model Trained and Saved at {path_for_saved_set_model}.")





# Card ID Model Creation

import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import numpy as np
from PIL import Image
import pathlib

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image

# Function to train the Card ID model with LOOCV for each set
def train_card_id_model_loocv(set_name, image_paths_with_ids):
    print(f"Training Card ID Model with LOOCV for set: {set_name}")

    # Card ID Model
    baseModel = MobileNetV2(weights='imagenet', include_top=False)
    x = baseModel.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    num_cards = len(image_paths_with_ids)
    predictLayer = Dense(num_cards, activation='softmax')(x)

    card_id_model = Model(inputs=baseModel.input, outputs=predictLayer)

    for layer in card_id_model.layers[:-5]:
        layer.trainable = False

    # Compile 
    optimizer = Adam(learning_rate=0.0001)
    card_id_model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    # Create a mapping from card_id to index
    card_id_to_index = {card_id: idx for idx, (img_path, card_id) in enumerate(image_paths_with_ids)}

    # Loop over all images for LOOCV
    for i, (test_image_path, test_card_id) in enumerate(image_paths_with_ids):
        print(f"Running LOOCV iteration {i + 1} / {num_cards}")

        test_img = Image.open(test_image_path)
        test_img = test_img.convert("RGB")  # Ensure it's in RGB mode
        test_img = test_img.resize((224, 224))
        x_test = np.expand_dims(image.img_to_array(test_img), axis=0)
        x_test = preprocess_input(x_test)

        # Create the train set (all images)
        train_images_data = []
        train_labels = []
        for j, (img_path, card_id) in enumerate(image_paths_with_ids):
            if i != j:  # Skip left-out image
                img = Image.open(img_path)
                img = img.convert("RGB")  # Ensure it's in RGB mode
                img = img.resize((224, 224))
                x_train = np.expand_dims(image.img_to_array(img), axis=0)
                x_train = preprocess_input(x_train)
                train_images_data.append(x_train)
                train_labels.append(card_id_to_index[card_id])

        X_train = np.vstack(train_images_data)

        y_train = np.zeros((len(X_train), num_cards))  # one-hot encoding
        for idx, label in enumerate(train_labels):
            y_train[idx, label] = 1  # Set the label to the actual card ID index

        # Train the model for this iteration
        card_id_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # After training, evaluate on the left-out test image
        test_prediction = card_id_model.predict(x_test)
        predicted_class = np.argmax(test_prediction, axis=1)[0]

        predicted_card_id = list(card_id_to_index.keys())[list(card_id_to_index.values()).index(predicted_class)]

        # Check if the prediction is correct or incorrect
        if predicted_card_id == test_card_id:
            result = "CORRECT"
        else:
            result = "INCORRECT"

        # Display the results
        print(f"Predicted card ID for the left-out card (image {i + 1}): {predicted_card_id}")
        print(f"Actual card ID for the left-out card: {test_card_id}")
        print(f"Prediction is {result}\n")  # Print whether the prediction is correct or incorrect

        # Show the image with actual and predicted labels
        plt.imshow(test_img)
        plt.title(f"Actual: {test_card_id}, Predicted: {predicted_card_id}\n{result}")
        plt.axis('off')  
        plt.draw()
        plt.show(block=False)
        plt.pause(2) 
        plt.close() 

    # Save the trained Card ID model
    card_id_model_path = os.path.join(r"C:\Users\lf220\Desktop\mobilenet project\dataset_for_model", f"{set_name}_model.h5")
    card_id_model.save(card_id_model_path)
    print(f"Card ID Model for set {set_name} trained with LOOCV and saved at {card_id_model_path}")




# Function to extract image paths and their corresponding IDs from the CSV file
def get_image_paths_and_ids_from_csv(csv_folder):
    image_paths_with_ids = []
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    for csv_file in csv_files:
        csv_file_path = os.path.join(csv_folder, csv_file)
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:] 
            for line in lines:
                parts = line.strip().split(',')
                card_id = parts[0]  # Treat 'id' as a string
                image_path = parts[4]  
                image_paths_with_ids.append((image_path, card_id))  # Store as a tuple of image path and ID
    return image_paths_with_ids




csv_folder = r'C:\Users\lf220\Desktop\mobilenet project\151_data'

# Get the image paths with card IDs from all CSV files in the folder
image_paths_with_ids = get_image_paths_and_ids_from_csv(csv_folder)

# Extract unique sets from the image paths
sets = set(os.path.dirname(img_path) for img_path, _ in image_paths_with_ids)


failed_sets = []

# Call the LOOCV function for each set
for set_name in sets:
    set_image_paths_with_ids = [(img_path, card_id) for img_path, card_id in image_paths_with_ids if os.path.dirname(img_path) == set_name]

    # Open and test images in the set
    print(f"Checking images in set: {set_name}")
    all_images_loaded = True  # check if all images are loaded
    for img_path, _ in set_image_paths_with_ids:
        img = Image.open(img_path)
        if img is not None:
            print(f"Successfully opened: {img_path}")
        else:
            print(f"Failed to open: {img_path}")
            all_images_loaded = False

    # If images loaded, proceed with training, otherwise mark the set as failed
    if all_images_loaded:
        print(f"All images in set {set_name} successfully loaded. Proceeding with training...\n")
        train_card_id_model_loocv(set_name, set_image_paths_with_ids)
    else:
        print(f"Some images in set {set_name} failed to load. Skipping training for this set.\n")
        failed_sets.append(set_name)

# Final report on which sets failed
if failed_sets:
    print(f"The following sets failed to load properly: {', '.join(failed_sets)}")
else:
    print("All sets loaded successfully.")






















