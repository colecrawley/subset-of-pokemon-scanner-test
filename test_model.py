import os
import random
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import re
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.models import load_model


train_path = r"C:\Users\Cole\Desktop\subset-of-pokemon-scanner-test\dataset_for_model\train"
validate_path = r"C:\Users\Cole\Desktop\subset-of-pokemon-scanner-test\dataset_for_model\validate"
path_for_saved_model = r"C:\Users\Cole\Desktop\subset-of-pokemon-scanner-test\dataset_for_model\setModel.h5"
metrics_folder = r"C:\Users\Cole\Desktop\subset-of-pokemon-scanner-test\metrics_for_set_model"

# Function to check if model has been trained or not
def check_if_model_trained(model_path):
    if os.path.exists(model_path):
        try:
            # Try loading the model to check if it's valid
            model = load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print(f"Model not found at {model_path}")
        return None

# Check if model is already trained (exists and is valid)
existing_model = check_if_model_trained(path_for_saved_model)

if existing_model is not None:
    # If the model exists and is valid, use it for further tasks
    model = existing_model
else:
    # If the model doesn't exist or is invalid, proceed with training
    print("Training a new model...")

    # Create data generators for training and validation
    trainGenerator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        train_path, target_size=(224, 224), batch_size=30)
    validGenerator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        validate_path, target_size=(224, 224), batch_size=30)

    # Build the model
    baseModel = MobileNetV2(weights='imagenet', include_top=False)
    x = baseModel.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    predictLayer = Dense(5, activation='softmax')(x)

    model = tf.keras.Model(inputs=baseModel.input, outputs=predictLayer)
    print(model.summary())

    # Freeze the MobileNetV2 pre-trained layers
    for layer in model.layers[:-5]:
        layer.trainable = False

    # Compile the model
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    # Train the model
    epochs = 50
    history = model.fit(trainGenerator, validation_data=validGenerator, epochs=epochs)

    # Save the trained model
    model.save(path_for_saved_model)

    # Create the folder to save metrics if it doesn't exist
    os.makedirs(metrics_folder, exist_ok=True)

    # Plot the training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(metrics_folder, 'accuracy_plot.png'))

    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(metrics_folder, 'loss_plot.png'))

    print(f"Metrics plots saved to {metrics_folder}")






# Card ID Model Creation

# Function to handle filenames with apostrophes or underscores dynamically
def open_image_with_fallback(img_path):
    """Open the image with the original filename as it is."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    img = Image.open(img_path)
    img = img.convert("RGB")  # Ensure it's in RGB mode
    return img

def extract_numeric_part(card_id):
    match = re.search(r'-(\d+)$', card_id)
    if match:
        return int(match.group(1))  # Return the numeric part as an integer
    return 0  # If no numeric part, return 0 (or handle as needed)


# Function to save training metrics plot

def save_metrics_plot(history, set_name):
    # Define the path to save the plot directly in the card_id_metrics folder
    metrics_folder = r"C:\Users\Cole\Desktop\subset-of-pokemon-scanner-test\card_id_metrics"

    # Ensure the directory exists, otherwise create it
    if not os.path.exists(metrics_folder):
        os.makedirs(metrics_folder)

    # Sanitize the set_name to ensure it's just the set name without any subdirectories
    set_name_cleaned = os.path.basename(set_name)  # Get the base name of the set, removing any path

    # Define the path for saving the accuracy and loss plot
    combined_graph_path = os.path.join(metrics_folder, f"{set_name_cleaned}_accuracy_and_loss.png")
    
    # Print the combined graph path to ensure it's correct
    print(f"Saving plot to: {combined_graph_path}")

    # Plot the accuracy and loss
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['loss'], label='Loss')
    plt.title(f"Accuracy and Loss for {set_name_cleaned}")
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()

    # Save the plot
    plt.savefig(combined_graph_path)
    print(f"Metrics plot saved at {combined_graph_path}")
    plt.close()

def card_id_model_exists(set_name):
    card_id_model_path = os.path.join(r"card_id_models", f"{set_name}_model.h5")
    return os.path.exists(card_id_model_path)

# Function to train the Card ID model with LOOCV for each set
def train_card_id_model_loocv(set_name, image_paths_with_ids, rarities, unique_rarities, metrics_folder):
    print(f"Training Card ID Model with LOOCV for set: {set_name}")
    
    # Check if model already exists for this set
    if card_id_model_exists(set_name):
        print(f"Card ID model for set {set_name} already exists. Skipping training.")
        return  # Skip training if the model already exists

    baseModel = MobileNetV2(weights='imagenet', include_top=False)
    x = baseModel.output
    x = GlobalAveragePooling2D()(x)

    # Input layer for rarity
    rarity_input = Input(shape=(1,), name='rarity')

    # Concatenate image features and rarity
    combined = Concatenate()([x, rarity_input])

    # Fully connected layers
    x = Dense(512, activation='relu')(combined)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    num_cards = len(image_paths_with_ids)
    predictLayer = Dense(num_cards, activation='softmax')(x)

    card_id_model = Model(inputs=[baseModel.input, rarity_input], outputs=predictLayer)

    for layer in card_id_model.layers[:-5]:
        layer.trainable = False

    optimizer = Adam(learning_rate=0.0001)
    card_id_model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    card_id_to_index = {card_id: idx for idx, (img_path, card_id) in enumerate(image_paths_with_ids)}

    # Encode the rarities using LabelEncoder
    rarity_encoder = LabelEncoder()
    rarity_encoder.fit(list(unique_rarities))  # Fit the encoder on the unique rarities for the set
    rarities_array = rarity_encoder.transform(rarities)  # Encode rarities for the set

    # Define the ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,         # Random rotations
        width_shift_range=0.1,     # Randomly shift images horizontally
        height_shift_range=0.1,    # Randomly shift images vertically
        shear_range=0.1,           # Random shear transformations
        zoom_range=0.1,            # Random zoom
        horizontal_flip=True,      # Randomly flip images horizontally
        fill_mode='reflect'        # Fill pixels that are newly created by transformations
    )

    # Loop over all images for LOOCV
    for i, ((test_image_path, test_card_id), test_rarity) in enumerate(zip(image_paths_with_ids, rarities)):
        print(f"Running LOOCV iteration {i + 1} / {num_cards}.")

        try:
            test_img = open_image_with_fallback(test_image_path)
        except FileNotFoundError as e:
            print(str(e))
            continue  

        test_img = test_img.resize((224, 224))
        x_test = np.expand_dims(image.img_to_array(test_img), axis=0)
        x_test = preprocess_input(x_test)

        # Create the train set (all images except for the test image)
        train_images_data = []
        train_labels = []
        train_rarities = []

        # Loop to gather all images except the test image
        for j, ((img_path, card_id), rarity) in enumerate(zip(image_paths_with_ids, rarities)):
            if i != j:  
                try:
                    img = open_image_with_fallback(img_path)
                except FileNotFoundError:
                    continue  

                img = img.resize((224, 224))
                x_train = np.expand_dims(image.img_to_array(img), axis=0)
                x_train = preprocess_input(x_train)

                # Augment the training image using the data generator
                augmented_images = datagen.flow(x_train, batch_size=1, shuffle=False)

                # Add a fixed number of augmented images (e.g., 1 per original image)
                for augmented_img in augmented_images:
                    train_images_data.append(augmented_img[0])  # Add augmented image
                    train_labels.append(card_id_to_index[card_id])  # Corresponding label
                    train_rarities.append(rarity)  # Corresponding rarity
                    break  # Only add one augmented image per original image

        # Ensure that X_train and y_train have the same size
        X_train = np.array(train_images_data)
        y_train = np.zeros((len(X_train), num_cards))

        for idx, label in enumerate(train_labels):
            y_train[idx, label] = 1  # One-hot encoding

        # Convert rarities to numpy array (encoded)
        rarities_array = np.array(train_rarities).flatten()  # Flatten here to ensure it's 1D
        rarities_array = rarity_encoder.transform(rarities_array)  # Encode the rarities for training

        # Train the model for this iteration
        history = card_id_model.fit([X_train, rarities_array], y_train, epochs=20, batch_size=32, verbose=1)

        # Prepare test rarity
        test_rarity = np.array([test_rarity]).flatten()  # Flatten here as well
        test_rarity = rarity_encoder.transform(test_rarity)  # Encode test rarity

        # Predict the card ID and rarity
        test_prediction = card_id_model.predict([x_test, test_rarity])
        predicted_class = np.argmax(test_prediction, axis=1)[0]

        # Decode predicted card ID
        predicted_card_id = list(card_id_to_index.keys())[list(card_id_to_index.values()).index(predicted_class)]

        # Decode predicted rarity and actual rarity
        predicted_card_rarity = rarity_encoder.inverse_transform([test_rarity[0]])[0]  # Decode to string
        actual_card_rarity = rarity_encoder.inverse_transform([test_rarity[0]])[0]  # Decode to string

        # Compare both card ID and rarity
        result = "CORRECT" if (predicted_card_id == test_card_id) and (predicted_card_rarity == actual_card_rarity) else "INCORRECT"

        # Print the results with both card ID and rarity
        print(f"Predicted card ID for the left-out card (image {i + 1}): {predicted_card_id} (Rarity: {predicted_card_rarity})")
        print(f"Actual card ID for the left-out card: {test_card_id} (Rarity: {actual_card_rarity})")
        print(f"Prediction is {result}\n")

        # Show the image with actual and predicted labels
        plt.imshow(test_img)
        plt.axis('off')

        # Add the actual and predicted card ID and rarity to the title
        plt.title(f"Actual: {test_card_id}, Predicted: {predicted_card_id}\nRarity: {predicted_card_rarity}")

        # Display the result (CORRECT or INCORRECT) as text on the image
        plt.text(0.5, 0.95, f"Result: {result}", color='green' if result == "CORRECT" else 'red',
                ha='center', va='center', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)

        # Display the image and the result
        plt.draw()
        plt.show(block=False)
        plt.pause(2)
        plt.close()

    # Save model and metrics to the card_id_models folder
    card_id_model_path = os.path.join(r"card_id_models", f"{set_name}_model.h5")
    card_id_model.save(card_id_model_path)
    print(f"Card ID Model for set {set_name} trained with LOOCV and saved at {card_id_model_path}")

    save_metrics_plot(history, set_name)



def check_and_sort_csv(csv_file_path):
    """Check if the CSV file is sorted by the numeric part of the card ID. If not, sort and save it."""
    # Read the CSV file
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Skip header
    header = lines[0]
    rows = lines[1:]

    # Extract the card IDs from the rows and check if they are in order
    card_ids = [line.strip().split(',')[0] for line in rows]
    is_sorted = all(int(card_ids[i].split('-')[1]) <= int(card_ids[i+1].split('-')[1]) for i in range(len(card_ids) - 1))

    if not is_sorted:
        # If not sorted, sort the rows
        print(f"Sorting CSV: {csv_file_path}")  # Show which CSV is being sorted
        rows.sort(key=lambda x: int(x.strip().split(',')[0].split('-')[1]))  # Sort by the number after the '-'

        # Write the sorted data back to the same CSV
        with open(csv_file_path, 'w', encoding='utf-8') as f:
            f.write(header)  # Write header
            f.writelines(rows)  # Write sorted rows
    else:
        print(f"Skipping already sorted CSV: {csv_file_path}")  # Show which CSV is already sorted







# Function to extract image paths and their corresponding IDs from the CSV file
def get_image_paths_and_ids_and_rarities_from_csv(csv_folder, set_name=None):
    image_paths_with_ids = []
    rarities = []
    unique_rarities = set()  # Set to collect unique rarities

    # Get all the CSV files in the folder
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    for csv_file in csv_files:
        csv_file_path = os.path.join(csv_folder, csv_file)

        # Sort the CSV before reading
        check_and_sort_csv(csv_file_path)

        # Now read the sorted CSV
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(',')
                card_id = parts[0]  # 'id' is in the first column
                image_path = parts[4]  # 'image_url' is in the fifth column (index 4)
                rarity = parts[3]  # 'rarity' is in the fourth column (index 3)

                # Only include paths for the current set if set_name is specified
                if set_name and set_name not in image_path:
                    continue

                image_paths_with_ids.append((image_path, card_id))
                rarities.append(rarity)
                unique_rarities.add(rarity)

    return image_paths_with_ids, rarities, list(unique_rarities)



csv_folder = r'C:\Users\Cole\Desktop\subset-of-pokemon-scanner-test\151_data'
metrics_folder = r"C:\Users\Cole\Desktop\subset-of-pokemon-scanner-test\card_id_metrics"

# Now unpack all three values returned by the function
image_paths_with_ids, rarities, unique_rarities = get_image_paths_and_ids_and_rarities_from_csv(csv_folder)

# Extract the unique set names from the image paths
sets = set(os.path.dirname(img_path) for img_path, _ in image_paths_with_ids)

failed_sets = []

# Call the LOOCV function for each set
for set_name in sets:
    print(f"Checking images in set: {set_name}")
    
    # Get image paths, card IDs, and rarities for the specific set
    set_image_paths_with_ids, set_rarities, unique_rarities = get_image_paths_and_ids_and_rarities_from_csv(csv_folder, set_name)
    
    # Print unique rarities for the current set before training
    print(f"Unique rarities for set {set_name}: {unique_rarities}\n")
    
    all_images_loaded = True  # Track if all images for this set loaded successfully

    # Attempt to open all images
    for img_path, _ in set_image_paths_with_ids:
        try:
            img = open_image_with_fallback(img_path)
            print(f"Successfully opened: {img_path}")
        except FileNotFoundError as e:
            print(f"Error opening {img_path}: {str(e)}")
            all_images_loaded = False

    # Proceed with training if images are loaded, otherwise mark the set as failed
    if all_images_loaded:
        print(f"All images in set {set_name} successfully loaded. Proceeding with training...\n")
        train_card_id_model_loocv(set_name, set_image_paths_with_ids, set_rarities, unique_rarities, metrics_folder)
    else:
        print(f"Some images in set {set_name} failed to load. Skipping training for this set.\n")
        failed_sets.append(set_name)

# Final report on which sets failed
if failed_sets:
    print(f"The following sets failed to load properly: {', '.join(failed_sets)}")
else:
    print("All sets loaded successfully.")




