import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import preprocess_input
from keras import layers, models, regularizers
from keras.optimizers import Adam
from keras.metrics import SparseCategoricalAccuracy
from keras.callbacks import ReduceLROnPlateau
from keras.applications import MobileNetV2
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

# Load the CSV file containing card data
csv_file_path = r"C:\Users\lf220\Desktop\mobilenet project\151_data\sv3pt5_cards.csv"
df = pd.read_csv(csv_file_path)

# List all image files in the image directory
image_dir = r"C:\Users\lf220\Desktop\mobilenet project\151_images"
image_files = os.listdir(image_dir)

# Map each image filename to its corresponding card ID
image_paths = []
labels = []

for _, row in df.iterrows():
    card_id = row['id']
    image_filename = f"{row['name']}_{card_id}.jpg"
    image_path = os.path.join(image_dir, image_filename)

    if os.path.exists(image_path):
        image_paths.append(image_path)
        labels.append(card_id)  

# Create a new DataFrame with image paths and labels
df_images = pd.DataFrame({'image_path': image_paths, 'id': labels})

# Map unique card IDs to enumerated labels
unique_ids = sorted(df_images['id'].unique())
id_to_label = {card_id: idx for idx, card_id in enumerate(unique_ids)}

# Replace IDs with enumerated labels
df_images['label'] = df_images['id'].map(id_to_label)

# Shuffle and split into train/validation (80-20 split)
df_images = shuffle(df_images, random_state=42)
train_df, val_df = train_test_split(df_images, test_size=0.2, random_state=42)

# ImageDataGenerator for preprocessing
train_image_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='reflect'
)

val_image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Compute class weights
train_labels = train_df['label'].values
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = {label: weight for label, weight in zip(np.unique(train_labels), class_weights)}

# Create training and validation data generators
train_generator = train_image_gen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="image_path",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',  # Outputs integer labels instead of one-hot encoding
    shuffle=True
)

val_generator = val_image_gen.flow_from_dataframe(
    dataframe=val_df,
    directory=None,
    x_col="image_path",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    shuffle=False
)

# MobileNetV2 Model
feature_extractor = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
feature_extractor.trainable = False  # Freeze MobileNetV2

model = models.Sequential([
    feature_extractor,
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(len(unique_ids), activation='softmax', trainable=True)  # Output layer
])

# Compile model with sparse categorical cross-entropy
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=[SparseCategoricalAccuracy()]
)

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=15,
    min_lr=1e-6,
    verbose=1
)

# Train model
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=[lr_scheduler],
    verbose=1
)

# Evaluate model
val_loss, val_accuracy = model.evaluate(val_generator)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# Save model
model.save("pokemon_model_151.h5")
