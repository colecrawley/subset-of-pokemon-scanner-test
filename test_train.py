import os 
import random
import shutil

splitsize = 0.85

categories = []

source_folder = r'C:\Users\lf220\Desktop\mobilenet project\151_images'
folders = os.listdir(source_folder)
print(folders)

for subfolder in folders:
    if os.path.isdir(source_folder + "/" + subfolder):
        categories.append(subfolder)

categories.sort()
print(categories)

#create target folder
target_folder = r"C:\Users\lf220\Desktop\mobilenet project\dataset_for_model"
existDataSetPath = os.path.exists(target_folder)
if existDataSetPath == False:
    os.mkdir(target_folder)

#create functon to split the data for train and validation

def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    files = []

    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        print(file)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + ' is 0 length, ignore it .....')
    print(len(files))

    trainingLength = int(len(files) * SPLIT_SIZE)
    shuffleSet = random.sample(files , len(files))
    trainingSet = shuffleSet[0:trainingLength]
    validSet = shuffleSet[trainingLength:]

    # Copy the train images

    for filename in trainingSet:
        thisFile = SOURCE + filename
        destination = TRAINING + filename
        shutil.copyfile(thisFile, destination)

    # Copy the val images
    for filename in validSet:
        thisFile = SOURCE + filename
        destination = VALIDATION + filename
        shutil.copyfile(thisFile, destination)


trainPath = target_folder + "/train"
validatePath = target_folder + "/validate"

#create the target folders
existDataSetPath = os.path.exists(trainPath)
if existDataSetPath == False:
    os.mkdir(trainPath)

existDataSetPath = os.path.exists(validatePath)
if existDataSetPath == False:
    os.mkdir(validatePath)

# Let's run the function
for category in categories:
    trainDestPath = trainPath + "/" + category
    validateDestPath = validatePath + "/" + category

    if os.path.exists(trainDestPath) == False:
        os.mkdir(trainDestPath)
    if os.path.exists(validateDestPath) == False:
        os.mkdir(validateDestPath)

    sourcePath = source_folder + "/" + category + "/"
    trainDestPath = trainDestPath + "/" 
    validateDestPath = validateDestPath + "/"

    print("Copy from : " + sourcePath + " to : " + trainDestPath + " and " + validateDestPath)

    split_data(sourcePath, trainDestPath, validateDestPath, splitsize)





################
#train and val id

import os
import random
import shutil
import pandas as pd

splitsize = 0.85

# Set the folder where the card image subfolders are stored
dataset_folder = r'C:\Users\lf220\Desktop\mobilenet project\151_data'  # Folder containing the subfolders (sets)
source_folder = r'C:\Users\lf220\Desktop\mobilenet project\dataset\151_images'  # Source folder containing card images
target_folder = r"C:\Users\lf220\Desktop\mobilenet project\dataset_for_model"  # Target folder for train and val directories

# Create the target folders if they don't exist
trainPath = os.path.join(target_folder, "train_id")
validatePath = os.path.join(target_folder, "val_id")
if not os.path.exists(trainPath):
    os.mkdir(trainPath)

if not os.path.exists(validatePath):
    os.mkdir(validatePath)

# Function to split images into train and validation sets
def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    files = []

    # Get all image files for this card_id
    for filename in os.listdir(SOURCE):
        file = os.path.join(SOURCE, filename)
        print(f"Checking file: {file}")
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(f"{filename} is 0 length, ignoring...")

    print(f"Total files for {SOURCE}: {len(files)}")

    if len(files) == 0:
        print(f"Warning: No valid files found in {SOURCE}. Skipping...")
        return

    # Split into training and validation sets
    trainingLength = int(len(files) * SPLIT_SIZE)
    shuffleSet = random.sample(files, len(files))
    trainingSet = shuffleSet[0:trainingLength]
    validSet = shuffleSet[trainingLength:]

    # Copy the train images
    for filename in trainingSet:
        thisFile = os.path.join(SOURCE, filename)
        destination = os.path.join(TRAINING, filename)
        print(f"Copying to train: {destination}")
        shutil.copyfile(thisFile, destination)

    # Copy the val images
    for filename in validSet:
        thisFile = os.path.join(SOURCE, filename)
        destination = os.path.join(VALIDATION, filename)
        print(f"Copying to val: {destination}")
        shutil.copyfile(thisFile, destination)

# Iterate through each subfolder (which corresponds to a different set)
for subfolder in os.listdir(dataset_folder):
    subfolder_path = os.path.join(dataset_folder, subfolder)

    if os.path.isdir(subfolder_path):  # Ensure it is a directory
        print(f"\nProcessing set: {subfolder}")

        # Look for CSV files in this subfolder
        csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]

        # Loop through each CSV file in the subfolder
        for csv_file in csv_files:
            csv_path = os.path.join(subfolder_path, csv_file)
            print(f"Processing CSV: {csv_file}")

            # Load the CSV file
            df = pd.read_csv(csv_path)

            # Assuming the card ID is in a column named 'card_id'. Adjust this if necessary.
            if 'card_id' not in df.columns:
                print(f"Warning: 'card_id' column not found in {csv_file}. Skipping.")
                continue

            card_ids = df['card_id'].tolist()  # List of card IDs in this CSV

            # Loop through each card_id in the CSV and process images
            for card_id in card_ids:
                # Create directories for each card_id in train and val sets
                trainDestPath = os.path.join(trainPath, card_id)
                validateDestPath = os.path.join(validatePath, card_id)

                # Ensure the directories for each card_id exist
                if not os.path.exists(trainDestPath):
                    os.makedirs(trainDestPath)

                if not os.path.exists(validateDestPath):
                    os.makedirs(validateDestPath)

                # Define source path where the card images are stored for this card_id
                sourcePath = os.path.join(source_folder, subfolder, card_id)  # Assuming images for each card_id are inside the set folder

                # Check if the source folder exists for this card_id
                if not os.path.exists(sourcePath):
                    print(f"Warning: Source folder for {card_id} does not exist in {subfolder}! Skipping.")
                    continue  # Skip to the next card_id if the folder does not exist

                # Run the split_data function for this card_id
                print(f"Processing card_id: {card_id} (Source: {sourcePath})")
                split_data(sourcePath, trainDestPath, validateDestPath, splitsize)

print("Data split completed.")

