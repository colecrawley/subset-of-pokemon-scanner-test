import os
import csv

def update_image_urls_in_csv(csv_file_path, base_folder):
    """Ensure image URLs in the CSV match the file structure of 151_images, replacing apostrophes with underscores."""
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Extract header and rows
    header = lines[0]
    rows = lines[1:]

    # Go through each row and collect changes by set
    set_changes = {}

    # Traverse all the subfolders in the 151_images folder
    set_image_files = {}
    
    for set_name in os.listdir(base_folder):
        set_folder_path = os.path.join(base_folder, set_name)
        if os.path.isdir(set_folder_path):
            set_image_files[set_name] = {}
            for filename in os.listdir(set_folder_path):
                if filename.lower().endswith('.jpg'):
                    # Replace apostrophes with underscores in filenames
                    filename = filename.replace("'", "_")
                    # Extract name and ID from filename (e.g., "name_id.jpg")
                    name_id = filename.replace('.jpg', '')  # Remove the extension
                    set_image_files[set_name][name_id] = os.path.join("151_images", set_name, filename).replace("\\", "/")

    # Go through each row in the CSV
    for i, row in enumerate(rows):
        parts = row.strip().split(',')

        if len(parts) < 5:
            print(f"Skipping malformed row in {csv_file_path}: {row.strip()}")
            continue

        card_id = parts[0].strip()  # Full card ID (e.g., "base1-10")
        name = parts[1].strip().replace("'", "_")  # Replace apostrophes with underscores
        set_name = parts[2].strip()  # Set column
        image_url = parts[4].strip()  # Image URL is in the 5th column (index 4)

        # Construct the correct name_ID format from name and card_id
        name_id = f"{name}_{card_id}"

        # Check if the image exists in the set's subfolder
        if set_name in set_image_files and name_id in set_image_files[set_name]:
            correct_image_url = set_image_files[set_name][name_id]

            # If the current file path doesn't match, add the change to the set_changes dictionary
            if image_url != correct_image_url:
                if set_name not in set_changes:
                    set_changes[set_name] = []
                set_changes[set_name].append((i + 1, row.strip(), f"{parts[0]},{parts[1]},{parts[2]},{parts[3]},{correct_image_url}"))

    # Show changes for each set
    for set_name, changes in set_changes.items():
        print(f"\nChanges for set: {set_name}")
        for change in changes:
            old_url = change[1].split(',')[-1]
            new_url = change[2].split(',')[-1]
            print(f"Row {change[0]}: {old_url} → {new_url}")
        
        confirm = input(f"\nApply changes for set '{set_name}'? (y/n): ").strip().lower()

        if confirm == 'y':
            # Apply changes to the rows
            for change in changes:
                row_idx = change[0] - 1  # Row index is 1 less than the row number
                rows[row_idx] = change[2] + '\n'
            print(f"✔ Changes applied to set '{set_name}'.")
        else:
            print(f"✖ Changes skipped for set '{set_name}'.")

    # Save only if changes were confirmed
    if set_changes:
        with open(csv_file_path, 'w', encoding='utf-8') as f:
            f.write(header)  # Write header
            f.writelines(rows)  # Write updated rows
        print(f"\n✅ Changes successfully saved to {csv_file_path}")
    else:
        print(f"\nNo changes made to {csv_file_path}.")


def update_all_csv_files(csv_folder, base_folder):
    """Update all CSV files in the provided folder."""
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    for csv_file in csv_files:
        csv_file_path = os.path.join(csv_folder, csv_file)
        print(f"\nProcessing file: {csv_file_path}")
        update_image_urls_in_csv(csv_file_path, base_folder)


# CSV Folder (make sure this path is correct)
csv_folder = r'C:\Users\Cole\Desktop\subset clone of pokemon scanner- latest\subset-of-pokemon-scanner-test\151_data'
base_folder = r'C:\Users\Cole\Desktop\subset clone of pokemon scanner- latest\subset-of-pokemon-scanner-test\151_images'

# Run the update on all CSV files in the folder
update_all_csv_files(csv_folder, base_folder)
