import os
import pandas as pd

def update_image_urls(data_folder, images_folder):
    # Ensure paths exist
    if not os.path.exists(data_folder) or not os.path.exists(images_folder):
        print("Data folder or images folder does not exist.")
        return
    
    # Process each CSV file in the data folder
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            set_name = file.replace("_cards.csv", "")  # Extract set name from file name
            
            # Special case mapping
            if set_name == "sv3pt5":
                set_folder = os.path.join(images_folder, "151")
            else:
                set_folder = os.path.join(images_folder, set_name)
            
            if not os.path.exists(set_folder):
                print(f"Skipping {file}, matching set folder {set_folder} not found.")
                continue
            
            file_path = os.path.join(data_folder, file)
            df = pd.read_csv(file_path)
            
            if "image_url" in df.columns:
                updated_paths = []
                
                for index, row in df.iterrows():
                    card_name = row.get("name", "").strip()
                    if not card_name:
                        updated_paths.append("")
                        continue
                    
                    found = False
                    for image_file in os.listdir(set_folder):
                        if image_file.lower().startswith(card_name.lower()):
                            updated_paths.append(os.path.join(set_folder, image_file))
                            found = True
                            break
                    
                    if not found:
                        updated_paths.append("")
                
                preview_df = df.copy()
                preview_df["image_url"] = updated_paths
                print(preview_df.head(10))
                
                confirm = input(f"Do you want to replace the image_url column in {file} with these values? (yes/no): ")
                if confirm.lower() in ["yes", "y"]:
                    df["image_url"] = updated_paths
                    df.to_csv(file_path, index=False)
                    print(f"Updated: {file}")
                else:
                    print(f"Skipping update for {file}.")
            else:
                print(f"Skipping {file}, no 'image_url' column found.")

# Define folders
data_folder = "151_data"
images_folder = "151_images"

update_image_urls(data_folder, images_folder)