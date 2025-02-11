import os
import requests
import csv

# Define the directory paths for images and CSV file
images_dir = '151_images'
data_dir = '151_data'

# Create directories if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Define your API Key
API_KEY = 'd1c5f9f4-52d4-43ed-b509-af7815150924'
headers = {'X-Api-Key': API_KEY}

# Function to get all cards in the set using set ID
def get_cards_in_set(set_code):
    url = f'https://api.pokemontcg.io/v2/cards'
    params = {
        'q': f'set.id:{set_code}',
        'pageSize': 250  # Adjust if necessary to get all cards
    }
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        return response.json()['data']
    else:
        print(f"Error fetching cards for set {set_code}: {response.status_code}")
        return []

# Function to download card image
def download_card_image(image_url, image_path):
    try:
        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(img_response.content)
            print(f"Downloaded image: {image_path}")
        else:
            print(f"Error downloading image: {image_url}")
    except Exception as e:
        print(f"Error downloading image {image_url}: {e}")

# Function to save card data to CSV
# Function to save card data to CSV
def save_card_data_to_csv(cards, file_path):
    fieldnames = ['id', 'name', 'set', 'rarity', 'image_url']
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:  # specify encoding='utf-8'
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for card in cards:
            writer.writerow({
                'id': card['id'],
                'name': card['name'],
                'set': card['set']['name'],
                'rarity': card.get('rarity', 'Unknown'),  # Default to 'Unknown' if 'rarity' is not found
                'image_url': card['images']['small']
            })
    print(f"Card data saved to CSV: {file_path}")


# Function to fetch all sets and retrieve their IDs
def get_set_codes():
    url = 'https://api.pokemontcg.io/v2/sets'
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        sets = response.json()['data']
        return [set_info['id'] for set_info in sets if set_info['id'] != 'sv3pt5']  # Exclude '151' set
    else:
        print(f"Error fetching sets: {response.status_code}")
        return []

# Main function to fetch and save all data and images for multiple sets
def fetch_and_save_data():
    # Get set codes for 4 sets, excluding '151'
    set_codes = get_set_codes()

    # Select the first 4 available sets
    selected_sets = set_codes[:4]  # You can adjust how you select the sets

    for set_code in selected_sets:
        print(f"Fetching cards for set: {set_code}...")
        
        # Create a subfolder for each set inside '151_images' and '151_data'
        set_images_dir = os.path.join(images_dir, set_code)
        set_data_dir = os.path.join(data_dir, f'{set_code}_cards.csv')

        os.makedirs(set_images_dir, exist_ok=True)

        # Get the cards for the current set
        cards = get_cards_in_set(set_code)
        
        if cards:
            # Save the card data to a CSV file for the current set
            save_card_data_to_csv(cards, set_data_dir)
            
            # Download the images for each card in the set
            for card in cards:
                image_url = card['images']['small']
                image_name = f"{card['name']}_{card['id']}.jpg"
                image_path = os.path.join(set_images_dir, image_name)
                download_card_image(image_url, image_path)

if __name__ == '__main__':
    fetch_and_save_data()
