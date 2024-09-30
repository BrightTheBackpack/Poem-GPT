import os

# Set the base path to the 'dataset' folder
base_path = 'dataset'

# Initialize a variable to store all poems
all_poems = []

# Walk through all folders and subfolders
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith('.txt'):  # Assuming poems are stored in .txt files
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    poem = f.read()
                    all_poems.append(poem)
                    print(f"Successfully read: {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Check if any poems were found
if all_poems:
    # Join all the poems with a separator (e.g., two new lines)
    combined_poems = "\n\n".join(all_poems)

    # Save to a new text file
    with open('all_poems.txt', 'w', encoding='utf-8') as output_file:
        output_file.write(combined_poems)

    print("All poems saved to all_poems.txt")
else:
    print("No poems found!")
