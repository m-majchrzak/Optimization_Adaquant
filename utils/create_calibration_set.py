import os
import csv
import shutil

# Set the path of the original calibration folder
original_folder = "D:/aga/MO/Optimization_Adaquant/data/calibration"

# Set the path of the new calibration folder
new_folder = "D:/aga/MO/Optimization_Adaquant/data/calibration_Wladek"

# Create the new folder if it doesn't exist
os.makedirs(new_folder, exist_ok=True)

# Create a CSV file for storing the id and subfolder information
csv_file = "D:/aga/MO/Optimization_Adaquant/data/calibration_labels.csv"

# Open the CSV file for writing
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "Subfolder"])

    # Iterate over the subfolders in the original folder
    for subfolder_name in os.listdir(original_folder):
        subfolder_path = os.path.join(original_folder, subfolder_name)

        # Check if the subfolder is a directory
        if os.path.isdir(subfolder_path):
            # Iterate over the PNG files in the subfolder
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".png"):
                    file_id = filename.split("_")[1].split(".")[0]
                    new_filename = f"{file_id}.png"
                    original_file_path = os.path.join(subfolder_path, filename)
                    new_file_path = os.path.join(new_folder, new_filename)

                    # Copy the file to the new folder
                    shutil.copy2(original_file_path, new_file_path)

                    # Write the ID and subfolder information to the CSV file
                    writer.writerow([file_id, subfolder_name])

print("Files copied successfully!")