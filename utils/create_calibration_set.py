import os
import csv
import shutil

""" 
    Code to create calibration dataset from the original folder (changing structure of files).
    Used by us to correct the structure.

    Params set below:
        origilan_foler - folder where data is currently stored
        new_folder - folder where data will be stored after changes
    """

#set the paths
original_folder = "original_path"
new_folder = "new_path"

#create the new folder if it doesn't exist
os.makedirs(new_folder, exist_ok=True)

#create a CSV file for storing the id and subfolder information
csv_file = "path_to_csv"

with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "label"])

    #iterate over the subfolders in the original directory
    for subfolder_name in os.listdir(original_folder):
        subfolder_path = os.path.join(original_folder, subfolder_name)

        #check if the subfolder is a directory
        if os.path.isdir(subfolder_path):
            #iterate over the PNG files in the subfolder
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".png"):
                    file_id = filename.split("_")[1].split(".")[0] #the names of images were specific
                    new_filename = f"{file_id}.png"
                    original_file_path = os.path.join(subfolder_path, filename)
                    new_file_path = os.path.join(new_folder, new_filename)

                    #copy the file to the new folder
                    shutil.copy2(original_file_path, new_file_path)

                    #write the id and label to the CSV file
                    writer.writerow([file_id, subfolder_name])

print("Files copied successfully!")