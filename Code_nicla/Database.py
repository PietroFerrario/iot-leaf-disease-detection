import os 
import random 
import shutil
import functions as fn

# Copy files 
def copy_files(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for file in os.listdir(src_folder):
        src_file = os.path.join(src_folder, file)
        if os.path.isfile(src_file):
            shutil.copy(src_file, os.path.join(dest_folder,file))

# Randomly select 1/3 for the Sick label dataset
def select_files_from_sick_subfolder(target_count, sick_folders, output_sick_folder):
    total_selected = 0
    for folder in sick_folders: 
        files = [file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder,file))]
        sample_size = target_count//3 
        
        selected_files = random.sample(files, sample_size)
        total_selected += len(selected_files)
        
        # Create the Sick output folder if it doesn't exist
        if not os.path.exists(output_sick_folder):
            os.makedirs(output_sick_folder)
        
        # Copy selected files to the output Sick folder
        for file in selected_files:
            shutil.copy(os.path.join(folder, file), os.path.join(output_sick_folder, file))
            
    return total_selected

        
# Creating main dataset 
def create_dataset(healthy_folder, sick_subfolders, dataset_folder):
    
    # Target folders 
    healthy_output_folder = os.path.join(dataset_folder, "Healthy")
    sick_output_folder = os.path.join(dataset_folder, "Sick")
    
    # copying healty 
    copy_files(healthy_folder, healthy_output_folder)
    num_healty_files = fn.count_images_in_folder(healthy_output_folder)
    print(f"Copied {num_healty_files} files to {healthy_output_folder}")
    
    # copying sick 
    total_selected_sick_files = select_files_from_sick_subfolder(num_healty_files, sick_subfolders, sick_output_folder)
    print(f"Copied {total_selected_sick_files} files to {sick_output_folder}")
    
healthy_folder = r"Original Data\Healthy"
sick_subfolders = [r"Original Data\Sick\Black Rot" , r"Original Data\Sick\ESCA", r"Original Data\Sick\Leaf Blight"]
dataset_folder = r"Dataset"

create_dataset(healthy_folder, sick_subfolders, dataset_folder)
	