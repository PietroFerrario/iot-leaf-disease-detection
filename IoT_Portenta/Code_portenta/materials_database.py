import os 
import random 
import shutil
import functions as fn

########################################

    
def copy_leaves(leaf_folder, dest_folder):
    
    os.makedirs(dest_folder, exist_ok=True)
    

    for dirpath, _, filenames in os.walk(leaf_folder):
        for f in filenames:
            src = os.path.join(dirpath, f)
            dst = os.path.join(dest_folder, f)
            shutil.copy(src, dst)
                
def copy_materials(materials_folder, target_count, dest_folder):
    
    os.makedirs(dest_folder, exist_ok=True)
    
    # Get list of subfolders 
    subfolders = [
        os.path.join(materials_folder, name)
        for name in os.listdir(materials_folder)
        if (os.path.isdir(os.path.join(materials_folder, name)) and name != "foliage")
    ]
    
    num_subfolder = len(subfolders)
    
    num_of_pic_from_each = target_count//num_subfolder
    
    total_selected = 0
    
    for sub in subfolders:
        files = [
            f
            for f in os.listdir(sub)
            if os.path.isfile(os.path.join(sub, f))
            ]
        
        sample_size = min(num_of_pic_from_each, len(files))
        selected_files = random.sample(files, sample_size)
        
        for fname in selected_files:
            shutil.copy(
                os.path.join(sub, fname),
                os.path.join(dest_folder, fname)
            )
        total_selected += len(selected_files)
    
    return total_selected
    
    
 
# Creating leaf-nonleaf dataset 
def create_leaf_binary_dataset(leaf_folder, materials_folder, dataset_folder):
    
    # Target folders
    leaf_output_folder = os.path.join(dataset_folder, "Leaf")
    non_leaf_output_folder = os.path.join(dataset_folder, "Non_Leaf")
    
    # Copying leaves from leaves database
    copy_leaves(leaf_folder, leaf_output_folder)
    num_leaf_files = fn.count_images_in_folder(leaf_output_folder)
    print(f"Copied {num_leaf_files} files to {leaf_output_folder}")
    
    # copying materials from mat. database
    total_selected_materials_files = copy_materials(materials_folder, num_leaf_files, non_leaf_output_folder )
    print(f"Copied {total_selected_materials_files} files to {non_leaf_output_folder}")
    
    
leaf_folder = r"..\Original Dataset"
material_folder = r"..\materials_dataset_original\minc-2500\images"
datased_folder = r"..\leaf_materials_dataset"

create_leaf_binary_dataset(leaf_folder,material_folder, datased_folder)
    
    
    
