import os 
import matplotlib.pyplot as plt
import functions as fn
import random
from PIL import Image


def plot_image_counts(folders, labels, plot_title):
    image_counts = [fn.count_images_in_folder(folder) for folder in folders]
    
    # plotting 
    plt.figure(figsize=(6,6))
    bar_width = 0.2
    plt.bar(labels, image_counts, color =['green'])
    
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title(plot_title, fontsize=14)
    
    for i, count in enumerate(image_counts):
        plt.text(i, count+1, str(count), ha='center', fontsize = 10)
    plt.tight_layout()
    plt.show()
    
def display_sample_images(folders, labels, plot_title, num_images):
    
    images_with_labels = []
    
    # Collect images
    for folder, label, in zip(folders, labels):
        print(f"Plotting images from folder {folder}")
        all_files = os.listdir(folder)
        all_images = [ file for file in all_files
            if os.path.isfile(os.path.join(folder, file)) and file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('.')]
        select_images = random.sample(all_images, min(len(all_images), num_images))
        print(f"Selected all images {len(select_images)}")
        images_with_labels.extend([(os.path.join(folder,img), label) for img in select_images])
        #images_with_labels.append([(os.path.join(folder,img), label) for img in select_images])
        
        # random.shuffle(images_with_labels)
        
        # plot
    fig, axes = plt.subplots(nrows=len(images_with_labels)//4, ncols=len(images_with_labels)//2, figsize=(10,10), subplot_kw={'xticks': [], 'yticks': []})
    for ax, (img_path, label) in zip(axes.flat, images_with_labels):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
                
    fig.suptitle(plot_title, fontsize=16)
    fig.tight_layout()
    fig.show()
    fig.savefig("leaves.jpg")
    
healthy_folder_1 = r"..\Original Data\Healthy"
sick_subfolders_1 = [r"..\Original Data\Sick\Black Rot" , r"..\Original Data\Sick\ESCA", r"..\Original Data\Sick\Leaf Blight"]

folders_to_plot_1 = [healthy_folder_1] + sick_subfolders_1
labels_1 = ["Healthy", "Black Rot", "ESCA", "Leaf Blight"]

plot_image_counts(folders_to_plot_1, labels_1, "Initial Dataset")
        
healthy_folder_2 = r"..\Dataset\Healthy"
sick_folder_2 = r"..\Dataset\Sick"
labels_2 = ["Healthy", "Sick"]
folder_to_plot_2 = [healthy_folder_2] + [sick_folder_2]


plot_image_counts(folder_to_plot_2, labels_2, "Even Dataset")


display_sample_images(folder_to_plot_2, labels_2, "Sample Images from Dataset", 4)