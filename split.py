import os
import shutil
import random

def split_dataset(source_folder, train_folder, val_folder, val_percentage=0.2):
    # Create train and val folders
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Iterate over gesture folders in the source folder
    for gesture_folder in os.listdir(source_folder):
        gesture_path = os.path.join(source_folder, gesture_folder)

        # Skip if not a directory
        if not os.path.isdir(gesture_path):
            continue

        # List all image files in the gesture folder
        image_files = [f for f in os.listdir(gesture_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        # Calculate the number of images to move to the validation set
        num_val_images = int(len(image_files) * val_percentage)

        # Randomly select images for validation
        val_images = random.sample(image_files, num_val_images)

        # Move images to the val folder
        for val_image in val_images:
            src_path = os.path.join(gesture_path, val_image)
            dest_path = os.path.join(val_folder, gesture_folder, val_image)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.move(src_path, dest_path)

        # Move the remaining images to the train folder
        for train_image in image_files:
            if train_image not in val_images:
                src_path = os.path.join(gesture_path, train_image)
                dest_path = os.path.join(train_folder, gesture_folder, train_image)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(src_path, dest_path)

# Specify paths
source_folder = "C:/Users/Student/asl_interpreter/source"
train_folder = "C:/Users/Student/asl_interpreter/dataset/train"
val_folder = "C:/Users/Student/asl_interpreter/dataset/val"

# Call the function to split the dataset
split_dataset(source_folder, train_folder, val_folder, val_percentage=0.2)
