import os
import cv2
import numpy as np
from tqdm import tqdm  # Optional: for progress bar

# Set the target size
target_size = (224, 224)

# Input and output directories
input_directory = r"D:\Lungs Xray\pneumonia\NORMAL"
output_directory = "D:\Lungs Xray\pneumonia\Transformed Normal"

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get a list of image files in the input directory
image_files = [f for f in os.listdir(input_directory) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Loop through each image and resize
for image_file in tqdm(image_files, desc="Resizing images"):
    # Read the image
    image_path = os.path.join(input_directory, image_file)
    img = cv2.imread(image_path)

    # Resize the image
    img_resized = cv2.resize(img, target_size)

    # Save the resized image to the output directory
    output_path = os.path.join(output_directory, image_file)
    cv2.imwrite(output_path, img_resized)
