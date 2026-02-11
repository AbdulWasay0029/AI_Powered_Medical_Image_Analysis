
import os
import cv2
import matplotlib.pyplot as plt
import sys

# Dynamic path finding
base_dir = None
search_paths = [
    "chest_xray/train/PNEUMONIA",
    os.path.join("chest_xray", "chest_xray", "train", "PNEUMONIA"),
    os.path.join("Medical_Image_Analysis", "chest_xray", "train", "PNEUMONIA")
]

for path in search_paths:
    if os.path.exists(path):
        base_dir = path
        break

if not base_dir:
    # Try one level up if current dir is Medical_Image_Analysis
    if os.path.basename(os.getcwd()) == "Medical_Image_Analysis":
         if os.path.exists("chest_xray/train/PNEUMONIA"):
             base_dir = "chest_xray/train/PNEUMONIA"

if not base_dir:
    print(f"Error: Directory not found. Searched: {search_paths}")
    print(f"Current Directory: {os.getcwd()}")
    print(f"Listing: {os.listdir('.')}")
    exit(1)

print(f"Using dataset path: {base_dir}")

files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
if not files:
    print("No images found.")
    exit(1)

image_path = os.path.join(base_dir, files[0])
print(f"Loading: {image_path}")

try:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Failed to load image.")
    else:
        # Show image
        plt.imshow(image, cmap='gray')
        plt.title(f"Sample: {files[0]}")
        output_path = "preview_sample.png"
        plt.savefig(output_path)
        print(f"Success! Image saved to {output_path}")
except Exception as e:
    print(f"An error occurred: {e}")
