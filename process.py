import os
from pathlib import Path
import numpy as np
import cv2
import struct

# Define image properties (modify based on actual format)
width, height, channels = 640, 480, 3  # Example for an RGB image
BASE_PATH = "data_q123"
IMAGE_PATH = os.path.join(BASE_PATH, "images/")
OUTPUT_PATH = os.path.join(BASE_PATH, "output/")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Read binary file

def read_image(file_path):
    with open(file_path, "rb") as f:
        width, height = struct.unpack("II", f.read(8))
        image_data = np.frombuffer(f.read(), dtype=np.uint8)

        # H * W * 3 => (R, G, B)
        image = image_data.reshape((height, width, 3))
    return image


images = []
image_timestamps = []
image_folder = Path(f'{IMAGE_PATH}')
for img_file in sorted(image_folder.iterdir()):
    if img_file.suffix == ".bin":
        img_data = read_image(img_file)
        output_file = os.path.join(OUTPUT_PATH, f"{img_file.stem}.png")
        print(output_file)
        cv2.imwrite(output_file, img_data)


