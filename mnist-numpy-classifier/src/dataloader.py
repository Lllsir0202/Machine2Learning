# Used to download and load the MNIST dataset
# src/dataloader.py

import os
import urllib.request
import gzip
import struct
import numpy as np
from utils import *

DEBUG = True

def download_mnist(dir='data'): 
    """Download the MNIST dataset (if not already downloaded)"""
    
    # To avoid unnecessary questions, we use absolute paths
    dir = os.path.join(project_root, dir)
    # Create the directory if it does not exist
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    base_url = "http://yann.lecun.com/exdb/mnist/"
    # Loop through the files to download
    files = {
        "train-images-idx3-ubyte.gz": base_url + "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz": base_url + "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz": base_url + "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz": base_url + "t10k-labels-idx1-ubyte.gz"
    }

    for file_name, url in files.items():
        file_path = os.path.join(dir, file_name)
        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded {file_name} to {file_path}")
    

def load_mnist_images(path):
    """Parse and load images into numpy arrays"""
    with gzip.open(path, "rb") as f:
        # Read header information
        header = f.read(16)
        # In MNIST, the first 16 bytes include following:
        # 0-3 magic number -> What kind of file it is -> 2051(Image)
        # 4-7 number of images
        # 8-11 number of rows
        # 12-15 number of columns
        magic_num, num_images, num_rows, num_cols = struct.unpack('>IIII', header)
        if magic_num != 2051:
            raise ValueError(f"Invalid magic number {magic_num}. Expected 2051 for images.")
        if DEBUG:
            print(f"Magic Number: {magic_num}, Number of Images: {num_images}, Rows: {num_rows}, Columns: {num_cols}")

        # Then we can calculate how many pixels are there in each image
        # Each pixel is represented by a single byte (0-255)
        num_pixels = num_rows * num_cols
        # Read all images from the file
        total_pixels = num_images * num_pixels
        image_data = f.read(total_pixels)

        # Convert the byte data into a numpy array
        images = np.frombuffer(image_data, dtype=np.uint8)
        # Reshape the array to (num_images, num_rows, num_cols)
        images = images.reshape(num_images, num_rows, num_cols)
        # Normalize the pixel values to [0, 1]
        images = images.astype(np.float32) / 255.0

        return images

def load_mnist_labels(path):
    """Parse and load labels into numpy arrays"""
    with gzip.open(path, "rb") as f:
        # First we read the header information
        header = f.read(8)
        # In MNIST, the first 8 bytes include following:
        # 0-3 magic number -> What kind of file it is -> 2049(Label)
        # 4-7 number of labels
        magic_num, num_labels = struct.unpack(">II", header)
        if magic_num != 2049:
            raise ValueError(f"Invalid magic number {magic_num}. Expected 2049 for labels.")
        if DEBUG:
            print(f"Magic Number: {magic_num}, Number of Labels: {num_labels}")
        # Read all labels from the file
        label_data = f.read(num_labels)
        # Convert the byte data into a numpy array
        labels = np.frombuffer(label_data, dtype=np.uint8)

        return labels


def load_dataset():
    """Combined function to return (train_images, train_labels, test_images, test_labels)"""
    pass
