# Used to download and load the MNIST dataset
# src/dataloader.py

import os
import urllib.request

def download_mnist(dir='data'): 
    """Download the MNIST dataset (if not already downloaded)"""
    
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
    pass

def load_mnist_labels(path):
    """Parse and load labels into numpy arrays"""
    pass

def load_dataset():
    """Combined function to return (train_images, train_labels, test_images, test_labels)"""
    pass
