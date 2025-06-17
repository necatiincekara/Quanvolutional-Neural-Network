"""
Data loading and preprocessing for the Ottoman handwritten character dataset.
"""

import os
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from pennylane import numpy as np
from src import config

def load_images_from_folder(folder_path, tags, image_size):
    """
    Loads images and their corresponding labels from a specified folder.

    Args:
        folder_path (str): The path to the folder containing images.
        tags (dict): A dictionary mapping label codes to character names.
        image_size (int): The target size (width and height) to resize images to.

    Returns:
        tuple: A tuple containing a numpy array of images and a numpy array of labels.
    """
    images = []
    labels = []
    
    print(f"Loading images from: {folder_path}")
    for filename in os.listdir(folder_path):
        label_code = filename[-6:-4]
        if label_code not in tags.keys():
            print(f"Warning: Skipping file with unknown label code: {filename}")
            continue

        if filename.endswith('.png'):
            try:
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read image file: {filename}")
                    continue
                
                img = cv2.resize(img, (image_size, image_size))
                images.append(img)
                
                label = int(label_code) - 1
                labels.append(label)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print(f"Loaded {len(images)} images.")
    return np.array(images), np.array(labels)

def get_dataloaders():
    """
    Creates and returns the data loaders for training, validation, and testing.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader).
    """
    # Load the train images and labels
    train_images, train_labels_int = load_images_from_folder(
        config.TRAIN_PATH, config.TAGS, config.IMAGE_SIZE
    )

    # Load the test images and labels
    test_images, test_labels_int = load_images_from_folder(
        config.TEST_PATH, config.TAGS, config.IMAGE_SIZE
    )

    # Convert labels to torch tensors
    train_labels_int = torch.tensor(train_labels_int, dtype=torch.long)
    test_labels_int = torch.tensor(test_labels_int, dtype=torch.long)

    # One-hot encode is not needed for CrossEntropyLoss, it expects class indices.
    # If you switch to a loss like MSE, you might need one-hot vectors.
    # train_labels = F.one_hot(train_labels_int, num_classes=config.NUM_CLASSES).float()
    # test_labels = F.one_hot(test_labels_int, num_classes=config.NUM_CLASSES).float()
    train_labels = train_labels_int
    test_labels = test_labels_int
    
    # Normalize pixel values between 0 and 1 and add channel dimension
    train_images = (train_images / 255.0)[:, np.newaxis, :, :]
    test_images = (test_images / 255.0)[:, np.newaxis, :, :]

    # Convert images to torch tensors
    train_images = torch.tensor(train_images, dtype=torch.float32)
    test_images = torch.tensor(test_images, dtype=torch.float32)

    # Create datasets
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    # Split training data into training and validation sets
    train_size = int((1.0 - config.VALIDATION_SPLIT) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader 