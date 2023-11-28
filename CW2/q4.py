# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:32:33 2023

@author: thao
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import random

# Set a fixed random seed for reproducibility
random_seed = 1
random.seed(random_seed)
torch.manual_seed(random_seed)

# Directory containing subfolders for each class
root_dir = "caltech101_30"

# Create a dataset from the custom folder structure
dataset = datasets.ImageFolder(root=root_dir, transform=None)

total_samples = len(dataset)
split_size = total_samples // 2  # 50% for training, 50% for testing

train_dataset, test_dataset = random_split(dataset, [split_size, total_samples - split_size])

# Define different augmentations for training and testing
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(60),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

# Apply the transformations to the datasets
train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
