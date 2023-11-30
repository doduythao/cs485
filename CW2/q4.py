# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:32:33 2023

@author: thao
"""
import random
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from models import AlexNet, ResNet34

# Set a fixed random seed for reproducibility
random_seed = 1
random.seed(random_seed)
torch.manual_seed(random_seed)
im_size = 224
num_classes = 10
# net_type = 'alex'
net_type = 'res'
norm = True
dropout = True
loss_type = 'entropy'
# loss_type = 'hinge'
lr = 0.001
num_ep = 500


root_dir = "caltech101_30"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create a dataset from the custom folder structure
dataset = datasets.ImageFolder(root=root_dir, transform=None)

total_samples = len(dataset)
split_size = total_samples // 2  # 50% for training, 50% for testing

train_dataset, test_dataset = random_split(
    dataset, [split_size, total_samples - split_size])

# Define different augmentations for training and testing
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((im_size, im_size), scale=(0.75, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomPerspective(),
    transforms.RandomPosterize(bits=2),
    transforms.RandomRotation(70),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Apply the transformations to the datasets
train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if net_type == 'alex':
    net = AlexNet(num_classes=num_classes, batch_norm=norm, dropout=dropout)
else:
    net = ResNet34(num_classes=num_classes)

net.to(device)


if loss_type == 'entropy':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.MultiMarginLoss()

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

max_test = 0
for epoch in range(num_ep):  # loop over the dataset multiple times
    for data in train_loader:
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(loss)

    if epoch % 100 == 0:
        scheduler.step()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = (correct / total)
    if max_test < test_acc:
        max_test = test_acc
    print(f'Accuracy on the test images: {100 * test_acc} % | {max_test}')
