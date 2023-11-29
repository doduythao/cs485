# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:07:31 2023

@author: thao
"""

# Models
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, batch_norm=True, dropout=True):
        super(AlexNet, self).__init__()
        layers = [nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
                  nn.ReLU(inplace=True),
                  nn.BatchNorm2d(96) if batch_norm else None,
                  nn.MaxPool2d(kernel_size=3, stride=2),

                  nn.Conv2d(96, 256, kernel_size=5, padding=2),
                  nn.ReLU(inplace=True),
                  nn.BatchNorm2d(256) if batch_norm else None,
                  nn.MaxPool2d(kernel_size=3, stride=2),

                  nn.Conv2d(256, 384, kernel_size=3, padding=1),
                  nn.ReLU(inplace=True),

                  nn.Conv2d(384, 384, kernel_size=3, padding=1),
                  nn.ReLU(inplace=True),

                  nn.Conv2d(384, 256, kernel_size=3, padding=1),
                  nn.ReLU(inplace=True),
                  nn.BatchNorm2d(256) if batch_norm else None,
                  nn.MaxPool2d(kernel_size=3, stride=2),]
        layers = [layer for layer in layers if layer is not None]
        self.features = nn.Sequential(*layers)

        layers_2 = [nn.Linear(256 * 6 * 6, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout() if dropout else None,

                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout() if dropout else None,

                    nn.Linear(4096, num_classes),]
        layers_2 = [layer for layer in layers_2 if layer is not None]
        self.classifier = nn.Sequential(*layers_2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)  # Adjust based on the input size
        x = self.classifier(x)
        return x



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

# ResNet-18 model


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
