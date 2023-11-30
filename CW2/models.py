# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:07:31 2023

@author: thao
"""

# Models
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

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, 
                               stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


# ResNet architecture
class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, layers=3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, layers=4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers=6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, layers, stride):
        strides = [stride] + [1] * (layers - 1)
        layers_list = []
        for stride in strides:
            layers_list.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
