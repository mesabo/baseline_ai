#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/01/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""
import os

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True)

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(8192, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        x = self.fc4(x)
        return x

if __name__ == "__main__":
    model = CIFAR10Model()
    if os.path.exists("cifar10model.pth"):
        model.load_state_dict(torch.load("cifar10model.pth"))

    plt.imshow(trainset.data[7])
    plt.show()

    X = torch.tensor([trainset.data[7]], dtype=torch.float32).permute(0,3,1,2)
    model.eval()
    with torch.no_grad():
        feature_maps = model.conv1(X)
    fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
    for i in range(0, 32):
        row, col = i//8, i%8
        ax[row][col].imshow(feature_maps[0][i])
    plt.show()

    with torch.no_grad():
        feature_maps = model.act1(model.conv1(X))
        feature_maps = model.drop1(feature_maps)
        feature_maps = model.conv2(feature_maps)
    fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
    for i in range(0, 32):
        row, col = i//8, i%8
        ax[row][col].imshow(feature_maps[0][i])
    plt.show()