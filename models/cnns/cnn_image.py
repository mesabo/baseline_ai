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
import torch.optim as optim
import matplotlib.pyplot as plt

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 32
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        # (Red, Green, Blue) = (255, 255, 255)

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
        c1 = self.conv1(x) # conv1 result
        a1 = self.act1(c1) # relu result
        d1 = self.drop1(a1) # dropout result

        # input 32x32x32, output 32x16x16
        c2 = self.conv2(d1)
        a2 = self.act2(c2)
        p2 = self.pool2(a2)


        # input 32x16x16, output 8192
        f1 = self.flat(p2)

        # input 8192, output 512
        fc3 = self.fc3(f1)
        a3 = self.act3(fc3)
        d3 = self.drop3(a3)

        # input 512, output 10
        x = self.fc4(d3)
        return x

def RunCNNImageModel1():

    plt.imshow(trainset.data[7])
    plt.show()
    X = torch.tensor([trainset.data[7]], dtype=torch.float32).permute(0,3,1,2)


    model = CIFAR10Model()
    if os.path.exists("cifar10_model.pth"):
        model.load_state_dict(torch.load("cifar10_model.pth"))

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

def RunCNNImageModel2():
    model = CIFAR10Model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    n_epochs = 2
    for epoch in range(n_epochs):
        for inputs, labels in trainloader:
            # forward, backward, and then weight update
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = 0
        count = 0
        for inputs, labels in testloader:
            y_pred = model(inputs)
            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
        acc /= count
        print("Epoch %d: model accuracy %.2f%%" % (epoch, acc * 100))

    torch.save(model.state_dict(), "cifar10model.pth")