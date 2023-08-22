import os
import random
import zipfile
import numpy as np
import pickle
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

with (open("data2.pkl", "rb")) as hf:
    hf = pickle.load(hf)
    img = []
    random.shuffle(hf)
    complexity = []
    # separate images from labels
    for h in hf:
        img.append(torch.unsqueeze(torch.Tensor(h[0]), 0))
        #complexity.append(torch.Tensor(h[1]))
        complexity_label = int(h[1])  # Convert the complexity label to an integer
        complexity.append(complexity_label)
    #img = img.unsqueeze(0)
    print(len(img), " ", len(complexity))
    hf = list(zip(img, complexity))
    print(len(hf))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Split the data into training/test features/targets
    split = int(len(hf) * 0.8)
    train = hf[:split]
    test = hf[split:]
    #x_train = torch.tensor(img[:split])
    #y_train = torch.tensor(complexity[:split])
    #x_test = torch.tensor(img[split:])
    #y_test = torch.tensor(complexity[split:])

    batch_size = 23
    #print(len(train)," ", len(test))
    # Define data loaders.
    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size)
    validation_loader = DataLoader(test, shuffle=True, batch_size=batch_size)

    class Conv3DNet(nn.Module):
        def __init__(self):
            super(Conv3DNet, self).__init__()

            self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool3d(kernel_size=2)
            self.batchnorm1 = nn.BatchNorm3d(64)

            self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool3d(kernel_size=2)
            self.batchnorm2 = nn.BatchNorm3d(64)

            self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            self.relu3 = nn.ReLU()
            self.pool3 = nn.MaxPool3d(kernel_size=2)
            self.batchnorm3 = nn.BatchNorm3d(128)

            self.conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
            self.relu4 = nn.ReLU()
            self.pool4 = nn.MaxPool3d(kernel_size=2)
            self.batchnorm4 = nn.BatchNorm3d(256)

            self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
            self.fc1 = nn.Linear(256, 512)
            self.relu_fc = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(512, 3)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.batchnorm1(x)
            x = self.pool1(x)

            x = self.relu2(self.conv2(x))

            x = self.batchnorm2(x)
            x = self.pool2(x)

            x = self.relu3(self.conv3(x))

            x = self.batchnorm3(x)
            x = self.pool3(x)

            x = self.relu4(self.conv4(x))

            x = self.batchnorm4(x)
            x = self.pool4(x)

            x = self.global_avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.relu_fc(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.softmax(x)

            return x


    # Build model
    model = Conv3DNet()
    print(model)
    model.to(device)

    initial_learning_rate = 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

    lr_schedule = ExponentialLR(optimizer, gamma = 0.96)
    epochs = 100

    def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs):
        best_valid_accuracy = 0.0
        accuracies = []
        losses = []
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                #print(data)
                inputs, targets = inputs.to(device), targets.to(device)
                #print(targets)
                optimizer.zero_grad()
                outputs = model(inputs.float())
                #print(outputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            losses.append(train_loss)

            # Validation
            model.eval()
            valid_accuracy = 0.0

            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs.float())
                    #print(outputs)
                    #print(targets)
                    predicted = torch.argmax(outputs,dim=1).float()
                    valid_accuracy += (predicted == targets).sum().item()

                valid_accuracy /= len(valid_loader.dataset)
                accuracies.append(valid_accuracy)

            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}")

            # Save the model if validation accuracy improves
            #if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            torch.save(model.state_dict(), "3d_image_classification.pth")

            # Adjust learning rate
            scheduler.step()

        #plt.figure(figsize=(10, 5))
        y = [i for i in range(epochs)]
        #print(accuracies)
        #print(y)
        plt.figure()
        plt.title("Training and Validation Loss")
        plt.plot(y,accuracies, label="accuracy")
        plt.plot(y,losses, label="loss")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.ylim(0,1.2)
        plt.show()


    # Assuming you have PyTorch DataLoader objects train_loader and valid_loader
    train_model(model, train_loader, validation_loader, criterion, optimizer, lr_schedule, epochs)



