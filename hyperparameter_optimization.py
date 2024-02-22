import random

import torch
import torch.nn as nn
import torch.optim as optim
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import pickle
from sklearn.metrics import f1_score

from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def load_data():
    with (open("train_mean_lh.pkl", "rb")) as data:
        data = pickle.load(data)
        img = []
        random.shuffle(data)
        complexity = []
        # separate images from labels
        for h in data:
            img.append(torch.Tensor(h[0]))
            complexity_label = int(h[1])  # Convert the complexity label to an integer
            complexity.append(complexity_label)
        data = list(zip(img, complexity))

        # Split the data into training/test features/targets
        split = int(len(data) * 0.8)
        train = data[:split]
        test = data[split:]
        return train, test


class Conv3DNet(nn.Module):
    def __init__(self, n_conv_layers, n_filters, pooling, activation, n_fc_layers, n_units_fc, dropout_rate_fc, dropout, kernel_size, stride, pool_size=1):
        super(Conv3DNet, self).__init__()

        self.n_conv_layers = n_conv_layers

        # Define convolutional layers dynamically based on n_conv_layers
        self.conv_layers = nn.ModuleList()
        in_channels = 1  # Input channels
        out_channels = n_filters
        print(n_conv_layers, n_filters, pooling, activation, n_fc_layers, n_units_fc, dropout_rate_fc, dropout, kernel_size, pool_size, stride)
        for _ in range(int(n_conv_layers)):
            self.conv_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=stride))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.BatchNorm3d(out_channels))
            if pooling == 'max':
                self.conv_layers.append(nn.MaxPool3d(kernel_size=pool_size))
            elif pooling == 'avg':
                self.conv_layers.append(nn.AvgPool3d(kernel_size=pool_size))

            in_channels = out_channels
            out_channels = out_channels

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        self.fc_layers = nn.ModuleList()
        in_features = n_filters
        out_features = n_units_fc
        for _ in range(n_fc_layers):
            self.fc_layers.append(nn.Linear(in_features, out_features))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_rate_fc))
            in_features = n_units_fc
            out_features = out_features
        #self.fc_layers.append(nn.Dropout(dropout_rate_fc))
        #self.dropout = nn.Dropout(dropout_rate_fc)
        self.fc_out = nn.Linear(in_features, 3)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        for layer in self.fc_layers:
            x = layer(x)

        #x = self.dropout(x)
        x = self.fc_out(x)
        x = self.softmax(x)
        return x

def train_and_evaluate_model(params):
    torch.cuda.empty_cache()
    #epochs = 100
    learning_rate, dropout_rate, n_conv_layers, n_filters, \
        n_fc_layers, n_units_fc, dropout_rate_fc, epochs = params
    #print(f"lr: {learning_rate}, dropout: {dropout_rate}, conv_layers: {n_conv_layers}, filters: {n_filters}, kernelsize: {kernel_size}, stride: {stride}, pooling: {pooling}, \
        #activation: {activation}, fc layers: {n_fc_layers}, fc units: {n_units_fc}, fc dropout: {dropout_rate_fc}, weight decay: {weight_decay}, epochs: {epochs}")
    print(params)
    kernel_size = 2

    criterion = nn.CrossEntropyLoss()

   # if pooling == 0:
    pooling = 'max'
    #elif pooling == 1:
        #pooling = 'avg'

    #if activation == 0:
    activation = 'relu'
    #elif activation == 1:
        #activation = 'leaky_relu'
    """
    if optimizer == 0:
        optimizer = 'adam'
    elif optimizer == 1:
        optimizer = 'sgd'
    """

    model = Conv3DNet(int(n_conv_layers), int(n_filters), pooling, activation, int(n_fc_layers), int(n_units_fc), dropout_rate_fc, dropout_rate, int(kernel_size), stride=1)  # Create a new model instance
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train, test = load_data()
    batch_size = 32
    print("batch size: ", batch_size)
    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(test, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    accuracies = []
    losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, targets)
            predicted = torch.argmax(outputs, dim=1).float()
            train_accuracy += (predicted == targets).sum().item()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader.dataset)
        losses.append(train_loss)
        valid_accuracy = 0.0
        actual = []
        model_prediction = []
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs.float())
                loss += criterion(outputs, targets).item()
                predicted = torch.argmax(outputs, dim=1).float()
                valid_accuracy += (predicted == targets).sum().item()
                actual.extend(list(targets.cpu()))
                model_prediction.extend(list(predicted.cpu()))

            valid_accuracy /= len(val_loader.dataset)
            accuracies.append(valid_accuracy)
            loss /= len(val_loader)
            losses.append(loss)
            f1 = f1_score(actual, model_prediction, average="macro")
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy}, Validation Accuracy: {valid_accuracy:.4f}, F1 Score: {f1}")


    return -valid_accuracy

param_space = [
    Real(1e-5, 1e-3, name='learning_rate'),
    Real(0.2, 0.4, name='dropout_rate'),
    Integer(2, 5, name='n_conv_layers'),
    Integer(8, 128, name='n_filters'),#31, 512
    #Integer(2, 3, name='kernel_size'),
    #Integer(1, 3, name='stride'),
    #Categorical(['max', 'avg'], name='pooling'),
    #Categorical(['relu', 'leaky_relu'], name='activation'),
    Integer(1, 3, name='n_fc_layers'),
    Integer(8, 256, name='n_units_fc'),
    Real(0.2, 0.4, name='dropout_rate_fc'),
    #Categorical(['adam', 'sgd'], name='optimizer'),
    #Real(1e-6, 1e-3, name='weight_decay'),
    Integer(25, 50, name='epochs'),#50, 150
]

def results_to_dict(results, param_names):
    return {param_names[i]: results.x[i] for i in range(len(param_names))}

param_names = [param.name for param in param_space]
def main():
    results = gp_minimize(
        func=train_and_evaluate_model,
        dimensions=param_space,
        n_calls=10,
    )

    best_hyperparameters = results_to_dict(results, param_names)

    # Print the best hyperparameters found
    print("Best hyperparameters found:")
    print(best_hyperparameters)
if __name__ == "__main__":
        main()

