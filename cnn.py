import pickle
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, LinearLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(device)
batch_size = 32

def load_data(load_new_data=False):
    with (open("sd_train2.pkl", "rb")) as data: # data is not balanced, data2 balanced, sd for standard deviation
        data = pickle.load(data)
        img = []
        complexity = []
        # separate images from labels

        for h in data:
            img.append(torch.unsqueeze(torch.Tensor(h[0]), 0))
            complexity_label = int(h[1])  # Convert the complexity label to an integer
            complexity.append(complexity_label)
        print(len(img), " ", len(complexity))
        print(len(data))

    if load_new_data:
        # Split the data into training/test features/targets
        X_train, X_test, y_train, y_test = train_test_split(img, complexity, stratify=complexity, test_size=0.1)
        new_data = list(zip(X_train, y_train))
        images = []
        labels = []
        for h in new_data:
            images.append(h[0])
            labels.append(h[1])

        test_dataset = list(zip(X_test, y_test))
        with open('sd_test2.pkl', 'wb') as f:
            pickle.dump(test_dataset, f)
        train_dataset = list(zip(X_train, y_train))
        with open('sd_train2.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)
        return new_data, images, labels
    return data, img, complexity

class Conv3DNet(nn.Module):
    def __init__(self):
        super(Conv3DNet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.batchnorm1 = nn.BatchNorm3d(64)

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.batchnorm2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        self.batchnorm3 = nn.BatchNorm3d(128)

        self.conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=2)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool3d(kernel_size=2)
        self.batchnorm4 = nn.BatchNorm3d(256)

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(256, 128)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 3)
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


epochs = 150
initial_learning_rate = 0.001

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, fold=0):
    accuracies = []
    losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        print(len(train_loader.dataset))
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        losses.append(train_loss)

        # Validation
        model.eval()
        avg_accuracy = 0.0
        actual = []
        model_prediction = []
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs.float())
                predicted = torch.argmax(outputs,dim=1).int()
                avg_accuracy += (predicted == targets).sum().item()
                actual.extend(list(targets.cpu()))
                model_prediction.extend(list(predicted.cpu()))

            avg_accuracy /= len(valid_loader.dataset)
            accuracies.append(avg_accuracy)
            #conf_mat(actual, model_prediction)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}")

        scheduler.step()
    torch.save(model.state_dict(), "models/sd/3d_image_classification"+str(fold)+".pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss},
        "models/sd/model"+str(fold)+".pth.tar")
    return accuracies, losses, actual, model_prediction

def conf_mat(actual, predicted, iteration=5, path= "Results/sd/confmat"):
    labels = [0, 1, 2]
    cm = confusion_matrix(predicted, actual, labels=labels, normalize='all')
    print(cm)
    sns.heatmap(cm,
                annot=True, cmap='Blues', fmt='.2%')
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.savefig(path+ str(iteration)+ ".png")
    plt.show()
    plt.close()
    return cm

def k_fold(k_folds=5, batch_size=batch_size):
    data, images, labels = load_data()
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    accuracies = []
    losses = []
    f1_scores = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(images, labels)):
        model = Conv3DNet()
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
        lr_schedule = ExponentialLR(optimizer, gamma=0.98)

        print(len(train_idx), len(test_idx))
        print(f"Fold {fold + 1}")
        print("-------")
        train_dataset = torch.utils.data.Subset(data, train_idx)
        test_dataset = torch.utils.data.Subset(data, test_idx)
        # Define the data loaders for the current fold
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,shuffle=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
        )
        # Train the model on the current fold
        accuracy, loss, actual, model_prediction = train_model(model, train_loader, test_loader, criterion, optimizer, lr_schedule, epochs, fold+1)
        accuracies.append(accuracy)
        losses.append(loss)
        conf_mat(actual, model_prediction, fold)
        f1 = f1_score(actual, model_prediction, average="macro")
        f1_scores.append(f1)
        print(f"F1 Score Fold {fold + 1}: {f1}")
        y = [i for i in range(epochs)]
        plt.figure(1)
        plt.title("Training and Validation Loss")
        plt.plot(y, accuracy, label="accuracy")
        plt.plot(y, loss, label="loss")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.ylim(0, 1.2)
        plt.savefig("Results/sd/TrainingValidationLoss"+ str(fold))
        plt.show()
    y = [i for i in range(k_folds)]
    plt.figure(2)
    plt.title("F1 Scores")
    plt.plot(y, f1_scores)
    plt.xlabel("folds")
    plt.ylabel("F1 Scores")
    plt.savefig("F1Scores")
    plt.show()
    results = {"Accuracies": accuracies, "TrainingLosses": losses, "F1Scores": f1_scores}
    with open('Results/sd/results.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    k_fold()





