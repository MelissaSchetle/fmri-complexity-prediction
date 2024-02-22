import os
import pickle
import torch
from sklearn.model_selection import KFold, train_test_split
from torch import optim
import torch.nn as nn
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
print(torch.cuda.is_available())


def load_data():
    with (open('all_train_variable.pkl', "rb")) as data:
        data = pickle.load(data)
    #data = list(zip(X_train, y_train))
    images = []
    labels = []
    #data = data[:500]
    for h in data:
        images.append(h[0])
        labels.append(h[1])

    return data, images, labels


def load_new_data():
    directory = 'subject_pickles_variable'
    data = []
    for filename in sorted(os.listdir(directory)):
        f = os.path.join(directory, filename)
        with (open(f, "rb")) as d:
            d = pickle.load(d)
            data.append(d)

    # separate images from labels
    train_data = []
    test_data = []

    for subject in data:
        complexity = []
        img = []
        for h in subject:
            img.append(torch.Tensor(h[0]))
            complexity_label = h[1]
            complexity.append(complexity_label)
        X_train, X_test, y_train, y_test = train_test_split(img, complexity, stratify=complexity, test_size=0.1)
        t = list(zip(X_train, y_train))
        print(len(t))
        train_data.extend(t)
        test_data.extend(list(zip(X_test, y_test)))

    data = train_data
    images = []
    labels = []
    for h in data:
        images.append(h[0])
        labels.append(h[1])
    print(len(train_data))
    with open('all_test_variable.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    with open('all_train_variable.pkl', 'wb') as f:
        pickle.dump(train_data, f)

    return data, images, labels

class Conv3DNet(nn.Module):
    def __init__(self):
        super(Conv3DNet, self).__init__()

        self.conv1 = self.conv_block(1, 16)
        self.conv2 = self.conv_block(16, 32)
        self.conv3 = self.conv_block(32, 64)

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(64, 256)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(256, 3)
        self.softmax = nn.Softmax(dim=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
            nn.MaxPool3d(kernel_size=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.conv4(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x,1)
        x = self.relu_fc(self.fc1(x))
        x = self.dropout(x)
        #x = self.relu_fc(self.fc2(x))
        #x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x



epochs = 70
initial_learning_rate = 0.0001
batch_size = 8

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, fold=0):
    accuracies = []
    losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        actual = []
        model_prediction = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            predicted = torch.argmax(outputs, dim=1).int()
            train_accuracy += (predicted == targets).sum().item()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            actual.extend(list(targets.cpu()))
            model_prediction.extend(list(predicted.cpu()))

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader.dataset)
        losses.append(train_loss)
        #conf_mat(actual, model_prediction, False, train=" Training")

        # Validation
        model.eval()
        avg_accuracy = 0.0
        actual = []
        model_prediction = []
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs.float())
                predicted = torch.argmax(outputs, dim=1).int()
                avg_accuracy += (predicted == targets).sum().item()
                actual.extend(list(targets.cpu()))
                model_prediction.extend(list(predicted.cpu()))

            avg_accuracy /= len(valid_loader.dataset)
            accuracies.append(avg_accuracy)
            f1 = f1_score(actual, model_prediction, average="macro")
            conf_mat(actual, model_prediction, False, train=" Validation")

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}, Train Accuracy: {train_accuracy:.4f}, F1 Score: {f1:.4f} ")

        scheduler.step()
    torch.save(model.state_dict(), "Results_all/3d_image_classification"+str(fold)+".pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss},
        "Results_all/model"+str(fold)+".pth.tar")
    return accuracies, losses, actual, model_prediction

def conf_mat(actual, predicted, save=True, iteration=5, train=""):
    labels = [0, 1, 2]
    cm = confusion_matrix(predicted, actual, labels=labels, normalize="pred")
    cm2 = confusion_matrix(predicted, actual, labels=labels)
    #print(cm)
    sns.heatmap(cm,
                annot=True, cmap='Blues', fmt='.2%')
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix'+train, fontsize=17)
    if save:
        plt.savefig("Results_all/eval_confmat"+ str(iteration)+ ".png")
    plt.show()


    plt.close()
    return cm2

def k_fold(k_folds=5, batch_size=batch_size):
    #hf, images, labels =  load_new_data()
    hf, images, labels = load_data()
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    accuracies = []
    losses = []
    f1_scores = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(images, labels)):
        train_dataset = torch.utils.data.Subset(hf, train_idx)
        test_dataset = torch.utils.data.Subset(hf, test_idx)
        model = Conv3DNet()
        model.to(device)


        print(len(train_idx), len(test_idx))
        print(f"Fold {fold + 1}")
        print("-------")

        # Define the data loaders for the current fold
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
        lr_schedule = ExponentialLR(optimizer, gamma=0.98)
        accuracy, loss, actual, model_prediction = train_model(model, train_loader, test_loader, criterion, optimizer, lr_schedule, epochs, fold+1)
        accuracies.append(accuracy)
        losses.append(loss)
        conf_mat(actual, model_prediction, True, fold)
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
        plt.savefig("Results_all/TrainingValidationLoss"+ str(fold))
        plt.show()
    y = [i for i in range(k_folds)]
    plt.figure(2)
    plt.title("F1 Scores")
    plt.plot(y, f1_scores)
    plt.xlabel("folds")
    plt.ylabel("F1 Scores")
    plt.savefig("Results_all/F1Scores")
    plt.show()
    results = {"Accuracies": accuracies, "TrainingLosses": losses, "F1Scores": f1_scores}
    with open('Results_all/results.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    k_fold()





