from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import cnn
import torch
import pickle
from torch.utils.data import DataLoader
from scipy.stats import chi2_contingency

batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
with (open("sd_test2.pkl", "rb")) as test:
    test = pickle.load(test)
valid_loader = DataLoader(test, shuffle=True, batch_size=batch_size)
for i in range(1, 6):
    model = cnn.Conv3DNet()
    model.load_state_dict(torch.load("models/sd3/3d_image_classification"+str(i)+".pth"))
    model = model.to(device)

    model.eval()
    valid_accuracy = 0.0
    actual = []
    model_prediction = []
    accuracies = []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float()).to(device)
            predicted = torch.argmax(outputs, dim=1).int()
            valid_accuracy += (predicted == targets).sum().item()
            actual.extend(list(targets.cpu()))
            model_prediction.extend(list(predicted.cpu()))

        valid_accuracy /= len(valid_loader.dataset)
    matrix = cnn.conf_mat(actual, model_prediction, iteration=i, path="evaluation/confmat")

    chi2, p, dof, expected = chi2_contingency(matrix)
    f1 = f1_score(actual, model_prediction, average="macro")
    precision = precision_score(actual, model_prediction, average="macro")
    recall = recall_score(actual, model_prediction, average="macro")
    print(f"Accuracy: {valid_accuracy}")
    print(f"chi:{chi2} p-value:{p}")
    print(f"F1 Score: {f1} Precision: {precision} Recall: {recall}")

