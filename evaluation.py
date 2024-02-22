from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support

import cnn_mean as cnn
import torch
import pickle
from torch.utils.data import DataLoader
from scipy.stats import chi2_contingency

batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
with (open("test_mean_lh.pkl", "rb")) as test:
    test = pickle.load(test)
valid_loader = DataLoader(test, shuffle=True, batch_size=batch_size)
actual = []
model_prediction = []
valid_accuracy = 0.0
#accuracies = []
for i in range(1, 6):
    model = cnn.Conv3DNet()
    model.load_state_dict(torch.load("newResults/mean/3d_image_classification"+str(i)+".pth"))
    model = model.to(device)
    #print(model)

    model.eval()


    with torch.no_grad():
        for inputs, targets in valid_loader:
            #inputs = torch.unsqueeze(inputs, 1)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float()).to(device)
            predicted = torch.argmax(outputs, dim=1).int()
            valid_accuracy += (predicted == targets).sum().item()
            actual.extend(list(targets.cpu()))
            model_prediction.extend(list(predicted.cpu()))

precision, recall, f1, _ = precision_recall_fscore_support(actual, model_prediction, average=None)

# Print class-wise F1 scores
for i in range(len(f1)):
    print(f"Class {i}: Precision={precision[i]}, Recall={recall[i]}, F1 Score={f1[i]}")

valid_accuracy /= (len(valid_loader.dataset)*5)
matrix = cnn.conf_mat(actual, model_prediction, iteration="", path="newResults/mean/test")
print(matrix)
chi2, p, dof, expected = chi2_contingency(matrix)
f1 = f1_score(actual, model_prediction, average="macro")
f1_w= f1_score(actual, model_prediction, average="weighted")
precision = precision_score(actual, model_prediction, average="macro")
recall = recall_score(actual, model_prediction, average="macro")
print(f"Accuracy: {valid_accuracy}")
print(f"chi:{chi2} p-value:{p}")
print(f"F1 Score: {f1}, weighted F1 Score: {f1_w} Precision: {precision} Recall: {recall}")

