import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle
from matplotlib import pyplot as plt

import hmax

# Initialize the model with the universal patch set
print('Constructing model')
model = hmax.HMAX('./universal_patch_set.mat')

# A folder with example images
example_images = datasets.ImageFolder(
    './presented_stimuli/',
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((375, 375)),  # Resize all images to (375, 375)
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])
)

# A dataloader that will run through all example images in one batch
dataloader = DataLoader(example_images)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Run the model on the example images
print('Running model on', device)
it = 0
c1_list = []
model = model.to(device)
for X, y in dataloader:
    c1 = model.get_c1_layers(X.to(device))
    c1_list.append(c1)
    print(it)
    it += 1
flatten_list = [j for sub in c1_list for j in sub]
split_list = [flatten_list[i:i + 8] for i in range(0, len(flatten_list), 8)]

parent_dir = './presented_stimuli/'
images = []
for subdir, dirs, files in os.walk(parent_dir):
    images.extend(files)
print(len(images))
print(len(split_list))

output = dict(zip(images, split_list))
print('Saving output of all layers to: output.pkl')
with open('output.pkl', 'wb') as f:
    pickle.dump(output, f)
print('[done]')

'''''
def show_images(c1_layer):
    fig, axs = plt.subplots(nrows=4, ncols=8)
    i = 0
    img_number = 3
    for data in c1_layer:
        axs[0, i].imshow(data[img_number, 0, :, :], interpolation='none')
        axs[1, i].imshow(data[img_number, 1, :, :], interpolation='none')
        axs[2, i].imshow(data[img_number, 2, :, :], interpolation='none')
        axs[3, i].imshow(data[img_number, 3, :, :], interpolation='none')
        i = i+1

    plt.show()
'''

def calculate_score(alldata):
    img_numb = len(alldata.items())
    avg = np.zeros((img_numb, 4))  # shape: image_amount, rotation_amount
    for x, data in enumerate(alldata.values()):  # iterate over c1 output for every image
        for scale in data:  # iterate over each scale
            # get different image for each rotation
            data1 = scale[0, 0, :, :]
            data2 = scale[0, 1, :, :]
            data3 = scale[0, 2, :, :]
            data4 = scale[0, 3, :, :]
            # calculate median for each rotation
            avg[x, 0] += np.median(data1)
            avg[x, 1] += np.median(data2)
            avg[x, 2] += np.median(data3)
            avg[x, 3] += np.median(data4)
    score = np.zeros(img_numb)  # array with one score per image
    for x in range(img_numb):
        score[x] = np.median(avg[x])
    score = (score - np.min(score)) / (np.max(score) - np.min(score))  # normalize score
    return score


with (open("output.pkl", "rb")) as openfile:
    complexity_classes = 5
    while True:
        try:
            output = pickle.load(openfile)
            score = calculate_score(output)
            image_names = output.keys()
            score_dict = dict(zip(image_names, score))
            with open('scores.pkl', 'wb') as f:
                pickle.dump(score_dict, f)
            sorted_scores = sorted(score_dict.items(), key=lambda x: x[1])
            split_scores = [sorted_scores[i:i + len(sorted_scores) // complexity_classes] for i in
                            range(0, len(sorted_scores) - len(sorted_scores) % complexity_classes,
                                  len(sorted_scores) // complexity_classes)]  # split the sorted dict into 5 (almost) even parts
            ''''
            split_values = [0.2, 0.4, 0.6, 0.8, 1]  # split scores into 5 parts
            split_scores = []

            for i in range(len(split_values)):
                if i == 0:
                    lower_bound = 0
                else:
                    lower_bound = split_values[i - 1]
                upper_bound = split_values[i]
                sublist = [item for item in sorted_scores if lower_bound < item[1] <= upper_bound]
                split_scores.append(sublist)
            '''
            if len(sorted_scores) % complexity_classes > 0:  # put the leftover scores into the last label
                split_scores[-1].extend(sorted_scores[-len(sorted_scores) % complexity_classes:])
            labels = [i for i in range(1, complexity_classes + 1)]
            sublist_dict = {labels[i]: [key for key, _ in sublist] for i, sublist in
                            enumerate(split_scores)}  # new dict with labels as key and filenames as value
            print(sublist_dict)
            with open('labels.pkl', 'wb') as f:
                pickle.dump(sublist_dict, f)
        except EOFError:
            break

