import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle
from matplotlib import pyplot as plt
import hmax

calculate = False
if calculate:
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
    c1_list = []
    model = model.to(device)
    for X, y in dataloader:
        c1 = model.get_c1_layers(X.to(device))
        c1_list.append(c1)
    #flatten_list = [j for sub in c1_list for j in sub]
    split_list=c1_list
    #split_list = [flatten_list[i:i + 8] for i in range(0, len(flatten_list), 8)]

    parent_dir = './presented_stimuli/'
    images = []
    for subdir, dirs, files in sorted(os.walk(parent_dir)):
        print(files)
        images.extend(sorted(files))

    output = dict(zip(images, split_list))
    with open('output_sorted.pkl', 'wb') as f:
        pickle.dump(output, f)
    print('[done]')

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

def show_distribution(score, border1, border2): # creates histogram for the complexity classes
    plt.hist(score)
    plt.axvline(border1, color='orange')
    plt.axvline(border2, color='orange')
    plt.ylabel("Number of stimuli")
    plt.xlabel("Complexity score")
    plt.show()

def get_classes():
    #with (open("output_sorted.pkl", "rb")) as openfile:
    with (open("output.pkl", "rb")) as openfile:
        complexity_classes = 3
        while True:
            try:
                output = pickle.load(openfile)
                score = calculate_score(output)
                image_names = output.keys()
                score_dict = dict(zip(image_names, score))
                sorted_scores = sorted(score_dict.items(), key=lambda x: x[1])
                # split into even parts
                split_scores = [sorted_scores[i:i + len(sorted_scores) // complexity_classes] for i in
                                range(0, len(sorted_scores) - len(sorted_scores) % complexity_classes,
                                      len(sorted_scores) // complexity_classes)]  # split the sorted dict into 3 (almost) even parts
                print(split_scores[0][-1], split_scores[1][-1])
                show_distribution(score, split_scores[0][-1][1], split_scores[1][-1][1])

                if len(sorted_scores) % complexity_classes > 0:  # put the leftover scores into the last label
                    split_scores[-1].extend(sorted_scores[-(len(sorted_scores) % complexity_classes):])

                labels = [i for i in range(complexity_classes)]
                sublist_dict = {labels[i]: [key for key, _ in sublist] for i, sublist in
                                enumerate(split_scores)}  # new dict with labels as key and filenames as value

                with open('labels_sorted.pkl', 'wb') as f:
                    pickle.dump(sublist_dict, f)

            except EOFError:
                break


get_classes()
