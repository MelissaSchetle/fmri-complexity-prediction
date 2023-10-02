import os
import random

import nibabel as nib
import numpy as np
import pickle
import pandas as pd
import torch
from matplotlib import pyplot as plt

save = False

if __name__ == '__main__':
    # load data and put it into a list
    fmri_image1 = nib.load('fMRI_images/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-01.nii.gz').get_fdata()
    fmri_image2 = nib.load('fMRI_images/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-02.nii.gz').get_fdata()
    fmri_image3 = nib.load('fMRI_images/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-03.nii.gz').get_fdata()
    fmri_image4 = nib.load('fMRI_images/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-04.nii.gz').get_fdata()
    fmri_image5 = nib.load('fMRI_images/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-05.nii.gz').get_fdata()
    fmri_image6 = nib.load('fMRI_images/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-06.nii.gz').get_fdata()
    fmri_image7 = nib.load('fMRI_images/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-07.nii.gz').get_fdata()
    fmri_image8 = nib.load('fMRI_images/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-08.nii.gz').get_fdata()
    fmri_image9 = nib.load('fMRI_images/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-09.nii.gz').get_fdata()
    fmri_image10 = nib.load('fMRI_images/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-10.nii.gz').get_fdata()
    fmri_image11 = nib.load('fMRI_images/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-11.nii.gz').get_fdata()
    fmri_image12 = nib.load('fMRI_images/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-12.nii.gz').get_fdata()
    fmri_image13 = nib.load('fMRI_images/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-13.nii.gz').get_fdata()
    fmri_image14 = nib.load('fMRI_images/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-14.nii.gz').get_fdata()
    fmri_image15 = nib.load('fMRI_images/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-15.nii.gz').get_fdata()
    sessions = [fmri_image1, fmri_image2, fmri_image3, fmri_image4, fmri_image5, fmri_image6, fmri_image7, fmri_image8, fmri_image9, fmri_image10, fmri_image11, fmri_image12, fmri_image13, fmri_image14, fmri_image15]
    def normalize(volume):
        min = 1
        max = 256
        volume[volume < min] = min
        volume[volume > max] = max
        volume = (volume - min) / (max - min)
        volume[np.isnan(volume)] = 0
        volume = volume.astype("float32")
        return volume

    sessions = [np.array([normalize(image) for image in session]) for session in sessions]
    # load complexity scores
    with open('labels.pkl', 'rb') as file_handle:
        scores = pickle.load(file_handle)

    # make list of sublists which consist of [complexity values, img names]
    i = 0
    stimuli_complexity = []
    for names in scores.values():
        for name in names:
            stimuli_complexity.append([i, name])
        i += 1

    # make list consisting of dataframes corresponding to the run tables
    path = "trial_info/"
    all_files = os.listdir(path)
    dfList = []
    columns = ['onset', 'duration', 'Subj', 'Sess', 'Run', 'Trial', 'ImgName', 'ImgType', 'StimOn(s)', 'StimOff(s)', 'Response', 'RT', 'stim_file', 'id']
    for fle in all_files:
        df = pd.read_csv("trial_info/"+fle, sep="\t", index_col=False, names=columns, skiprows=1)
        df['id'] = 370*(df['Sess']-1)+(df['Run']-1)*37+df['Trial']
        dfList.append(df)

    # list of images used in sessions in correct order
    stimuli = []
    for df in dfList:
        stimuli.append(df['stim_file'])
    l = 0
    for df in dfList:
        df['id'] = list(range(l, l+len(df)))
        l = len(df)+l

    # Iterate over the stimuli list and match with stimuli_complexity
    for i, df in enumerate(dfList):
        df['Complexity'] = None  # Create a new column to store the stimulus number
        for j, stim_file in enumerate(df['stim_file']):
            for complexity in stimuli_complexity:
                if stim_file == complexity[1]:  # Match the stim_file with stimuli_complexity
                    df.at[j, 'Complexity'] = complexity[0]  # Store the corresponding number in the new column
                    break  # Break the loop once a match is found

    dfList = [df.set_index('id') for df in dfList]
    df = pd.concat(dfList, axis=0)
    print(df)
    data = []
    i = 0
    images = []
    labels = []
    for session in sessions:
        for volume in range(session.shape[3]):
            if df.at[i, 'Complexity'] is not None:
                images.append(torch.unsqueeze(torch.Tensor(session[:, :, :, volume]), dim=0))
                labels.append(int(df.at[i, 'Complexity']))
                data.append((session[:, :, :, volume], df.at[i, 'Complexity']))
            i += 1

    split = int(len(data) * 0.9)
    random_data = data
    random.shuffle(random_data)
    train_dataset = random_data[:split]
    test_dataset = random_data[split:]
    if save:
        with open('new_train.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)
        with open('new_test.pkl', 'wb') as f:
            pickle.dump(test_dataset, f)

    def calculate_std(list):
        pixel_lists = [img[0].flatten() for img in list]
        std_values = np.std(pixel_lists, axis=0)
        return std_values.reshape(list[0][0].shape)
    def bin_data(complexities):
        # put data into new list which consists of tuples: ([x,y,z], complexity, id)
        trials = []
        for session in sessions:
            trials.append([(session[:, :, :, i], complexities[i], i) for i in range(session.shape[3])])

        complexity1 = []
        complexity2 = []
        complexity3 = []

        # bin fmri into 5 complexity bins
        for d in data:
            for trial in d:
                if trial[1] == 1:
                    complexity1.append(trial)
                elif trial[1] == 2:
                    complexity2.append(trial)
                elif trial[1] == 3:
                    complexity3.append(trial)

        print(len(complexity1))
        print(len(complexity2))
        print(len(complexity3))

        binned_trials = [complexity1, complexity2, complexity3]

        with open('binned_trials.pkl', 'wb') as f:
            pickle.dump(binned_trials, f)

        image1 = calculate_std(complexity1)
        image2 = calculate_std(complexity2)
        image3 = calculate_std(complexity3)

        print(image1[30, 40, 30])
        print(image1.shape)

def calculate_sd():
    shape = data[0][0].shape

    # Define the size of the cubes
    cube_size = 5

    # Calculate the number of cubes in each dimension
    num_cubes_x = shape[0] // cube_size
    num_cubes_y = shape[1] // cube_size
    num_cubes_z = shape[2] // cube_size
    print(num_cubes_x, num_cubes_z, num_cubes_z)

    # Initialize an empty list to store the cubes
    fmri_sd = []
    for fmri in data:
        sd = (np.zeros((num_cubes_x, num_cubes_y, num_cubes_z)), fmri[1])
        # Iterate through the original array and split it into cubes
        for x in range(num_cubes_x):
            for y in range(num_cubes_y):
                for z in range(num_cubes_z):
                    # Extract a cube from the original array
                    cube = fmri[0][x * cube_size:(x + 1) * cube_size,
                           y * cube_size:(y + 1) * cube_size,
                           z * cube_size:(z + 1) * cube_size]
                    sd[0][x, y, z] = np.std(cube)
        fmri_sd.append(sd)
    #with open('sd_data.pkl', 'wb') as f:
        #pickle.dump(fmri_sd, f)
    print(fmri_sd[0][0].shape)
    print(fmri_sd[0])
    return fmri_sd
fmri_sd = calculate_sd()

def show_data(elem = 0):
    for i in range(fmri_sd[elem][0].shape[2]):
        plt.plot(fmri_sd[elem][0][: , :, i])
        plt.show()

show_data()




