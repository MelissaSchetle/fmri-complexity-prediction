import os

import nibabel as nib
import numpy as np
import pickle
import pandas as pd
from functools import reduce


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
    #sessions = [fmri_image1, fmri_image2, fmri_image3, fmri_image4]
    #print(fmri_image1.shape)
    #sessions = [fmri_image1]
    def normalize(volume):
        """Normalize the volume"""
        min = 1
        max = 255
        volume[volume < min] = min
        volume[volume > max] = max
        volume = (volume - min) / (max - min)
        volume[np.isnan(volume)] = 0
        volume = volume.astype("float32")
        return volume
    '''
    for session in sessions:
        for image in session:
            image = normalize(image)
            #image[np.isnan(image)] = 0
    '''
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
    #print(stimuli_complexity)

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
        #print(df['id'].tolist())

    # Iterate over the stimuli list and match with stimuli_complexity
    for i, df in enumerate(dfList):
        df['Complexity'] = None  # Create a new column to store the stimulus number
        for j, stim_file in enumerate(df['stim_file']):
            for complexity in stimuli_complexity:
                if stim_file == complexity[1]:  # Match the stim_file with stimuli_complexity
                    df.at[j, 'Complexity'] = complexity[0]  # Store the corresponding number in the new column
                    break  # Break the loop once a match is found
    #print(DF_list[0])
    dfList = [df.set_index('id') for df in dfList]
    df = pd.concat(dfList, axis=0)
    print(df)
    #print(len(df))
    """
    complexities = np.zeros(5254)  # change 370 to variable
    for i, df in enumerate(dfList):
        complexities[i*37:i*37+37] = df['Complexity'].values
    print(len(complexities))
    """
    data = []
    i = 0
    #print(len(sessions))
    #print(sessions[0][0])
    #print(len((sessions[0][0,0,0,:])))
    #print(len(df.index.to_list()))
    #print(df.index.to_list()[1399:])
    for session in sessions:
        for volume in range(session.shape[3]):
            data.append((session[:, :, :, volume], df.at[i, 'Complexity']))
            i += 1

    """
    for s, session in (sessions):
        for r, run in (session):
            print(session)
            data.append((run, dfList[s].at[r, 'Complexity']))
            #data.append([(run, complexities[i]) for i in range(5254)])
    """
    print(data[4480:])
    #print(data[0][0][30,40,30])
    #print(len(data))


    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
    """
    # put data into new list which consists of tuples: ([x,y,z], complexity, id)
    trials = []
    for session in sessions:
        trials.append([(session[:, :, :, i], complexities[i], i) for i in range(session.shape[3])])
    print(trials[0][0][0][30, 40, 30])
    

    complexity1 = []
    complexity2 = []
    complexity3 = []
    #complexity4 = []
    #complexity5 = []

   # bin fmri into 5 complexity bins
    for d in data:
        for trial in d:
            if trial[1] == 1:
                complexity1.append(trial)
            elif trial[1] == 2:
                complexity2.append(trial)
            elif trial[1] == 3:
                complexity3.append(trial)
            #elif trial[1] == 4:
                #complexity4.append(trial)
            #else:
                #complexity5.append(trial)

    print(len(complexity1))
    print(len(complexity2))
    print(len(complexity3))
    #print(len(complexity4))
    #print(len(complexity5))

    binned_trials = [complexity1, complexity2, complexity3]
    
    with open('binned_trials.pkl', 'wb') as f:
        pickle.dump(binned_trials, f)
    #print(binned_trials)
    def calculate_std(list):
        pixel_lists = [img[0].flatten() for img in list]
        std_values = np.std(pixel_lists, axis=0)
        return std_values.reshape(list[0][0].shape)

    image1 = calculate_std(complexity1)
    image2 = calculate_std(complexity2)
    image3 = calculate_std(complexity3)
    #image4 = calculate_std(complexity4)
    #image5 = calculate_std(complexity5)

    print(image1[30, 40, 30])
    print(image1.shape)
    """



