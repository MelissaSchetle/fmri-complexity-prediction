import os

import nibabel as nib
import numpy as np
import pickle
import pandas as pd


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

    # load complexity scores
    with open('labels.pkl', 'rb') as file_handle:
        scores = pickle.load(file_handle)

    # make list of sublists which consist of [complexity values, img names]
    i = 1
    stimuli_complexity = []
    for names in scores.values():
        for name in names:
            stimuli_complexity.append([i, name])
        i += 1
    #print(stimuli_complexity)

    # make list consisting of dataframes corresponding to the run tables
    path = "trial_info/"
    all_files = os.listdir(path)
    DF_list = []
    columns = ['onset', 'duration', 'Subj', 'Sess', 'Run', 'Trial', 'ImgName', 'ImgType', 'StimOn(s)', 'StimOff(s)', 'Response', 'RT', 'stim_file']
    for fle in all_files:
        df = pd.read_csv("trial_info/"+fle, sep="\t", index_col=False, names=columns, skiprows=1)
        DF_list.append(df)

    # list of images used in sessions in correct order
    stimuli = []
    for df in DF_list:
        stimuli.append(df['stim_file'])

    # Iterate over the stimuli list and match with stimuli_complexity
    for i, df in enumerate(DF_list):
        df['Complexity'] = None  # Create a new column to store the stimulus number
        for j, stim_file in enumerate(df['stim_file']):
            for complexity in stimuli_complexity:
                if stim_file == complexity[1]:  # Match the stim_file with stimuli_complexity
                    df.at[j, 'Complexity'] = complexity[0]  # Store the corresponding number in the new column
                    break  # Break the loop once a match is found
    #print(DF_list[0])

    complexities = np.zeros(5254)  # change 370 to variable
    for i, df in enumerate(DF_list):
        complexities[i*37:i*37+37] = df['Complexity'].values
    print(len(complexities))

    # put data into new list which consists of tuples: ([x,y,z], complexity, id)
    trials = []
    for session in sessions:
        trials.append([(session[:, :, :, i], complexities[i], i) for i in range(session.shape[3])])
    print(trials[0][0][0][30, 40, 30])

    complexity1 = []
    complexity2 = []
    complexity3 = []
    complexity4 = []
    complexity5 = []

   # bin fmri into 5 complexity bins
    for session in trials:
        for trial in session:
            if trial[1] == 1:
                complexity1.append(trial)
            elif trial[1] == 2:
                complexity2.append(trial)
            elif trial[1] == 3:
                complexity3.append(trial)
            elif trial[1] == 4:
                complexity4.append(trial)
            else:
                complexity5.append(trial)

    print(len(complexity1))
    print(len(complexity2))
    print(len(complexity3))
    print(len(complexity4))
    print(len(complexity5))

    binned_trials = [complexity1, complexity2, complexity3, complexity4, complexity5]
    #print(binned_trials)
