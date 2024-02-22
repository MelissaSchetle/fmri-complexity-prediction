import os

import nibabel as nib
import numpy as np
import pickle
import torch


def normalize(volume):
    x = volume.flatten()
    flattened_array = x[~np.isnan(x)]
    indices_of_lowest = np.argsort(flattened_array)[int(len(flattened_array) * 0.01)]
    lowest_values = flattened_array[indices_of_lowest]
    indices_of_highest = np.argsort(flattened_array)[int(len(flattened_array) * 0.99)]
    highest_values = flattened_array[indices_of_highest]
    min = lowest_values
    max = highest_values
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume[np.isnan(volume)] = 0
    return volume


def padding(array, xx, yy, zz):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]
    z = array.shape[2]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    c = (zz - z) // 2
    cc = zz - c - z

    return np.pad(array, pad_width=((a, aa), (b, bb), (c, cc)), mode='edge')


def load_data():
    masks = []
    directory = 'brain_masks'
    for filename in sorted(os.listdir(directory)):
        f = os.path.join(directory, filename)
        #print(f)
        masks.append(nib.load(f).get_fdata()[:, :, :, np.newaxis])
    subjects = []
    directory = 'fMRI_images'

    counter = -1
    for subdir, dirs, files in sorted(os.walk(directory)):
        sessions = []
        for file in sorted(files):
            print(os.path.join(subdir, file))
            sessions.append(np.array(nib.load(os.path.join(subdir, file)).get_fdata()*masks[counter]))

        subject = []
        for session in sessions:
            s = []
            for volume in range(session.shape[3]):
                s.append(np.array(normalize(session[:, :, :, volume])))
                #s.append(np.array(session[:, :, :, volume]))       #for sd and mean
            subject.append(s)
        #sessions = [np.array([padding(normalize(image),72,92,72) for image in session]) for session in sessions]
        if counter != -1:
            print(len(subject))
            subjects.append(subject)
        counter += 1
    return subjects


def make_test_train_data(subjects):
    # load complexity scores
    with open('labels_sorted.pkl', 'rb') as file_handle:
        scores = pickle.load(file_handle)

    complexity_dict = {}
    i = 0
    for names in scores.values():
        for name in names:
            complexity_dict[name] = i
        i += 1

    content = []
    # list of image names in right order
    directory = 'imgnames'
    img_names = []
    for filename in sorted(os.listdir(directory)):
        f = os.path.join(directory, filename)
        img_names_file = open(f, "r")
        names = img_names_file.read()
        img_names.append(filter(None, names.split("\n")))
        # print(img_names)
        img_names_file.close()

    for c in range(len(img_names)):
        # for session in sessions:
        subject = []
        for i in img_names[c]:
            complexity = complexity_dict.get(i)
            subject.append(complexity)
        content.append(subject)

    print("len content", len(content[0]))
    print(content[0])
    print("length subjects", len(subjects))

    data = []
    for c, subject in enumerate(subjects):
        images = []
        for session in subject:
            for img in session:
                images.append(img)
        s = list(zip(images, content[c]))
        data.append(s)


    print("done data")
    return data



def extract_sd_mean(data, sd): #extracts the mean or standard deviation
    shape = data[0][0][0].shape
    print(shape)

    # Define the size of the cubes
    cube_size = 5

    # Calculate the number of cubes in each dimension
    num_cubes_x = shape[0] // cube_size
    num_cubes_y = shape[1] // cube_size
    num_cubes_z = shape[2] // cube_size
    print(num_cubes_x, num_cubes_z, num_cubes_z)

    # Initialize an empty list to store the cubes
    fmri_sd = []
    for subject in data:
        print("data ",len(data))
        d = []
        print(len(subject))
        for fmri in subject:
            sd = (np.zeros((num_cubes_x, num_cubes_y, num_cubes_z)), fmri[1])

            # Iterate through the original array and split it into cubes
            for x in range(num_cubes_x):
                for y in range(num_cubes_y):
                    for z in range(num_cubes_z):
                        # Extract a cube from the original array
                        cube = fmri[0][x * cube_size:(x + 1) * cube_size,
                               y * cube_size:(y + 1) * cube_size,
                               z * cube_size:(z + 1) * cube_size]
                        if sd:
                            sd[0][x, y, z] = np.nanstd(cube)
                        else:
                            sd[0][x, y, z] = np.nanmean(cube)
            #d.append((torch.unsqueeze(torch.Tensor(sd[0]), dim=0), sd[1]))
            d.append((torch.unsqueeze(torch.Tensor(normalize(sd[0])), dim=0), sd[1]))
        fmri_sd.append(d)
    print("done with loop")


    print(fmri_sd[0][0][0].shape)
    print(fmri_sd[0][0])
    return fmri_sd

def save(data):
    directory = "subject_pickles"
    os.makedirs(directory, exist_ok=True)
    for c, subject in enumerate(data):
        path = os.path.join(directory, f"subject_{c}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(subject, f, protocol=4)

    print("done saving")


if __name__ == '__main__':
    sessions = load_data()
    #data = calculate_sd(sessions)
    data = make_test_train_data(sessions)
    #data = calculate_sd(data)
    save(data)





