from sklearn.feature_extraction import image
import operator
import time as time
from tqdm import tqdm
import GPR_Package.GPR_Script as GP
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os as os
import pickle


def read_label(filename, label_choices):
    basename = os.path.basename(filename)
    labels: bytes = basename.split('_')[0]

    indices = [label_choices.index(l) for l in labels]
    data = [[0 for i in range(len(label_choices))] for j in range(len(labels))]
    for iL in range(0, len(labels)):
        data[iL][indices[iL]] = 1

    return data


def read_labels(dir_name, ext='.png', label_choices='0'):
    fd = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name) if fn.endswith(ext)]
    label_array = np.array([read_label(ifd, label_choices) for ifd in fd])
    n_examples = label_array.shape[0]
    n_labels = label_array.shape[1]
    n_label_choices = label_array.shape[2]
    label_array = label_array.reshape([n_examples, n_label_choices * n_labels])
    # for fn in os.listdir(dir_name):
    #     if fn.endswith(ext):
    #         fd = os.path.join(dir_name, fn)
    #         labels.append(read_label(fd, label_choices))
    return label_array


# Todo: no threshold as otherwise a picture is purely black + normalizing
def read_images(dir_name, ext='.png'):
    fd = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name) if fn.endswith(ext)]
    im_raw = [Image.open(iFile).convert('L') for iFile in fd]
    image_data = [np.asarray(i) for i in im_raw]
    image_data = [i / 255 for i in image_data]
    # binary_image_array = [(iData > 125.5) * 1.0 for iData in image_data]
    # image_data/255
    return image_data


def generate_patch_aggregation(images_array, scaling_factor=[2, 2]):
    features_list = \
        [
            [
                np.sum(iPatch)
                for iPatch
                in image.extract_patches_2d(image=iImage,
                                            patch_size=(round(iImage.shape[0] / scaling_factor[0]),
                                                        round(iImage.shape[1] / scaling_factor[1])))
            ]
            for iImage in images_array
        ]

    return np.array(features_list)


# dir = 'images/test_data_images/for_testing'
dir = 'images/real_2/test'
dir_test = 'images/real_2/test_short'
label_choices = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
# label_choices='0123456789'

label_array = read_labels(dir, label_choices=label_choices)
images_array = read_images(dir)
features = generate_patch_aggregation(images_array, scaling_factor=[2, 2])

label_array_test = read_labels(dir_test, label_choices=label_choices)
images_array_test = read_images(dir_test)
features_test = generate_patch_aggregation(images_array_test, scaling_factor=[2, 2])

gp_test = GP.GPR_Class()

t1 = time.time()
gp_test.create_GPR_model(features, label_array)
t2 = time.time()
print('GPR Modelling took: ' + str(t1 - t2) + 's')

m, s = gp_test.predict(features_test[2])
indices = np.where(m > 1e-10)[1]
print(label_choices[m[0][0:62].argmax()])
print(label_choices[m[0][63:].argmax()])

file_pi = open('filename_2Letters.obj', 'wb')
pickle.dump(obj=gp_test, file=file_pi)

test_array = [False] * len(label_array)
for iTest in tqdm(range(0, len(label_array))):
    m, s = gp_test.predict(features[iTest])
    test_array[iTest] = ((label_array[iTest] == 1) == (m > 0.1)).sum() == len(label_array)

# gp_test.plot_1D_interpolation()
#
#
#
# im = Image.open('0_2e7558e8-9062-443d-a3a0-8808c519dab6.png').convert('L')
#
#
# data = np.asarray(im)
#
# n_factor = 1.1
# iImage = images_array[0]
# patches = image.extract_patches_2d(image=iImage,
#                                     patch_size=(round(iImage.shape[0] / 1.2),
#                                                 round(iImage.shape[1] / 1.2)))
#
