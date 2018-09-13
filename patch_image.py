from sklearn.feature_extraction import image
import time as time
import GPR_Package.GPR_Script as GP
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os as os


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
    label_array = label_array.reshape([n_examples, n_label_choices*n_labels])
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
    image_data = [i/255 for i in image_data]
    # binary_image_array = [(iData > 125.5) * 1.0 for iData in image_data]
    # image_data/255
    return image_data


def generate_patch_aggregation(images_array, scaling_factor=[2, 2]):
    patches = [image.extract_patches_2d(image=iImage,
                                        patch_size=(round(iImage.shape[0] / scaling_factor[0]),
                                                    round(iImage.shape[1] / scaling_factor[1])))
               for iImage in images_array]
    features = np.array([[np.sum(iPatch) for iPatch in iImage] for iImage in patches])

    return features


# label_array = read_labels('images/test_data_images/for_testing', label_choices='0123456789')
label_array = read_labels('images/test_data_images/for_testing', label_choices='01')
images_array = read_images('images/test_data_images/for_testing')
features = generate_patch_aggregation(images_array, scaling_factor=[2, 2])

gp_test = GP.GPR_Class()

t1 = time.time()
gp_test.create_GPR_model(features, label_array)
t2 = time.time()
print('GPR Modelling took: ' + str(t1 - t2) + 's')
gp_test.predict(features[0])


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
