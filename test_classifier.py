from sklearn.feature_extraction import image
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import operator
import time as time
from tqdm import tqdm
import GPR_Package.GPR_Script as GP
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os as os
import pickle

dir = 'images/real_2/test'
label_choices = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

gp_test = GP.GPR_Class()

label_array = gp_test.read_labels(dir, label_choices=label_choices, output_type='index')
images_array = gp_test.read_images(dir)
features = gp_test.generate_patch_aggregation(images_array, scaling_factor=[2, 2])

X = features
Y = label_array
gp_test.x_min = np.amin(X, axis=0)
gp_test.x_max = np.amax(X, axis=0)
gp_test.y_min = np.amin(Y, axis=0)
gp_test.y_max = np.amax(Y, axis=0)
assert np.min(gp_test.y_min) < np.max(gp_test.y_max), 'y_min should be smaller than y_max'
assert np.min(gp_test.x_min) < np.max(gp_test.x_max), 'x_min should be smaller than x_max'

gp_test.X = X
gp_test.Y = Y

x_norm = gp_test.normalize(X, min_in=gp_test.x_min, max_in=gp_test.x_max, min_out=0, max_out=1)
y_norm = gp_test.normalize(Y, min_in=gp_test.y_min, max_in=gp_test.y_max, min_out=0, max_out=1)
kernel = 1.0 * Matern(length_scale=0.1, length_scale_bounds=(1e-5, 1e5), nu=1.5) + WhiteKernel()

# gp_test.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(x_norm, y_norm)
gp0 = GaussianProcessClassifier(kernel=kernel).fit(X=x_norm, y=Y[:, 0])
gp1 = GaussianProcessClassifier(kernel=kernel).fit(X=x_norm, y=Y[:, 1])

dir_test = 'images/real_2/test_short'
label_array_test = gp_test.read_labels(dir_test, label_choices=label_choices, output_type='index')
images_array_test = gp_test.read_images(dir_test)
features_test = gp_test.generate_patch_aggregation(images_array_test, scaling_factor=[2, 2])

x_interpolation = gp_test.normalize(features_test,
                                 min_in=gp_test.x_min, max_in=gp_test.x_max,
                                 min_out=0, max_out=1)
y_mean_interpol0= gp0.predict(np.array(x_interpolation))
y_mean_interpol1= gp1.predict(np.array(x_interpolation))

y_mean = self.normalize(y_mean_interpol,
                        min_in=0, max_in=1,
                        min_out=gp_test.y_min, max_out=gp_test.y_max)

import pickle

with open('filename_2Letters.obj', 'wb') as file_pi:
    pickle.dump(obj=gp_test, file=file_pi)
