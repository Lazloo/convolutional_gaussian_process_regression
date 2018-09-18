from sklearn.feature_extraction import image
from matplotlib import pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from typing import List
from PIL import Image
import os


class GPR_Class:

    def __init__(self):
        self.data = []
        self.x_min = []
        self.x_max = []
        self.y_min = []
        self.y_max = []

    @staticmethod
    def normalize(x: List[float], min_in: float, max_in: float, min_out: float, max_out: float) -> list:
        x_norm = ((x - min_in) / (max_in - min_in) * (max_out - min_out) + min_out)
        return x_norm

    def create_GPR_model(self, X, Y):
        self.x_min = np.amin(X, axis=0)
        self.x_max = np.amax(X, axis=0)
        self.y_min = np.amin(Y, axis=0)
        self.y_max = np.amax(Y, axis=0)
        assert np.min(self.y_min) < np.max(self.y_max), 'y_min should be smaller than y_max'
        assert np.min(self.x_min) < np.max(self.x_max), 'x_min should be smaller than x_max'

        self.X = X
        self.Y = Y

        x_norm = self.normalize(X, min_in=self.x_min, max_in=self.x_max, min_out=0, max_out=1)
        y_norm = self.normalize(Y, min_in=self.y_min, max_in=self.y_max, min_out=0, max_out=1)
        kernel = 1.0 * Matern(length_scale=0.1, length_scale_bounds=(1e-5, 1e5), nu=1.5) + WhiteKernel()
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(x_norm, y_norm)

        return self.gp

    def predict(self, x_test):
        x_interpolation = self.normalize(x_test,
                                         min_in=self.x_min, max_in=self.x_max,
                                         min_out=0, max_out=1)
        y_mean_interpol, y_cov_norm = self.gp.predict(x_interpolation[np.newaxis, :], return_cov=True)

        y_mean = self.normalize(y_mean_interpol,
                                min_in=0, max_in=1,
                                min_out=self.y_min, max_out=self.y_max)
        y_cov = y_cov_norm * ((self.y_max - self.y_min) ** 2)

        return y_mean, y_cov

    def plot_1D_interpolation(self):
        x_test = np.linspace(0, 1, 1000)
        x_interpolation = self.normalize(x_test,
                                         min_in=0, max_in=1,
                                         min_out=self.x_min, max_out=self.x_max)
        # Rescale and perform GPR prediction
        y_mean, y_cov = self.predict(x_interpolation)

        plt.figure()
        plt.plot(x_interpolation, y_mean, 'k', lw=3, zorder=9)
        plt.fill_between(x_interpolation, (y_mean - np.sqrt(np.diag(y_cov))).transpose(),
                         (y_mean + np.sqrt(np.diag(y_cov))).transpose(),
                         alpha=0.5, color='k')
        plt.scatter(self.X[:, 0], self.Y,
                    c='r', s=1, zorder=10, edgecolors=(0, 0, 0))
        plt.tight_layout()
        plt.show()

    @staticmethod
    def read_label(filename, label_choices, output_type: str='binary'):
        basename = os.path.basename(filename)
        labels: bytes = basename.split('_')[0]

        indices = [label_choices.index(l) for l in labels]
        if output_type == 'binary':
            data = [[0 for i in range(len(label_choices))] for j in range(len(labels))]
            for iL in range(0, len(labels)):
                data[iL][indices[iL]] = 1
        else:
            data = indices

        return data

    def read_labels(self, dir_name, ext='.png', label_choices='0', output_type: str='binary'):
        fd = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name) if fn.endswith(ext)]
        label_array = np.array([self.read_label(ifd, label_choices, output_type) for ifd in fd])
        n_examples = label_array.shape[0]
        n_labels = label_array.shape[1]
        if output_type == 'binary':
            n_label_choices = label_array.shape[2]
        else:
            n_label_choices = 1
        label_array = label_array.reshape([n_examples, n_label_choices * n_labels])

        return label_array

    @staticmethod
    def read_images(dir_name, ext='.png', is_single_image=False):
        if is_single_image:
            im_raw = [Image.open(dir_name).convert('L')]
        else:
            fd = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name) if fn.endswith(ext)]
            im_raw = [Image.open(iFile).convert('L') for iFile in fd]
        image_data = [np.asarray(i) for i in im_raw]
        image_data = [i / 255 for i in image_data]
        # binary_image_array = [(iData > 125.5) * 1.0 for iData in image_data]
        # image_data/255
        return image_data

    @staticmethod
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
