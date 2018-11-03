from sklearn.feature_extraction import image
from matplotlib import pyplot as plt
import logging
import gpflow
import time as time
import itertools
import sys
import numpy as np
from operator import itemgetter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, Exponentiation
import scipy as scipy
import skimage.measure as measure
from typing import List
from PIL import Image
import os


class GPR_Class:

    def __init__(self):

        self.logger = logging.getLogger('GPR Classfication')
        self.gp_package = 'sklearn'

        self.verbose = True
        self.verbose_debug = True

        self.gp = GaussianProcessRegressor()

        self.data = []
        self.x_min = []
        self.x_max = []
        self.y_min = []
        self.y_max = []
        self.X = []
        self.Y = []

        self.label_array = np.array([])
        self.label_array_train = np.array([])
        self.label_array_test = np.array([])
        self.label_array_eval = np.array([])

        self.images_array = np.array([])
        self.images_array_train = np.array([])
        self.images_array_test = np.array([])
        self.images_array_eval = np.array([])

        self.indices_distributed = np.array([])
        self.train_indices = np.array([])
        self.test_indices = np.array([])
        self.evaluation_indices = np.array([])

        self.features_train = np.array([])
        self.features_test = np.array([])
        self.features_eval = np.array([])

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

        for i in range(len(self.x_min)):
            if self.x_min[i] == self.x_max[i]:
                if abs(self.x_max[i]) > 0:
                    add_value = min(1e-1, abs(self.x_max[i] * 1e-2))
                else:
                    add_value = 1e-1

                self.x_max[i] += add_value
                # self.logger.warning('Added ' + str(add_value) + ' to maximum of variable ' +
                #                     str(i) + ' as min and max was equal')

        assert np.min(self.x_min) < np.max(self.x_max), 'x_min should be smaller than x_max'
        self.X = X
        self.Y = Y

        t1 = time.time()
        x_norm = self.normalize(X, min_in=self.x_min, max_in=self.x_max, min_out=0, max_out=1)
        t2 = time.time()
        if self.verbose_debug:
            print('normalizing X took : ' + str(t2 - t1) + 's')

        t1 = time.time()
        y_norm = self.normalize(Y, min_in=self.y_min, max_in=self.y_max, min_out=0, max_out=1)
        t2 = time.time()
        if self.verbose_debug:
            print('normalizing Y took : ' + str(t2 - t1) + 's')

        kernel = 1.0 * Matern(length_scale=0.1, length_scale_bounds=(1e-5, 1e5), nu=2.5) \
                 + WhiteKernel()
        # kernel = 1.0 * RBF()
        t1 = time.time()
        if self.gp_package == 'sklearn':
            self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(x_norm, y_norm)
        elif self.gp_package == 'gp_flow':
            self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(x_norm, y_norm)
            self.gp = gpflow.models.GPR(x_norm, y_norm,
                                        kern=gpflow.kernels.Matern52(input_dim=x_norm.shape[1]) +
                                             gpflow.kernels.Linear(input_dim=x_norm.shape[1]) +
                                             gpflow.kernels.White(input_dim=x_norm.shape[1])
                                        )
            self.gp.compile()
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(self.gp)
        else:
            raise AssertionError('unknown GP Method: ' + self.gp_package)
        t2 = time.time()
        if self.verbose_debug:
            print('training GPR : ' + str(t2 - t1) + 's')

        return self.gp

    def predict(self, x_test):

        x_interpolation = x_test
        if len(x_test.shape) == 1:
            x_interpolation = x_interpolation[np.newaxis, :]

        n_test_points = x_interpolation.shape[0]
        for i in range(n_test_points):
            x_interpolation[i] = self.normalize(x_interpolation[i],
                                                min_in=self.x_min, max_in=self.x_max,
                                                min_out=0, max_out=1)

        if self.gp_package == 'sklearn':
            y_mean_interpol, y_std_norm = self.gp.predict(x_interpolation, return_std=True)
        elif self.gp_package == 'gp_flow':
            y_mean_interpol, y_var_norm = self.gp.predict_y(x_interpolation)
            y_std_norm = y_var_norm ** 0.5
        else:
            raise AssertionError('unknown GP Method: ' + self.gp_package)

        y_mean = y_mean_interpol
        y_std = y_std_norm
        for i in range(n_test_points):
            y_mean[i] = self.normalize(y_mean[i],
                                       min_in=0, max_in=1,
                                       min_out=self.y_min, max_out=self.y_max)
            if isinstance(self.y_max, np.float):
                y_std[i] = y_std_norm[i] * ((self.y_max - self.y_min) ** 2)
            else:
                y_std[i] = y_std_norm[i] * ((max(self.y_max) - min(self.y_min)) ** 2)

        return y_mean, y_std

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
    def read_label(filename, label_choices, output_type: str = 'binary'):
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

    def read_labels(self, dir_name, ext='.png', label_choices='0', output_type: str = 'binary'):
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

    @staticmethod
    def convolution_step_edges_max_pooling(im_array,
                                           windows_size: tuple,
                                           weights: np.array = np.array([]),
                                           weight_shape: list = []):
        if len(weights) > 0:
            if len(weight_shape) == 0:
                raise AssertionError('weight_shape is empty')
            weight_matrix = weights.reshape(weight_shape)
        else:
            weight_matrix = [[1, -1], [-1, 1]]
        im_array_convolve = scipy.signal.convolve(in1=im_array, in2=weight_matrix, mode="valid")
        im_array_convolve_max = im_array_convolve * (
                im_array_convolve == scipy.ndimage.filters.maximum_filter(input=im_array_convolve, size=windows_size)
        )
        im_array_convolve_max_reduce = measure.block_reduce(im_array_convolve_max, windows_size, np.max)

        return im_array_convolve_max_reduce

    @staticmethod
    def generate_indices_train_test_evaluation(label_array, ratio: tuple = (0.8, 0.1, 0.1)):

        assert (max(ratio) <= 1) & (min(ratio) >= 0), 'Each entry of ratio must be between 0.0 and 1.0'

        unique_label = len(label_array[0])
        indices = [None] * unique_label
        n_data = len(label_array)
        for i in range(n_data):
            idx = label_array[i].argmax()
            if indices[idx] is None:
                indices[idx] = []
            indices[idx] += [i]

        np.random.seed(0)
        indices_distributed = [None] * unique_label

        train_indices = []
        test_indices = []
        evaluation_indices = []
        for i in range(unique_label):
            n_data_partial = len(indices[i])
            indices_rand = np.random.permutation(n_data_partial)
            n_train = round(ratio[0] * n_data_partial)
            n_test = round((n_data_partial - n_train) / 2)
            n_evaluation = n_data_partial - n_train - n_test
            # 0:X does not include X
            n_test_actual = round(n_test * ratio[1])
            n_evaluation_actual = round(n_evaluation * ratio[2])
            indices_distributed[i] = [
                itemgetter(*indices_rand[0:n_train])(indices[i]),
                itemgetter(*indices_rand[n_train:n_train + n_test_actual])(indices[i]),
                itemgetter(*indices_rand[n_train + n_test:n_train + n_test + n_evaluation_actual])(indices[i])
            ]
            # if (len(indices_distributed[i][0]) + len(indices_distributed[i][1]) + len(indices_distributed[i][2])) \
            #         != n_data_partial:
            #     print('stop')
            train_indices += indices_distributed[i][0]
            test_indices += indices_distributed[i][1]
            evaluation_indices += indices_distributed[i][2]

        return indices_distributed, train_indices, test_indices, evaluation_indices

    def generate_train_test_eval_indices(self, ratio_train_test_eval: tuple):
        self.indices_distributed, self.train_indices, self.test_indices, self.evaluation_indices = \
            self.generate_indices_train_test_evaluation(self.label_array, ratio=ratio_train_test_eval)

    def load_data(self, dir_name: str, label_choices: str, ratio_train_test_eval: tuple):
        self.label_array = self.read_labels(dir_name=dir_name, label_choices=label_choices, output_type='binary')
        self.images_array = self.read_images(dir_name=dir_name)
        self.generate_train_test_eval_indices(ratio_train_test_eval)

        self.images_array_train = itemgetter(*self.train_indices)(self.images_array)
        self.images_array_test = itemgetter(*self.test_indices)(self.images_array)
        self.images_array_eval = itemgetter(*self.evaluation_indices)(self.images_array)

        self.label_array_train = itemgetter(*self.train_indices)(self.label_array)
        self.label_array_test = itemgetter(*self.test_indices)(self.label_array)
        self.label_array_eval = itemgetter(*self.evaluation_indices)(self.label_array)

    def generate_test_train_eval_data(self,
                                      windows_size: list,
                                      weights: np.ndarray = np.ndarray([]),
                                      weight_shape: list = [[]]):

        def generate_data(data_set_type: str = 'train'):
            if data_set_type == 'train':
                data_set = self.images_array_train
            elif data_set_type == 'test':
                data_set = self.images_array_test
            elif data_set_type == 'eval':
                data_set = self.images_array_eval
            else:
                raise ValueError('Unknown data_set_type: ' + data_set_type)

            data_out = np.array(
                [list(itertools.chain(*[self.convolution_step_edges_max_pooling(i, windows_size=windows_size[iW],
                                                                    weights=weights[iW],
                                                                    weight_shape=weight_shape[
                                                                        iW]).flatten().tolist()
                            for iW in range(len(windows_size))]))
                 for i in data_set])
            # try:
            #     data_out = data_out.reshape(data_out.shape[0:2])
            # except:
            #     pass

            return data_out

        self.features_train = generate_data('train')
        self.features_test = generate_data('test')
        self.features_eval = generate_data('eval')

    def train_and_test(self,
                       windows_size: list,
                       weights=np.array([]),
                       weight_shape: list = []):

        assert len(weights.shape) == 2, 'weights need to be a 2d-Array'
        assert type(weight_shape) == list, 'weight_shape need to be a list'

        # reset to integer
        windows_size = [tuple([int(round(i)) for i in iW]) for iW in windows_size]

        t1 = time.time()
        self.generate_test_train_eval_data(windows_size=windows_size, weights=weights, weight_shape=weight_shape)
        t2 = time.time()
        if self.verbose_debug:
            print('generating data took : ' + str(t2 - t1) + 's')
        print('windows_size: ' + str(windows_size) + '- weights: ', weights)

        t1 = time.time()
        self.create_GPR_model(
            self.features_train,
            self.label_array_train
        )
        t2 = time.time()
        if self.verbose_debug:
            print('generating GPR model took : ' + str(t2 - t1) + 's')

        t1 = time.time()
        pred_vec = self.predict(np.array(self.features_test))
        predicted_output = [pred_vec[0][i].argmax()
                            for i in range(len(self.features_test))]
        t2 = time.time()
        if self.verbose_debug:
            print('test prediction took : ' + str(t2 - t1) + 's')

        t1 = time.time()
        test_accuracy = len([predicted_output[i] for i in range(len(predicted_output)) if
                             predicted_output[i] == self.label_array_test[i].argmax()
                             ]) / len(predicted_output)
        t2 = time.time()
        if self.verbose_debug:
            print('test accuracy took : ' + str(t2 - t1) + 's')
        # test_accuracy = len([predicted_output[i] for i in range(len(predicted_output)) if
        #                      predicted_output[i] == (itemgetter(*self.test_indices)(self.label_array))[i].argmax()
        #                      ])/len(predicted_output)
        if self.verbose:
            pred_vec = self.predict(np.array(self.features_train))
            predicted_output = [pred_vec[0][i].argmax()
                                for i in range(len(self.features_train))]
            train_accuracy = len([predicted_output[i] for i in range(len(predicted_output)) if
                                  predicted_output[i] == self.label_array_train[i].argmax()
                                  ]) / len(predicted_output)

            pred_vec = self.predict(np.array(self.features_eval))
            predicted_output = [pred_vec[0][i].argmax()
                                for i in range(len(self.features_eval))]
            eval_accuracy = len([predicted_output[i] for i in range(len(predicted_output)) if
                                 predicted_output[i] == self.label_array_eval[i].argmax()
                                 ]) / len(predicted_output)

            print('Train accuracy: ' + str(train_accuracy) +
                  ' - Test accuracy: ' + str(test_accuracy) +
                  ' - Evaluation Accuracy: ' + str(eval_accuracy))

        return test_accuracy
