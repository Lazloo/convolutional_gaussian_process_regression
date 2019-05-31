from sklearn.feature_extraction import image
import logging
import time as time
import itertools
import numpy as np
from operator import itemgetter
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy as scipy
import skimage.measure as measure
from typing import List
from PIL import Image
import os
from BayesianCommiteeMaschine import BayesianCommiteeMaschine


class toy_example:

    def __init__(self):

        self.logger = logging.getLogger('Toy Example')
        self.verbose = False
        self.verbose_debug = False

        self.gp = BayesianCommiteeMaschine.BayesianCommiteeMaschine()

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

        # Number of inducing point for gyptorch SGPR
        self.n_inducing_points = 100

    @staticmethod
    def normalize(x: List[float], min_in: float, max_in: float, min_out: float, max_out: float) -> list:
        x_norm = ((x - min_in) / (max_in - min_in) * (max_out - min_out) + min_out)
        return x_norm

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

            return data_out

        self.features_train = generate_data('train')
        self.features_test = generate_data('test')
        self.features_eval = generate_data('eval')

    def train_and_test(self,
                       windows_size: list,
                       weights=np.array([]),
                       weight_shape: list = [],
                       n_cluster: int = 10):

        assert len(weights.shape) == 2, 'weights need to be a 2d-Array'
        assert type(weight_shape) == list, 'weight_shape need to be a list'

        # reset to integer
        windows_size = [tuple([int(round(i)) for i in iW]) for iW in windows_size]

        t1 = time.time()
        self.generate_test_train_eval_data(windows_size=windows_size, weights=weights, weight_shape=weight_shape)
        t2 = time.time()
        if self.verbose_debug:
            self.logger.info('generating data took : ' + str(t2 - t1) + 's')
        # print('windows_size: ' + str(windows_size) + '- weights: ', weights)

        t1 = time.time()
        self.gp.generate_cluster(
            x=self.features_train,
            y=self.label_array_train,
            n_cluster=n_cluster
        )
        t2 = time.time()
        if self.verbose_debug:
            self.logger.info('generating cluster took : ' + str(t2 - t1) + 's')

        t1 = time.time()
        self.gp.create_bcm()
        t2 = time.time()
        if self.verbose_debug:
            self.logger.info('generating GPR model took : ' + str(t2 - t1) + 's')

        t1 = time.time()
        pred_vec = self.gp.prediction(np.array(self.features_test))
        predicted_output = [pred_vec[0][i].argmax()
                            for i in range(len(self.features_test))]
        t2 = time.time()
        if self.verbose_debug:
            self.logger.info('test prediction took : ' + str(t2 - t1) + 's')

        t1 = time.time()
        test_accuracy = len([predicted_output[i] for i in range(len(predicted_output)) if
                             predicted_output[i] == self.label_array_test[i].argmax()
                             ]) / len(predicted_output)
        t2 = time.time()
        if self.verbose_debug:
            self.logger.info('test accuracy took : ' + str(t2 - t1) + 's')
        if self.verbose:
            pred_vec = self.gp.prediction(np.array(self.features_train))
            predicted_output = [pred_vec[0][i].argmax()
                                for i in range(len(self.features_train))]
            train_accuracy = len([predicted_output[i] for i in range(len(predicted_output)) if
                                  predicted_output[i] == self.label_array_train[i].argmax()
                                  ]) / len(predicted_output)

            pred_vec = self.gp.prediction(np.array(self.features_eval))
            predicted_output = [pred_vec[0][i].argmax()
                                for i in range(len(self.features_eval))]
            eval_accuracy = len([predicted_output[i] for i in range(len(predicted_output)) if
                                 predicted_output[i] == self.label_array_eval[i].argmax()
                                 ]) / len(predicted_output)

            self.logger.info('Train accuracy: ' + str(train_accuracy) +
                  ' - Test accuracy: ' + str(test_accuracy) +
                  ' - Evaluation Accuracy: ' + str(eval_accuracy))

        return test_accuracy
