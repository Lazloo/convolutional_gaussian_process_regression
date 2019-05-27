from loguru import logger
import math
from random import randint
from GPR_Package import GPR_Script
from sklearn.cluster import KMeans
import numpy as np


# We will use the simplest form of GP model, exact inference
class bcm:
    cluster_label = KMeans()
    x_train = np.ndarray
    y_train = np.ndarray
    x_center = np.ndarray
    y_center = np.ndarray

    def __init__(self):
        pass

    def generate_cluster(self, x: np.ndarray, y: np.ndarray, n_cluster: int = 2):
        def generate_correct_format_train_data(type_var: str = 'input'):
            if type_var == 'input':
                index = 0
            elif type_var == 'output':
                index = 1
            else:
                AssertionError('Unknown type_var: ' + str(type_var))

            proto = [
                # np.expand_dims(
                np.array([i_data[index] for i_data in cluster_list[i_cluster]])
                for i_cluster in range(n_cluster)
            ]

            # Extend by array to nd-array if needed
            if len(proto[0].shape) == 1:
                final = [np.expand_dims(i_data, axis=1) for i_data in proto]
            else:
                final = proto

            return final

        def define_center_points(type_var: str = 'input'):
            if type_var == 'input':
                dim = self.x_train
            elif type_var == 'output':
                dim = self.y_train
            else:
                AssertionError('Unknown type_var: ' + str(type_var))

            return [
                [dim[i_cluster][i_index] for i_index in rand_index[i_cluster]] for i_cluster in range(n_cluster)
            ]

        assert x.shape[0] > x.shape[1], 'More dimension(columns) then data(rows)'
        n_data_points = x.shape[0]

        self.cluster_label = KMeans(n_clusters=n_cluster, random_state=0).fit(x).labels_
        cluster_list = [
            np.concatenate([(x[i, :], y[i]) for i in range(n_data_points) if self.cluster_label[i] == i_cluster],
                           axis=1).transpose()
            for i_cluster in range(n_cluster)
        ]
        self.x_train = generate_correct_format_train_data(type_var='input')
        self.y_train = generate_correct_format_train_data(type_var='output')

        n_data = [len(i_data) for i_data in self.x_train]
        n_center_points_each_cluster = math.ceil(min(n_data) / 10)
        rand_index = [
            np.random.random_integers(0, i_data_points - 1, n_center_points_each_cluster) for i_data_points in n_data
        ]

        self.x_center = define_center_points(type_var='input')
        self.y_center = define_center_points(type_var='output')

        # print('stop')
    # def create_bcm(self):
