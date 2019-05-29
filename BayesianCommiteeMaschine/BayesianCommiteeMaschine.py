from loguru import logger
from functools import reduce
import math
from GPR_Package import GPR_Script
from sklearn.cluster import KMeans
import numpy as np


# We will use the simplest form of GP model, exact inference
class BayesianCommiteeMaschine:
    cluster_label = KMeans()
    n_cluster = 1
    x_train = np.ndarray
    y_train = np.ndarray
    x_center = np.ndarray
    y_center = np.ndarray
    gp_clusters = [GPR_Script.GPR_Class()] * 1
    gp_center = GPR_Script.GPR_Class()

    def __init__(self):
        pass

    def generate_cluster(self, x: np.ndarray, y: np.ndarray, n_cluster: int = 2):
        def generate_correct_format_train_data(type_var: str = 'input'):
            if type_var == 'input':
                index = 0
            elif type_var == 'output':
                index = 1
            else:
                index = None
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
                dim = None
                AssertionError('Unknown type_var: ' + str(type_var))

            # Shape will be [n_data, n_dim, 1]
            proto = np.array([
                [dim[i_cluster][i_index] for i_index in rand_index[i_cluster]]
                for i_cluster in range(self.n_cluster)
            ])

            if len(proto.shape) > 2:
                final = np.squeeze(proto, 2)
            else:
                final = proto

            return final

        assert x.shape[0] > x.shape[1], 'More dimension(columns) then data(rows)'
        n_data_points = x.shape[0]

        self.n_cluster = n_cluster

        self.cluster_label = KMeans(n_clusters=n_cluster, random_state=0).fit(x).labels_
        cluster_list = [
            np.concatenate([(x[i, :], y[i]) for i in range(n_data_points) if self.cluster_label[i] == i_cluster],
                           axis=1).transpose()
            for i_cluster in range(n_cluster)
        ]
        self.x_train = generate_correct_format_train_data(type_var='input')
        self.y_train = generate_correct_format_train_data(type_var='output')

        n_data = [len(i_data) for i_data in self.x_train]
        n_center_points_each_cluster = int(math.ceil(min(n_data) / 10))
        rand_index = [
            np.random.random_integers(0, i_data_points - 1, n_center_points_each_cluster) for i_data_points in n_data
        ]

        # self.x_center = define_center_points(type_var='input')
        self.x_center = self.x_train[0::3][0]
        # self.y_center = define_center_points(type_var='output')
        self.y_center = self.y_train[0::3][0]

        # print('stop')

    def create_bcm(self):

        # Generate objects
        self.gp_clusters = [GPR_Script.GPR_Class()] * self.n_cluster

        # Train cluster (experts)
        [
            self.gp_clusters[i_cluster].create_GPR_model(
                np.concatenate((self.x_train[i_cluster], self.x_center)),
                np.concatenate((self.y_train[i_cluster], self.y_center)))
            for i_cluster in range(self.n_cluster)
        ]
        self.gp_center.create_GPR_model(self.x_center, self.y_center)
        print('stop')

    def prediction(self, x_test: np.ndarray):
        n_test_points = x_test.shape[0]

        expert_prediction = [self.gp_clusters[i_cluster].predict(x_test)
                             for i_cluster in range(self.n_cluster)]
        y_mean_list = [expert_prediction[i_cluster][0] for i_cluster in range(self.n_cluster)]
        y_std_list = [expert_prediction[i_cluster][1] for i_cluster in range(self.n_cluster)]

        y_mean_center, y_std_center = self.gp_center.predict(x_test)
        # Calculate betas
        beta_1 = [np.ones(n_test_points)]
        # expert_prediction[i_cluster][1] ... std of expert prediction i_cluster
        beta_2 = [
            0.5 * (np.log(y_std_center) - np.log(y_std_list[i_cluster]))
            for i_cluster in range(1, self.n_cluster)
        ]
        beta = beta_1 + beta_2
        integral_beta = reduce(
            (lambda x, y: x + y),
            [np.subtract(beta[i], 1) for i in range(self.n_cluster)]) / y_std_center ** 2

        # Final Prediction
        y_std_total = np.sqrt(1 /
                              (reduce((lambda x, y: x + y),
                                      [beta[i] * 1 / y_std_list[i] ** 2 for i in range(self.n_cluster)]) -
                               integral_beta)
                              )
        y_mean_total = (
                (y_std_total ** 2) *
                (
                        reduce((lambda x, y: x + y),
                               [
                                   beta[i] * y_mean_list[i][:, 0] / (y_std_list[i] ** 2)
                                   for i in range(self.n_cluster)
                               ]
                               ) -
                        integral_beta * y_mean_center.transpose()
                )).transpose()

        y_std_total = np.expand_dims(y_std_total, axis=1)
        return y_mean_total, y_std_total
