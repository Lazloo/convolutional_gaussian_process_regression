import GPR_Package.GPR_Script as GPR_Script
# import torch
import operator as op
from scipy.stats import norm
import pickle
import logging
import numpy as np
import functools
from tqdm import tqdm
import time as time
import sys

class Bayesian_Optimization(GPR_Script.GPR_Class):
    x_list = []
    y_list = []
    n_random_for_optimization = 1E5
    exploration_factor = 1

    def __init__(self, objective_function, lower_bound: list = [], upper_bound: list = []):
        super().__init__()
        self.logger = logging.getLogger('Bayesian Optimization')
        self.logger.setLevel(logging.INFO)

        assert len(lower_bound) == len(upper_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.objective_function = objective_function

    def calculated_expected_improvement(self, x, y_max):
        n_samples = len(self.y_list)
        n_test_points = len(x)
        n_var = x.shape[1]
        # n_iterations = round(n_samples*n_test_points/(1E5*10))
        n_iterations = round(n_var*n_samples*n_test_points/1E4/10)

        mean_y_iter = [None]*n_iterations
        std_y_iter = [None]*n_iterations
        for i in range(n_iterations):
            x_iter = x[round(i*n_test_points/n_iterations):round((i+1)*n_test_points/n_iterations)]
            mean_y_iter[i], std_y_iter[i] = self.gp.predict(x_iter, return_std=True)
        # mean_y = functools.reduce(op.add, mean_y_iter)
        mean_y = np.concatenate(mean_y_iter)
        std_y = np.concatenate(std_y_iter)
        # std_y = functools.reduce(op.add, std_y_iter)

        z = (mean_y - y_max) / std_y

        t_k = [0]*(self.exploration_factor+1)
        t_k[0] = norm.cdf(z)
        t_k[1] = -norm.pdf(z)

        for i in range(2, self.exploration_factor+1):
            t_k[i] = -np.power(z, i-1)*t_k[1] + (i-1)*t_k[i-2]

        expected_improvement_vec = [
            (-1)**i*self.ncr(self.exploration_factor, i)*np.power(z, self.exploration_factor - i)*t_k[i]
            for i in range(self.exploration_factor+1)]
        expected_improvement = (std_y**self.exploration_factor) * np.sum(expected_improvement_vec, axis=0)

        # ToDo: Validation
        # expected_improvement = (mean_y - y_max) * norm.cdf(z) + std_y * norm.pdf(z)
        return expected_improvement

    def random_choice_max_ei(self):
        y_max = max(self.gp.y_train_)
        random_number = np.random.uniform(self.lower_bound, self.upper_bound,
                                          [int(self.n_random_for_optimization), len(self.upper_bound)])

        expected_improvement = self.calculated_expected_improvement(x=random_number, y_max=y_max)
        idx_opt = np.argmax(expected_improvement)
        para_opt = random_number[idx_opt]

        return para_opt

    def do_optimization(self, n_random: int, n_iteration: int, continue_flag=False):

        def output_print(i, t_delta, y, x):
            self.logger.info('Iteration: {} - Time: {} - Output: {} - Parameter: {}'.format(i, t_delta, y, x))

        self.x_list = [None] * (n_random + n_iteration)
        self.y_list = [None] * (n_random + n_iteration)

        if continue_flag:
            self.load_model('intermediate')
            x_list = [i for i in self.x_list if i is not None]
            n_entries = len(x_list)
            if n_entries >= n_random:
                i_start_rand = n_random
                i_start_iter = n_entries - n_random
            else:
                i_start_rand = n_entries
                i_start_iter = 0
        else:
            i_start_rand = 0
            i_start_iter = 0
        random_number = np.random.uniform(self.lower_bound, self.upper_bound, [n_random, len(self.upper_bound)])

        self.logger.info('Start: Random Initialization')
        for i_random_number in range(i_start_rand, n_random):
            t1 = time.time()
            self.x_list[i_random_number] = random_number[i_random_number]
            # try:
            self.y_list[i_random_number] = self.objective_function(random_number[i_random_number])
            # except:
            #     self.logger.error(sys.exc_info()[1])
            #     self.y_list[i_random_number] = 0
            # torch.cuda.empty_cache()
            t2 = time.time()
            t_delta = t2 - t1
            output_print(i_random_number, t_delta, self.y_list[i_random_number], self.x_list[i_random_number])
            self.save_model('intermediate')

        self.logger.info('Start: Sequential Iteration')
        for i_iteration in range(i_start_iter, n_iteration):
            t1 = time.time()

            self.create_GPR_model(self.x_list[0:(n_random + i_iteration)], self.y_list[0:(n_random + i_iteration)])
            x = self.random_choice_max_ei()
            y = self.objective_function(x)
            # torch.cuda.empty_cache()
            
            self.x_list[n_random + i_iteration] = x
            self.y_list[n_random + i_iteration] = y

            t2 = time.time()
            t_delta = t2 - t1

            output_print(i_iteration, t_delta, y, x)
            self.save_model('intermediate')

    def save_model(self, file_template: str = 'bo'):
        with open(file_template + '_x_list.pickle', 'wb') as file_pi:
            pickle.dump(obj=self.x_list, file=file_pi)

        with open(file_template + '_y_list.pickle', 'wb') as file_pi:
            pickle.dump(obj=self.y_list, file=file_pi)

        with open(file_template + '_gp.pickle', 'wb') as file_pi:
            pickle.dump(obj=self.gp, file=file_pi)

    def load_model(self, file_template: str = 'bo'):
        with open(file_template + '_x_list.pickle', 'rb') as file_pi:
            self.x_list = pickle.load(file=file_pi)

        with open(file_template + '_y_list.pickle', 'rb') as file_pi:
            self.y_list = pickle.load(file=file_pi)

        with open(file_template + '_gp.pickle', 'rb') as file_pi:
            self.gp = pickle.load(file=file_pi)

    @staticmethod
    def ncr(n, r):
        r = min(r, n-r)
        numer = functools.reduce(op.mul, range(n, n-r, -1), 1)
        denom = functools.reduce(op.mul, range(1, r+1), 1)
        return numer//denom
