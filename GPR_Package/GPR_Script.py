from matplotlib import pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from typing import List


class GPR_Class:

    def __init__(self):
        self.data = []

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

