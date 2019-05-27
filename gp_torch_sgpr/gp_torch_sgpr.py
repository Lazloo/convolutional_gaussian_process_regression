from loguru import logger
# from LBFGS import FullBatchLBFGS
from torch.utils.data import TensorDataset, DataLoader
import itertools
import operator
from functools import reduce
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import torch
import gpytorch


class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1, 1)


# We will use the simplest form of GP model, exact inference
class gp_torch_sgpr(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, verbose=False, kernel='Matern_2_5', n_inducing_points: int = 0):
        super(gp_torch_sgpr, self).__init__(train_x, train_y, likelihood)

        if len(train_y.shape) == 1:
            num_tasks = 1
        else:
            num_tasks = train_y.shape[1]

        if len(train_x.shape) == 1:
            num_dims = 1
        else:
            num_dims = train_x.shape[-1]

        self.likelihood = likelihood

        if kernel == 'Matern_0_5':
            base_kernel = gpytorch.kernels.MaternKernel(nu=0.5)
        elif kernel == 'Matern_1_5':
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel == 'Matern_2_5':
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        elif kernel == 'Linear':
            base_kernel = gpytorch.kernels.LinearKernel(num_dimensions=num_dims)
        elif kernel == 'RBF':
            base_kernel = gpytorch.kernels.RBFKernel()
        elif kernel == 'Cosine':
            base_kernel = gpytorch.kernels.CosineKernel()
        else:
            raise AssertionError('Unknown kernel type: ' + kernel)

        self.mean_module = gpytorch.means.ConstantMean()

        if n_inducing_points > 0:
            self.covar_module = gpytorch.kernels.InducingPointKernel(base_kernel,
                                                                     inducing_points=train_x[1, :n_inducing_points, :],
                                                                     likelihood=likelihood)
        else:
            self.covar_module = base_kernel

        self.verbose = verbose

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def train_gp_model(self, train_x, train_y):
        # Use full-batch L-BFGS optimizer
        # optimizer = FullBatchLBFGS(self.parameters(), lr=1)
        optimizer = FullBatchLBFGS(self.parameters(), lr=1E-1)
        # Access aprameters: self.likelihood.noise_covar.raw_noise
        self.likelihood.noise_covar.initialize(raw_noise=0.)
        self.mean_module.initialize(constant=0.)
        para_to_check = self.covar_module.base_kernel
        # para_to_check = self.covar_module
        # para_to_check.initialize(raw_lengthscale=0.)
        # # self.covar_module.initialize(raw_lengthscale=0.)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        self.float()

        # define closure
        def closure():
            # optimizer.zero_grad()
            # torch.cuda.empty_cache()
            torch.cuda.empty_cache()

            output = self(train_x.float())
            loss = -mll(output, train_y.float())
            loss_final = loss.sum()

            del loss, output
            return loss_final

        loss = closure()
        loss.backward()

        training_iter = 25
        for i in range(training_iter):

            # perform step and update curvature
            optimizer.zero_grad()
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10, 'eta': 2}
            optimizer.step(options)
            loss, g_new, lr, _, F_eval, G_eval, desc_dir, fail = optimizer.step(options)
            if self.verbose:
                logger.info(
                    'Iter %d/%d - Loss: %.3f - LR: %.3f - Func Evals: %0.0f - Grad Evals: %0.0f - fail: %0.0f' % (
                        i + 1, training_iter, loss.item(), lr, F_eval, G_eval, fail
                    ))
                # logger.info(str(g_new))
            if torch.isnan(para_to_check.raw_lengthscale.data):
                logger.warning('NaN detected')
                # self.covar_module.initialize(raw_lengthscale=1E-6)
                para_to_check.initialize(raw_lengthscale=1E-6)
        print('stop')

        del loss
        del closure
        del optimizer
        del g_new, lr, F_eval, G_eval, desc_dir, fail
        del options
        torch.cuda.empty_cache()

    def train_gp_model_adam(self, train_x, train_y):
        # Find optimal model hyperparameters
        self.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        n_iter = 50
        for i in range(n_iter):
            # optimizer.zero_grad()
            output = self(train_x.float())
            loss = -mll(output, train_y.float()).sum()
            loss.backward()

            if self.verbose:
                logger.info('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
            optimizer.step()

    def train_gp_model_sgd(self, train_x, train_y):
        # Find optimal model hyperparameters
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.SGD(self.parameters(), lr=1.0)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        training_iterations = 20

        for i in range(training_iterations):
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = self(train_x)
            # Calc loss and backprop derivatives
            loss = -mll(output, train_y)
            loss.backward()
            if self.verbose:
                logger.info('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()
            torch.cuda.EMPTY_CACHE()

    def make_prediction(self, test_x):
        # Get into evaluation (predictive posterior) mode
        self.eval()
        self.likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self(test_x.float()))

        mean_array = observed_pred.mean.detach().cpu().numpy().transpose()
        std_array = observed_pred.stddev.detach().cpu().numpy().transpose()

        return mean_array, std_array
