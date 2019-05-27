from loguru import logger
from LBFGS import FullBatchLBFGS
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
class gp_torch_svgp(gpytorch.models.AbstractVariationalGP):

    def __init__(self, train_x, train_y, likelihood, verbose=False, kernel='Matern_2_5'):
        # super(gp_torch, self).__init__(train_x, train_y, likelihood)

        if len(train_y.shape) == 1:
            num_tasks = 1
        else:
            num_tasks = train_y.shape[1]

        if len(train_x.shape) == 1:
            num_dims = 1
        else:
            num_dims = train_x.shape[-1]

        grid_size = 32
        grid_bounds = [(-1, 1)] * num_dims
        grid_size = int(max(min(1E10**(1/num_dims), 32), 3))
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=
                                                                                        grid_size ** 2)
        variational_strategy = gpytorch.variational.GridInterpolationVariationalStrategy(self,
                                                                                         grid_size=grid_size,
                                                                                         grid_bounds=grid_bounds,
                                                                                         variational_distribution=
                                                                                         variational_distribution)
        # super(GPRegressionLayer, self).__init__(variational_strategy)
        super(gp_torch_svgp, self).__init__(variational_strategy)
        self.likelihood = likelihood

        if kernel == 'Matern_0_5':
            base_kernel = gpytorch.kernels.MaternKernel(nu=0.5)
        elif kernel == 'Matern_1_5':
            # base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=num_dims)
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
        self.covar_module = base_kernel

        self.verbose = verbose

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def train_gp_model(self, train_x, train_y):
        # Use full-batch L-BFGS optimizer
        # optimizer = FullBatchLBFGS(self.parameters(), lr=1)
        train_dataset = TensorDataset(train_x, train_y)
        n_batch = int(max(min(train_y.size(0) / 100, 1E4), 100))
        train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True)
        self.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=train_y.size(0), combine_terms=False).cuda()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 5], gamma=0.1)

        # define closure

        training_iter = 20
        num_epochs = 2
        for i in range(num_epochs):
            scheduler.step()
            for minibatch_i, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                with gpytorch.settings.use_toeplitz(False):
                    output = self(x_batch.double())
                    log_lik, kl_div, log_prior = mll(output, y_batch.double())
                    loss = -(log_lik - kl_div + log_prior).sum()

                # The actual optimization step
                loss.backward()
                optimizer.step()

                if self.verbose:
                    logger.info(
                        'Iter %d/%d[%d/%d] - Loss: %.3f'
                        % (i + 1, num_epochs, minibatch_i, len(train_loader), loss.item())
                    )

    def make_prediction(self, test_x):
        # Get into evaluation (predictive posterior) mode
        self.eval()
        self.likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self(test_x))

        mean_array = observed_pred.mean.detach().cpu().numpy().transpose()
        std_array = observed_pred.stddev.detach().cpu().numpy().transpose()

        # with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
        #     with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30), \
        #          gpytorch.settings.fast_pred_var():
        #         observed_pred = self(test_x)
        #         mean_array = observed_pred.mean.numpy()
        #         std_array = observed_pred.stddev.numpy()

        return mean_array, std_array
