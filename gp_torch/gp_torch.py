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
class gp_torch(gpytorch.models.AbstractVariationalGP):
    # class gp_torch(gpytorch.models.ExactGP):

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
        super(gp_torch, self).__init__(variational_strategy)
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

        # grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
        # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
        #     gpytorch.kernels.MaternKernel(nu=1.5)
        #     , grid_size=max(grid_size,100), num_dims=train_x.shape[-1],
        # )

        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
        if grid_size <= 1:
            logger.warning(
                'Too less measurements for grid calculation: need at least: ' + str(2 ** num_dims) + 'but got: '
                + str(train_x.shape[-2]))

        self.covar_module = base_kernel
        # grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
        # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
        #     base_kernel
        #     , grid_size=grid_size, num_dims=num_dims,
        # )

        self.verbose = verbose

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def train_gp_model(self, train_x, train_y):
        # Use full-batch L-BFGS optimizer
        # optimizer = FullBatchLBFGS(self.parameters(), lr=1)
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=int(max(min(train_y.size(0) / 100, 1E4), 100)), shuffle=True)
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
                        'Iter %d[%d/%d] - Loss: %.3f'
                        % (i + 1, minibatch_i, len(train_loader), loss.item())
                    )

    def train_gp_model_LBFGS_SGP(self, train_x, train_y):
        # Use full-batch L-BFGS optimizer
        # optimizer = FullBatchLBFGS(self.parameters(), lr=1)
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=int(max(min(train_y.size(0) / 100, 1E4), 100)), shuffle=True)
        self.train()
        self.likelihood.train()

        optimizer = FullBatchLBFGS(self.parameters(), lr=1E-1)
        # Access aprameters: self.likelihood.noise_covar.raw_noise
        # self.likelihood.noise_covar.initialize(raw_noise=0.)
        # self.mean_module.initialize(constant=0.)
        # para_to_check = self.covar_module.base_kernel
        # para_to_check.initialize(raw_lengthscale=0.)
        # self.covar_module.initialize(raw_lengthscale=0.)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self, num_data=train_y.size(0), combine_terms=False).cuda()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 5], gamma=0.1)

        # define closure

        training_iter = 20
        num_epochs = 2
        for i in range(num_epochs):
            scheduler.step()
            for minibatch_i, (x_batch, y_batch) in enumerate(train_loader):
                # define closure
                def closure():
                    optimizer.zero_grad()
                    # output = self(train_x.double())
                    # loss = -mll(output, train_y.double()).sum()
                    with gpytorch.settings.use_toeplitz(False):
                        output = self(x_batch.double())
                        log_lik, kl_div, log_prior = mll(output, y_batch.double())
                        loss = -(log_lik - kl_div + log_prior).sum()

                    return loss

                loss = closure()
                loss.backward()
                # perform step and update curvature
                # optimizer.zero_grad()
                options = {'closure': closure, 'current_loss': loss, 'max_ls': 1, 'eta': 2}
                loss, g_new, lr, _, F_eval, G_eval, desc_dir, fail = optimizer.step(options)

                if self.verbose:
                    logger.info(
                        'Iter %d[%d/%d] - Loss: %.3f - LR: %.3f - Func Evals: %0.0f - Grad Evals: %0.0f - fail: %0.0f'
                        % (i + 1, minibatch_i, len(train_loader), loss.item(), lr, F_eval, G_eval, fail)
                    )
                    # logger.info(str(g_new))
                # if torch.isnan(para_to_check.raw_lengthscale.data):
                #     # logger.warning('NaN detected')
                #     # self.covar_module.initialize(raw_lengthscale=1E-6)
                #     para_to_check.initialize(raw_lengthscale=1E-6)

    def train_gp_model_LBFGS(self, train_x, train_y):
        # Use full-batch L-BFGS optimizer
        # optimizer = FullBatchLBFGS(self.parameters(), lr=1)
        optimizer = FullBatchLBFGS(self.parameters(), lr=1E-1)
        # Access aprameters: self.likelihood.noise_covar.raw_noise
        self.likelihood.noise_covar.initialize(raw_noise=0.)
        self.mean_module.initialize(constant=0.)
        para_to_check = self.covar_module.base_kernel
        para_to_check.initialize(raw_lengthscale=0.)
        # self.covar_module.initialize(raw_lengthscale=0.)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        # define closure
        # define closure
        def closure():
            optimizer.zero_grad()
            output = self(train_x.double())
            loss = -mll(output, train_y.double()).sum()
            return loss

        loss = closure()
        loss.backward()

        training_iter = 20
        for i in range(training_iter):

            # perform step and update curvature
            # optimizer.zero_grad()
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 20, 'eta': 2}
            loss, g_new, lr, _, F_eval, G_eval, desc_dir, fail = optimizer.step(options)
            if self.verbose:
                logger.info(
                    'Iter %d/%d - Loss: %.3f - LR: %.3f - Func Evals: %0.0f - Grad Evals: %0.0f - fail: %0.0f' % (
                        i + 1, training_iter, loss.item(), lr, F_eval, G_eval, fail
                    ))
                # logger.info(str(g_new))
            if torch.isnan(para_to_check.raw_lengthscale.data):
                # logger.warning('NaN detected')
                # self.covar_module.initialize(raw_lengthscale=1E-6)
                para_to_check.initialize(raw_lengthscale=1E-6)

    def train_gp_model_2(self, train_x, train_y):
        # Find optimal model hyperparameters
        self.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.0001)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        n_iter = 50
        for i in range(n_iter):
            # optimizer.zero_grad()
            output = self(train_x.double())
            # loss = -mll(output, train_y)
            loss = -mll(output, train_y).sum()
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
            optimizer.step()

    def train_gp_model_test(self, train_x, train_y):
        # Switch to train mode
        self.train()
        self.likelihood.train()

        # random.seed(opt.manualSeed)
        # torch.manual_seed(opt.manualSeed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(opt.manualSeed)

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        n_trials = 50
        for i_trial in tqdm(range(n_trials)):
            new_state_dict = OrderedDict({'covar_module.task_covar_module.raw_var':
                                              (torch.rand((1, len(self.train_targets[0]))) - 0.5) * 2
                                          })
            self.load_state_dict(new_state_dict, strict=False)

            training_iterations = 10
            for i in range(training_iterations):
                # print(list(self.named_parameters()))
                optimizer.zero_grad()
                output = self(train_x)
                # loss = -mll(output, train_y)
                loss = -mll(output, train_y).sum()

                c_min = []
                c_max = []
                for i_para in list(self.parameters()):
                    i_test = i_para.data.cpu().detach().numpy()[0]
                    if type(i_test) == np.ndarray:
                        if type(i_test[0]) == np.ndarray:
                            i_test = i_test[0]
                        c_min.append(min(i_test))
                        c_max.append(max(i_test))
                    else:
                        c_min.append(i_test)
                        c_max.append(i_test)

                check_constants = np.array(c_min + c_max)

                check_constrain_mean = len([value_para for value_para in check_constants if abs(value_para) > 1]) != 0
                if check_constrain_mean:
                    loss = loss / (10 * check_constants.max())
                # loss = (output - train_y) ** 2
                loss.backward()
                # if self.verbose:
                #     logger.info('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
                # print(list(self.parameters()))
                optimizer.step()

            if i_trial == 0:
                para = self.state_dict()
                current_best_loss = loss.item()
            else:
                if current_best_loss > loss.item():
                    para = self.state_dict()
                    current_best_loss = loss.item()
            if self.verbose:
                logger.info('MLL after optimization: %.3f, best so far: %.3f' % (loss.item(), current_best_loss))

        logger.info('MLL after optimization: %.3f, best so far: %.3f' % (loss.item(), current_best_loss))
        self.load_state_dict(para, strict=False)

        # def train():
        #     for i in range(training_iterations):
        #         # Zero backprop gradients
        #         optimizer.zero_grad()
        #         # Get output from model
        #         output = self(train_x)
        #         # Calc loss and backprop derivatives
        #         loss = -mll(output, train_y)
        #         loss.backward()
        #         if self.verbose:
        #             logger.info('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        #         optimizer.step()
        #         torch.cuda.empty_cache()

        # # See dkl_mnist.ipynb for explanation of this flag
        # with gpytorch.settings.use_toeplitz(True):
        #     train()

        # training_iter = 50
        # for i in tqdm(range(training_iter)):
        #     # Zero gradients from previous iteration
        #     optimizer.zero_grad()
        #     # Output from model
        #     output = self(train_x)
        #     # Calc loss and backprop gradients
        #     loss = -mll(output, train_y)
        #     loss.backward()
        #     if self.verbose:
        #         logger.info('Iter %d/%d - Loss: %.3f' % (
        #             i + 1, training_iter, loss.item(),
        #         ))
        #     optimizer.step()

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
