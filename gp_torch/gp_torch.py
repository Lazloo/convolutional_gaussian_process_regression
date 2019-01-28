from loguru import logger
from LBFGS import FullBatchLBFGS
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
class gp_torch(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, verbose=False, kernel='Matern_2_5'):
        super(gp_torch, self).__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        if len(train_y.shape) == 1:
            num_tasks = 1
        else:
            num_tasks = train_y.shape[1]

        if len(train_x.shape) == 1:
            num_dims = 1
        else:
            num_dims = train_x.shape[1]

        # self.mean_module = gpytorch.means.MultitaskMean(
        #     gpytorch.means.ConstantMean(), num_tasks=num_tasks
        # )
        # self.covar_module = gpytorch.kernels.MultitaskKernel(
        #     gpytorch.kernels.MaternKernel(nu=2.5), num_tasks=num_tasks, rank=1
        # )

        # SKI requires a grid size hyperparameter. This util can help with that
        grid_size = gpytorch.utils.grid.choose_grid_size(train_x.transpose(dim0=0, dim1=1))
        grid_size = len(train_x)

        # self.mean_module = gpytorch.means.MultitaskMean(
        #     gpytorch.means.ConstantMean(), num_tasks=num_tasks
        # )
        self.mean_module = gpytorch.means.ConstantMean()

        # wn_variances = torch.randn((train_x.shape[0],train_x.shape[1],1))
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

        self.covar_module = base_kernel
        # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
        #     base_kernel
        #     , grid_size=grid_size, num_dims=num_dims,
        # )
        # gpytorch.kernels.MultitaskKernel(
        # , num_tasks=num_tasks, rank=1
        # )
        self.verbose = verbose

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        # return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def train_gp_model(self, train_x, train_y):
        # Use full-batch L-BFGS optimizer
        optimizer = FullBatchLBFGS(self.parameters())

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        # define closure
        # define closure
        def closure():
            optimizer.zero_grad()
            output = self(train_x)
            loss = -mll(output, train_y).sum()
            return loss

        loss = closure()
        loss.backward()

        training_iter = 50
        for i in range(training_iter):

            # perform step and update curvature
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
            if self.verbose:
                logger.info(
                    'Iter %d/%d - Loss: %.3f - LR: %.3f - Func Evals: %0.0f - Grad Evals: %0.0f' % (
                        i + 1, training_iter, loss.item(), lr, F_eval, G_eval,
                    ))

    def train_gp_model_std(self, train_x, train_y):
        # Use the adam optimizer
        optimizer = torch.optim.Adagrad([
            {'params': self.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.2)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        n_iter = 50
        for i in range(n_iter):
            optimizer.zero_grad()
            output = self(train_x)
            # loss = -mll(output, train_y)
            loss = -mll(output, train_y).sum()
            loss.backward()
            if self.verbose:
                logger.info('Iter %d - Loss: %.3f' % (i + 1, loss.item()))
            # print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
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
