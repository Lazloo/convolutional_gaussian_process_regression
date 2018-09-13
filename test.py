import numpy as np
import requests
import time as time
import GPR_Package.GPR_Script as GP

X = np.array([[1,2,3,4,5,6,7,8,9,10]]).transpose()
Y = np.array([1,2.5,2.8,4.1,2.8,2.2,0.8,2.1,3.2,3.9]).transpose()


gp_test = GP.GPR_Class()

t1 = time.time()
gp_test.create_GPR_model(X, Y)
t2 = time.time()
print('GPR Modelling took: '+str(t1-t2)+'s')

gp_test.plot_1D_interpolation()
#
#
#
# X_norm = normalize(X, min_in=min(X), max_in=max(X), min_out= 0, max_out= 1)
# Y_norm = normalize(Y, min_in=min(Y), max_in=max(Y), min_out=0, max_out=1)
# kernel = 1.0 * Matern(length_scale=0.1, length_scale_bounds=(1e-5, 1e5), nu=1.5) + WhiteKernel()
# gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(X_norm, Y_norm)
#
# ##### Interpolation
# X_ = np.linspace(0, 1, 100)
# y_mean, y_cov_norm = gp.predict(X_[:, np.newaxis], return_cov=True)
#
# ####
# plt.figure()
# plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
# plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov_norm)),
#                  y_mean + np.sqrt(np.diag(y_cov_norm)),
#                  alpha=0.5, color='k')
# plt.scatter(X_norm[:, 0], Y_norm, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
# plt.tight_layout()
#
# #### Rescale
# X_interpolation = normalize(np.linspace(0, 1, 100), min_in=0, max_in=1, min_out=min(X), max_out=max(X))
# y_mean_interpol = y_mean
# y_mean_interpol = normalize(y_mean, min_in=0, max_in=1, min_out=min(Y), max_out=max(Y))
# y_cov = y_cov_norm*((max(Y)-min(Y))**2)
#
# plt.figure()
# plt.plot(X_interpolation, y_mean_interpol, 'k', lw=3, zorder=9)
# plt.fill_between(X_interpolation, y_mean_interpol - np.sqrt(np.diag(y_cov)),
#                  y_mean_interpol + np.sqrt(np.diag(y_cov)),
#                  alpha=0.5, color='k')
# plt.scatter(X[:, 0], Y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
# plt.tight_layout()
# plt.show()
#
# print('stop')
# #
# #
# # # First run
# # plt.figure(0)
# # kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
# #     + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
# # gp = GaussianProcessRegressor(kernel=kernel,
# #                               alpha=0.0).fit(X, y)
# # X_ = np.linspace(0, 5, 100)
# # y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
# # plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
# # plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
# #                  y_mean + np.sqrt(np.diag(y_cov)),
# #                  alpha=0.5, color='k')
# # plt.plot(X_, 0.5*np.sin(3*X_), 'r', lw=3, zorder=9)
# # plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
# # plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
# #           % (kernel, gp.kernel_,
# #              gp.log_marginal_likelihood(gp.kernel_.theta)))
# # plt.tight_layout()
