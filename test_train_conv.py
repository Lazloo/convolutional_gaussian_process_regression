import GPR_Package.GPR_Script as GP
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import matplotlib.cm as cm
import pickle
import time as time
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter
from bayes_opt import BayesianOptimization

dir = 'D:/Programmieren/image_data/test_short'
# dir = 'C:/Users/Lazloo/Programmieren/image_data/short'

label_choices = '01234567'
# label_choices = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

gp_test = GP.GPR_Class()
gp_test.verbose_debug = False
# gp_test.gp_package = 'sklearn' #'sklearn' 'gp_flow'
gp_test.gp_package = 'sklearn'  # 'sklearn' 'gp_flow'

windows_size = (10, 5)

gp_test.load_data(dir_name=dir, label_choices=label_choices, ratio_train_test_eval=(0.001, 0.1, 0.1))
print('start')
# gp_test.train_and_test(windows_size=(2, 20), weights=np.array([1, 1, -1, -1]), weight_shape=(2, 2))
gp_test.train_and_test(windows_size=[(15, 12)], weights=np.array([[1, 1, -1, -1]]), weight_shape=[(2, 2)])

lower_bounds = (10, 5, -5)
upper_bounds = (20, 15, 5)
# bo = BayesianOptimization(lambda x, y: gp_test.train_and_test(windows_size=[(x, y)],
#                                                               weights=np.array([[1, 1, -1, -1]]),
#                                                               weight_shape=[(2, 2)]),
#                           {'x': (lower_bounds[0], upper_bounds[0]), 'y': (lower_bounds[1], upper_bounds[1])})
bo = BayesianOptimization(lambda x1, y1, x2, y2, a1, b1, c1, d1, a2, b2, c2, d2:
                          gp_test.train_and_test(windows_size=[(x1, y1), (x2, y2)],
                                                 weights=np.array([[a1, b1, c1, d1], [a2, b2, c2, d2]]),
                                                 weight_shape=[(2, 2), (2, 2)]),
                          {
                              'x1': (lower_bounds[0], upper_bounds[0]), 'y1': (lower_bounds[1], upper_bounds[1]),
                              'x2': (lower_bounds[0], upper_bounds[0]), 'y2': (lower_bounds[1], upper_bounds[1]),
                              'a1': (lower_bounds[2], upper_bounds[2]), 'b1': (lower_bounds[2], upper_bounds[2]),
                              'c1': (lower_bounds[2], upper_bounds[2]), 'd1': (lower_bounds[2], upper_bounds[2]),
                              'a2': (lower_bounds[2], upper_bounds[2]), 'b2': (lower_bounds[2], upper_bounds[2]),
                              'c2': (lower_bounds[2], upper_bounds[2]), 'd2': (lower_bounds[2], upper_bounds[2])
                          })

kernel = 1.0 * Matern(nu=2.5) + WhiteKernel()
gp_params = {'kernel': kernel}
bo.maximize(init_points=25, n_iter=400, acq='ei')
file_name = 'gp_model_2.obj'
# with open(file_name, 'rb') as file_pi:
#      bo.gp = pickle.load(file=file_pi)
#
# Save model
with open('gp_model_two_windows.obj', 'wb') as file_pi:
    pickle.dump(obj=bo.gp, file=file_pi)

gp_validation = GP.GPR_Class()
gp_validation.create_GPR_model(bo.gp.X_train_, bo.gp.y_train_)
import numpy as np

n_points = 100
n_0 = np.random.uniform(lower_bounds[0], upper_bounds[0], size=1000)
n_1 = np.random.uniform(lower_bounds[1], upper_bounds[1], size=1000)
x = np.arange(lower_bounds[0], upper_bounds[0], (upper_bounds[0] - lower_bounds[0]) / (n_points - 1))
y = np.arange(lower_bounds[1], upper_bounds[1], (upper_bounds[1] - lower_bounds[1]) / (n_points - 1))
n_diff_points = x.shape[0]
X, Y = np.meshgrid(x, y)
n = np.vstack([X.reshape(n_diff_points ** 2, 1), Y.reshape(n_diff_points ** 2, 1)]).reshape(n_diff_points ** 2, 2)
out_pred = gp_validation.predict(n)[0]
out_pred_square = out_pred.reshape(n_diff_points, n_diff_points)

# Or you can use a colormap to specify the colors; the default
# colormap will be used for the contour lines
plt.figure()
im = plt.imshow(out_pred_square, interpolation='bilinear', origin='lower',
                cmap=cm.gray, extent=(X.min(), X.max(), Y.min(), Y.max()))
levels = np.arange(-1.2, 1.6, 0.2)
CS = plt.contour(X, Y, out_pred_square,
                 origin='lower')
# linewidths=2,
# extent=(X.min(), X.max(), Y.min(), Y.max()))
# make a colorbar for the contour lines
# CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.clabel(CS, inline=1, fontsize=14, fmt='%1.1f')
plt.title('Lines with colorbar')

gp_test.train_and_test(windows_size=[(15, 12)], weights=np.array([[1, 1, -1, -1]]), weight_shape=[(2, 2)])
bo.gp.predict(np.array([15, 12])[np.newaxis, :])
#
# bo.gp.X_train_

# plt.plot(bo.gp.X_train_)

x_array = [i[0] for i in bo.gp.X_train_]
y_array = [i[1] for i in bo.gp.X_train_]

plt.figure()
plt.scatter(x_array, y_array, c=bo.gp.y_train_, cmap='Greys')
plt.colorbar()
plt.show()

# plt.figure()
# plt.scatter(X.reshape(n_diff_points**2, 1), Y.reshape(n_diff_points**2, 1), c=out_pred[:, np.newaxis])
# plt.show()


# with open('gp_xmin.obj', 'wb') as file_pi:
#     pickle.dump(obj=gp_test.x_min, file=file_pi)
# with open('gp_xmax.obj', 'wb') as file_pi:
#     pickle.dump(obj=gp_test.x_max, file=file_pi)
# with open('gp_ymin.obj', 'wb') as file_pi:
#     pickle.dump(obj=gp_test.y_min, file=file_pi)
# with open('gp_ymax.obj', 'wb') as file_pi:
#     pickle.dump(obj=gp_test.y_min, file=file_pi)
