import numpy as np
from bcm import bcm
from GPR_Package import GPR_Script
from matplotlib import pyplot as plt
from functools import reduce
from sklearn.cluster import KMeans


def plot_gp(iFig: int,
            x: np.ndarray,
            y_mean: np.ndarray,
            y_std: np.array,
            x_train: np.ndarray,
            y_train: np.ndarray):
    plt.figure(iFig)
    plt.plot(x, y_mean, 'k', lw=3)
    plt.fill_between(x=x[:, 0], y1=(y_mean[:, 0] - y_std[:, 0]).transpose(),
                     y2=(y_mean[:, 0] + y_std[:, 0]).transpose(),
                     alpha=0.5, color='k')
    plt.plot(x_train, y_train, 'xr')


def fct_qtr(x):
    # return x ** 2
    return np.sin(x)


# Full
x_min = -10
x_max = 10
n_data_points = 16
x_train = np.linspace(x_min, x_max, n_data_points)
# x_train = np.linspace(-2, 2, 7)
x_train = np.expand_dims(x_train, axis=0).transpose()
y_train = fct_qtr(x_train)

obj_bcm = bcm.bcm()
obj_bcm.generate_cluster(x=x_train, y=y_train, n_cluster=2)
# kmeans = KMeans(n_clusters=2, random_state=0).fit(x_train)

# np.array([x_train[i] for i in range(n_data_points) if kmeans.labels_[i] == 0])
obj_gp = GPR_Script.GPR_Class()
obj_gp.create_GPR_model(x_train, y_train)
n_test_points = 1000
x_test = np.linspace(x_min, x_max, n_test_points)
x_test = np.expand_dims(x_test, axis=0).transpose()
y_mean, y_std = obj_gp.predict(x_test)

# Two Parts
x_train_1 = np.expand_dims(x_train[x_train <= 0], axis=0).transpose()
y_train_1 = fct_qtr(x_train_1)
x_train_2 = np.expand_dims(x_train[x_train >= 0], axis=0).transpose()
y_train_2 = fct_qtr(x_train_2)
x_train_center = x_train[0::3]
y_train_center = y_train[0::3]


# Mix center values with remaining values
def set_diff(x, y):
    return np.expand_dims(np.setdiff1d(x, y), axis=0).transpose()


x_train_1 = np.concatenate((x_train_1, x_train_center))
x_train_2 = np.concatenate((x_train_2, x_train_center))
y_train_1 = np.concatenate((y_train_1, y_train_center))
y_train_2 = np.concatenate((y_train_2, y_train_center))

obj_gp_1 = GPR_Script.GPR_Class()
obj_gp_1.create_GPR_model(x_train_1, y_train_1)
obj_gp_2 = GPR_Script.GPR_Class()
obj_gp_2.create_GPR_model(x_train_2, y_train_2)
obj_gp_center = GPR_Script.GPR_Class()
obj_gp_center.create_GPR_model(x_train_center, y_train_center)
y_mean_1, y_std_1 = obj_gp_1.predict(x_test)
y_mean_2, y_std_2 = obj_gp_2.predict(x_test)
y_mean_center, y_std_center = obj_gp_2.predict(x_test)

n_parts = 2
beta = [0.5] * 2
beta = [np.ones(n_test_points), [0.5 * (np.log(y_std_center) + np.log(y_std_2))]]
y_mean_list = [y_mean_1, y_mean_2]
y_std_list = [y_std_1, y_std_2]
integral_beta = reduce((lambda x, y: x + y), [np.subtract(beta[i], 1) for i in range(n_parts)]) / y_std_center ** 2
y_std_total = np.sqrt(1 /
                      (reduce((lambda x, y: x + y), [beta[i] * 1 / y_std_list[i] ** 2 for i in range(n_parts)]) -
                       integral_beta)
                      )
# y_std_total = np.sqrt(1 / reduce((lambda x, y: x + y), [beta[i] / (y_std_list[i] ** 2) for i in range(n_parts)]))
y_mean_total = (
        (y_std_total ** 2) *
        (
                reduce((lambda x, y: x + y),
                       [beta[i] * y_mean_list[i][:, 0] / (y_std_list[i] ** 2) for i in range(n_parts)]
                       ) -
                integral_beta * y_mean_center.transpose()
        )).transpose()

# y_std_total = np.sqrt(1 / reduce((lambda x, y: x + y), [beta[i] / (y_std_list[i] ** 2) for i in range(n_parts)]))
# y_mean_total = np.expand_dims(
#     (y_std_total ** 2) *
#     reduce((lambda x, y: x + y),
#            [beta[i] * y_mean_list[i][:, 0] / (y_std_list[i] ** 2) for i in range(n_parts)]
#            )
#     , axis=0).transpose()

# obj_gp_1.gp.get_params()
# obj_gp_2.gp.get_params()
# obj_gp_1.gp.kernel
#
plot_gp(iFig=1, x=x_test, y_mean=y_mean, y_std=np.expand_dims(y_std, axis=0).transpose(), x_train=x_train,
        y_train=y_train)
# plot_gp(iFig=2, x=x_test, y_mean=y_mean_1, y_std=y_std_1, x_train=x_train_1, y_train=y_train_1)
# plot_gp(iFig=3, x=x_test, y_mean=y_mean_2, y_std=y_std_2, x_train=x_train_2, y_train=y_train_2)
plot_gp(iFig=4, x=x_test, y_mean=y_mean_total, y_std=y_std_total.transpose(), x_train=x_train, y_train=y_train)
