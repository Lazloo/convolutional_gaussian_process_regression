import GPR_Package.GPR_Script as GP
import pickle
import numpy as np
import Bayesian_Optimization_Package.Bayesian_Optimization_Script as BO

#dir = 'D:/Programmieren/image_data/test_short'
#label_choices = '01234567'
dir = 'D:/Programmieren/image_data/all2/all/train'
label_choices = '0123456789bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

gp_test = GP.GPR_Class()
gp_test.verbose_debug = False
gp_test.gp_package = 'sklearn'  # 'sklearn' 'gp_flow'

gp_test.load_data(dir_name=dir, label_choices=label_choices, ratio_train_test_eval=(0.005, 0.01, 0.01))
print('start')
# gp_test.train_and_test(windows_size=[(15, 12)], weights=np.array([[1, 1, -1, -1]]), weight_shape=[(2, 2)])

# lower_bounds = [*[2] * 4, *[-5] * 8]
# upper_bounds = [*[15] * 4, *[5] * 8]
# f = lambda x: \
#     gp_test.train_and_test(windows_size=[(x[0], x[1]), (x[2], x[3])],
#                            weights=np.array([[x[4], x[5], x[6], x[7]], [x[8], x[9], x[10], x[11]]]),
#                            weight_shape=[(2, 2), (2, 2)])
lower_bounds = [*[2] * 2 * 4, *[-5] * 4 * 4]
upper_bounds = [*[15] * 2 * 4, *[5] * 4 * 4]
f = lambda x: \
    gp_test.train_and_test(windows_size=[(x[0], x[1]), (x[2], x[3]), (x[4], x[5]), (x[6], x[7])],
                           weights=np.array([[x[8], x[9], x[10], x[11]], [x[12], x[13], x[14], x[15]],
                                             [x[16], x[17], x[18], x[19]], [x[20], x[21], x[22], x[23]]]),
                           weight_shape=[(2, 2), (2, 2), (2, 2), (2, 2)])
# lower_bounds = [*[2] * 2]
# upper_bounds = [*[15] * 2]
# f = lambda x: \
#     gp_test.train_and_test(windows_size=[(x[0], x[1])],
#                            weights=np.array([[1, 1, -1, -1]]),
#                            weight_shape=[(2, 2), (2, 2)])

bo_obj = BO.Bayesian_Optimization(objective_function=f, lower_bound=lower_bounds, upper_bound=upper_bounds)
bo_obj.verbose_debug = False
bo_obj.exploration_factor = 3
bo_obj.n_random_for_optimization = 1E6
bo_obj.do_optimization(n_random=500, n_iteration=100)
bo_obj.save_model('results/bo_100_all_2window')
# bo_obj.load_model('results/bo_100_2window')

# x_array = [bo_obj.x_list[i][0] for i in range(len(bo_obj.x_list))]
# y_array = [bo_obj.x_list[i][1] for i in range(len(bo_obj.x_list))]
# plt.figure()
# plt.scatter(x_array, y_array, c=bo_obj.y_list, cmap='Greys')
# plt.colorbar()
# plt.show()


# gp_test.train_and_test(windows_size=[(10, 10)], weights=np.array([[1, 1, -1, -1]]), weight_shape=[(2, 2)])
# [i for i in range(len(bo_obj.y_list)) if bo_obj.y_list[i] == max(bo_obj.y_list)]
# x = bo_obj.x_list[118]
# gp_test.load_data(dir_name=dir, label_choices=label_choices, ratio_train_test_eval=(0.25, 0.1, 0.1))
# out_test = gp_test.train_and_test(windows_size=[(x[0], x[1]), (x[2], x[3])],
#                        weights=np.array([[x[4], x[5], x[6], x[7]], [x[8], x[9], x[10], x[11]]]),
#                        weight_shape=[(2, 2), (2, 2)])