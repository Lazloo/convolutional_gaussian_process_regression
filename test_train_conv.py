import GPR_Package.GPR_Script as GP
import time as time
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter

dir = 'images/short'
dir = 'images/test_short'
label_choices = '01234567'
# label_choices = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

gp_test = GP.GPR_Class()

windows_size = (10, 5)

t1 = time.time()
label_array = gp_test.read_labels(dir, label_choices=label_choices, output_type='binary')
images_array = gp_test.read_images(dir)
features = [gp_test.convolution_step_edges_max_pooling(i, windows_size=windows_size).flatten() for i in images_array]
indices_distributed, train_indices, test_indices, evaluation_indices = \
    gp_test.generate_indices_train_test_evaluation(label_array, ratio=(0.1, 0.1, 0.1))
t2 = time.time()
print('loading: ' + str(t2-t1))

t1 = time.time()
gp_test.create_GPR_model(itemgetter(*train_indices)(features), itemgetter(*train_indices)(label_array))
t2 = time.time()
print('training: ' + str(t2-t1))

t1 = time.time()
label_array_test = itemgetter(*test_indices)(label_array)
A = [gp_test.predict(features[test_indices[i]])[0].argmax()
     for i in range(len(test_indices))]
print(len([A[i] for i in range(len(A)) if A[i] == label_array_test[i].argmax()]) / len(A))
t2 = time.time()
print('evaluation: ' + str(t2-t1))

#
# unique_label = len(label_array[0])
# indices = [None]*unique_label
# n_data = len(label_array)
# for i in range(n_data):
#     idx = label_array[i].argmax()
#     if indices[idx] is None:
#         indices[idx] = []
#     else:
#         indices[idx] += [i]
#
# np.random.seed(0)
# indices_distributed = [None]*unique_label
# for i in range(unique_label):
#     n_data_partial = len(indices[i])
#     indices_rand = np.random.permutation(n_data_partial)
#     n_train = round(0.8*n_data_partial)
#     n_test = round((n_data_partial-n_train)/2)
#     # 0:X does not include X
#     indices_distributed[i] = [
#         itemgetter(*indices_rand[0:n_train])(indices[i]),
#         itemgetter(*indices_rand[n_train:n_train+n_test])(indices[i]),
#         itemgetter(*indices_rand[n_train+n_test:])(indices[i])
#     ]
#
# # label_array_test = gp_test.read_labels(dir_test, label_choices=label_choices, output_type='binary')
# # images_array_test = gp_test.read_images(dir_test)
# # features_test = [gp_test.convolution_step_edges_max_pooling(i, windows_size=windows_size).flatten()
# #                  for i in images_array_test]
#
#
#
# # t1 = time.time()
# # gp_test.create_GPR_model(features, label_array)
# # t2 = time.time()
# #
# # print('GPR took ' + str(t2 - t1) + 's')
# # gp_test.predict(features_test[0])[0].argmax()
# # print(gp_test.predict(features_test[0]))
# #
# # A = [gp_test.predict(features_test[i])[0].argmax() for i in range(len(features_test))]
# # print(len([A[i] for i in range(len(A)) if A[i] == label_array_test[i].argmax()]) / len(A))
#
#
#
# # t1 = time.time()
# # gp_test.create_GPR_model(features_test, label_array_test)
# # t2 = time.time()
# # A = [gp_test.predict(features[i])[0].argmax() for i in range(len(features))]
# # print(len([A[i] for i in range(len(A)) if A[i] == label_array[i].argmax()]) / len(A))
