from scipy import signal as sg
import scipy as scipy
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

dir_name = 'C:/Users/Lazloo/test/all/short/7_02124.png'
im = Image.open(dir_name).convert('L')
im_array = np.asarray(im)
# im_array_convolve = scipy.signal.convolve(in1=im_array, in2=[[1, -1]], mode="valid")


# plt.imshow(im_array)
# plt.show()
im_array_convolve = scipy.signal.convolve(in1=im_array, in2=[[1, -1], [-1, 1]], mode="full")
plt.imshow(im_array_convolve)
plt.show()

bool_max = im_array_convolve*(im_array_convolve == scipy.ndimage.filters.maximum_filter(input=im_array_convolve, size=(3, 3))) != 0
im_array_convolve[bool_max]
a = im_array_convolve*(im_array_convolve == scipy.ndimage.filters.maximum_filter(input=im_array_convolve, size=(3, 3)))
plt.imshow(a)
plt.show()
b = rescale(image=a, scale=0.5, mode='reflect', preserve_range = True)
plt.imshow(b)
plt.show()

scipy.misc.imshow(im_array_convolve)