from scipy import signal as sg
import scipy as scipy
from  skimage.transform import rescale, resize, downscale_local_mean
import skimage.measure as measure
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def convolution_step_edges_max_pooling(im_array):
    im_array_convolve = scipy.signal.convolve(in1=im_array, in2=[[1, -1], [-1, 1]], mode="valid")
    bool_max = im_array_convolve*(
                   im_array_convolve == scipy.ndimage.filters.maximum_filter(input=im_array_convolve, size=(3, 3))
           ) != 0
    im_array_convolve_max = im_array_convolve*(
            im_array_convolve == scipy.ndimage.filters.maximum_filter(input=im_array_convolve, size=(3, 3))
    )
    im_array_convolve_max_reduce = measure.block_reduce(im_array_convolve_max, (2, 2), np.max)

    return im_array_convolve_max_reduce


dir_name = 'C:/Users/Lazloo/test/all/short/7_02124.png'
im = Image.open(dir_name).convert('L')
im_array = np.asarray(im)
# im_array_convolve = scipy.signal.convolve(in1=im_array, in2=[[1, -1]], mode="valid")


# plt.imshow(im_array)
# plt.show()
im_array_convolve = scipy.signal.convolve(in1=im_array, in2=[[1, -1], [-1, 1]], mode="valid")


bool_max = im_array_convolve*\
           (
                   im_array_convolve == scipy.ndimage.filters.maximum_filter(input=im_array_convolve, size=(3, 3))
           ) != 0
im_array_convolve_max = im_array_convolve*(
        im_array_convolve == scipy.ndimage.filters.maximum_filter(input=im_array_convolve, size=(3, 3))
)
im_array_convolve_max_rescale = rescale(image=im_array_convolve_max, scale=0.5, mode='reflect', preserve_range = True)
im_array_convolve_max_reduce = measure.block_reduce(im_array_convolve_max, (2, 2), np.max)
im_array_convolve_max_reduce_2 = measure.block_reduce(im_array_convolve, (2, 2), np.max)

plt.show()
plt.imshow(im_array)
plt.imshow(im_array_convolve)
plt.imshow(im_array_convolve_max)
plt.imshow(im_array_convolve_max_rescale)
plt.imshow(im_array_convolve_max_reduce)
plt.imshow(im_array_convolve_max_reduce_2)

plt.imshow(convolution_step_edges_max_pooling(im_array=im_array))

