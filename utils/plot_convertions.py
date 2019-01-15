import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from conversions import get_center_slice, get_center_of_mass, crop_slices, crop_xy

np.set_printoptions(threshold=np.inf)

depth = 80
hight = 336
width = 336

data_dir = os.path.join(os.getcwd(), 'data')
img_dir = os.path.join(data_dir, 'train/img')
seg_dir = os.path.join(data_dir, 'train/seg')


def plot_cropped(image_segm_path, image_us_path):
    # read mask and US
    image_segm = sitk.ReadImage(image_segm_path)
    image_us = sitk.ReadImage(image_us_path)
    print("Plot {} volume".format(image_us_path))



    # convert to array
    image_segm_arr = sitk.GetArrayFromImage(image_segm)
    image_us_arr = sitk.GetArrayFromImage(image_us)

    # normalize the US matrix
    # int in [0, 255] -> float in [0, 1]
    image_us_arr = image_us_arr.astype(np.float32) / 255

    # binary the mask array
    threshold, upper, lower = 0.6, 1, 0
    image_segm_arr = np.where(image_segm_arr > threshold, upper, lower)

    center_slice_index = get_center_slice(image_segm_arr)
    cx, cy = get_center_of_mass(image_segm_arr, center_slice_index)

    # crop in depth (slices)
    image_segm_arr_d, image_us_arr_d = crop_slices(image_segm_arr, image_us_arr, center_slice_index,
                                                   max_n_slice=depth)
    # print('any', np.any(image_segm_arr_d, axis=(1, 2)))
    # print('shapes after crop in depth {} and  {}'.format(image_segm_arr_d.shape, image_segm_arr_d.shape))

    # crop in xy dimensions
    image_segm_arr_norm_size, image_us_arr_norm_size = crop_xy(image_segm_arr_d, image_us_arr_d,
                                                               cx, cy,
                                                               shape_width=width, shape_height=hight)

    # print('shapes after crop in xy dimensions {} and  {}'.format(image_segm_arr_norm_size.shape,
    #                                                              image_us_arr_norm_size.shape))

    slice_seg = center_slice_index
    slice_seg_d = 40

    # plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    alpha = 0.2

    axs[0].imshow(image_us_arr[slice_seg, :, :], cmap=plt.cm.Greys_r)
    axs[1].imshow(image_us_arr_d[slice_seg_d, :, :], cmap=plt.cm.Greys_r)
    axs[2].imshow(image_us_arr_norm_size[slice_seg_d, :, :], cmap=plt.cm.Greys_r)

    # axs[0].imshow(image_segm_arr[slice_seg, :, :], cmap=plt.cm.viridis, alpha=alpha)
    # axs[1].imshow(image_segm_arr_d[slice_seg_d, :, :], cmap=plt.cm.viridis, alpha=alpha)
    # axs[2].imshow(image_segm_arr_norm_size[slice_seg_d, :, :], cmap=plt.cm.viridis, alpha=alpha)

    # lot CoM
    patches = [Circle((cx, cy), radius=1, color='red')]
    for p in patches:
        axs[1].add_patch(p)
    plt.show(fig)


for index, filename_img in enumerate(os.listdir(img_dir)):
    if filename_img != '.DS_Store':
        filename_seg = filename_img[:4] + '_seg.nii.gz'
        plot_cropped(os.path.join(seg_dir, filename_seg), os.path.join(img_dir, filename_img))