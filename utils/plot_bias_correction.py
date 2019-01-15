import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
import myshow

def plot_corrected(image_us_path, image_corr_path, image_segm_path):
    # read mask and US
    image_segm = sitk.ReadImage(image_segm_path)
    image_us = sitk.ReadImage(image_us_path)
    image_corr = sitk.ReadImage(image_corr_path)

    # convert to array
    image_segm_arr = sitk.GetArrayFromImage(image_segm)
    image_us_arr = sitk.GetArrayFromImage(image_us)
    image_corr_arr = sitk.GetArrayFromImage(image_corr)

    # normalize the US matrix
    # int in [0, 255] -> float in [0, 1]
    # image_us_arr = image_us_arr.astype(np.float32) / 255

    # binary the mask array
    # threshold, upper, lower = 0.6, 1, 0
    # image_segm_arr = np.where(image_segm_arr > threshold, upper, lower)

    # plot
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))

    slice_seg = 40
    alpha = 0.2

    # axs[0].imshow(image_segm_arr[slice_seg, :, :], cmap=plt.cm.Greys_r)
    axs[0][0].imshow(image_us_arr[slice_seg-10, :, :], cmap=plt.cm.Greys_r)
    axs[1][0].imshow(image_corr_arr[slice_seg-10, :, :], cmap=plt.cm.Greys_r)

    axs[0][1].imshow(image_us_arr[slice_seg, :, :], cmap=plt.cm.Greys_r)
    axs[1][1].imshow(image_corr_arr[slice_seg, :, :], cmap=plt.cm.Greys_r)

    axs[0][2].imshow(image_us_arr[slice_seg+10, :, :], cmap=plt.cm.Greys_r)
    axs[1][2].imshow(image_corr_arr[slice_seg+10, :, :], cmap=plt.cm.Greys_r)

    # axs[0].imshow(image_segm_arr[slice_seg, :, :], cmap=plt.cm.viridis, alpha=alpha)
    # axs[1].imshow(image_segm_arr_d[slice_seg_d, :, :], cmap=plt.cm.viridis, alpha=alpha)
    # axs[2].imshow(image_segm_arr_norm_size[slice_seg_d, :, :], cmap=plt.cm.viridis, alpha=alpha)

    plt.show(fig)


if __name__ == '__main__':


    # plot_corrected('data/train/img/d021_img.nii.gz', 'data/train/corrected_invert/d021_img.nii.gz', 'data/train/seg/d021_seg.nii.gz')
    # plot_corrected('data/train/img/d043_img.nii.gz', 'data/train/corrected/d043_img.nii.gz', 'data/train/seg/d043_seg.nii.gz')
    #
    plot_corrected('data/train/img/d068_img.nii.gz', 'data/train/corrected_invert/d068_img.nii.gz', 'data/train/seg/d068_seg.nii.gz')
    # img_T1 = sitk.ReadImage('data/train/seg/d021_seg.nii.gz')
    # myshow.myshow3d(img_T1, "Corrected")


