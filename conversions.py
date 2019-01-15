import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def get_center_slice(arr_mask):
    """Define optimal center slice based on non-zero slices.
    :return: index of center_slice
    """
    slice_not_zero = np.any(arr_mask, axis=(1,2))
    first_occ = np.argmax(slice_not_zero == True)
    slice_not_zero_reverse = slice_not_zero[::-1]
    last_occ = len(slice_not_zero_reverse) - np.argmax(slice_not_zero_reverse) - 1
    return first_occ + (last_occ - first_occ) // 2


def get_center_of_mass(arr_mask, center_slice_index):
    # Define Center of Mass
    center_y, center_x = ndimage.measurements.center_of_mass(arr_mask[center_slice_index, :, :])
    return center_x, center_y


def crop_slices(arr_mask, arr_us, center_slice, max_n_slice=85):
    """
    Then crop slices in both directions started from center slice.
    Naive approach. It is assumed thar figures is convex
    :return: two 3d cropped matrices
    """
    depth = arr_mask.shape[0]
    assert arr_mask.shape == arr_us.shape, 'Matrices should have ' \
                                           'the same dimensionality {} != {}'.format(arr_mask.shape, arr_us.shape)

    # If number of slices to cut is more then matrix has add zero-padding-slices
    if depth < max_n_slice:
        padding_slice = np.zeros((arr_mask.shape[1], arr_mask.shape[2]))
        for i in range(max_n_slice - depth):
            if i % 2 == 0:
                arr_mask = np.insert(arr_mask, 0, padding_slice, axis=0)
                arr_us = np.insert(arr_us, 0, padding_slice, axis=0)
            else:
                arr_mask = np.insert(arr_mask, arr_mask.shape[0], padding_slice, axis=0)
                arr_us = np.insert(arr_us, arr_us.shape[0], padding_slice, axis=0)
        return arr_mask, arr_us

    if (max_n_slice % 2) == 0:
        half1 = half2 = int(max_n_slice/2)
    else:
        half1, half2 = max_n_slice//2, max_n_slice//2 + 1

    lower_bound = center_slice - half1
    upper_bound = center_slice + half2

    if lower_bound < 0:
        upper_bound = upper_bound + abs(lower_bound)
        lower_bound = 0
    if upper_bound > depth:
        diff = upper_bound - depth
        upper_bound = upper_bound - diff
        lower_bound = lower_bound - diff

    return arr_mask[lower_bound:upper_bound, :, :], arr_us[lower_bound:upper_bound, :, :]


def crop_xy(arr_mask, arr_us, center_x, center_y, target_x, target_y):
    """
    The shape of input arrays is (z,y,x),
    where z corresponds to number of slices.
    :return: Normalized pair of arrays
    """
    assert arr_mask.shape == arr_us.shape, 'Matrices should have ' \
                                           'the same dimensionality {} != {}'.format(arr_mask.shape, arr_us.shape)

    target_sizes = {1: target_y, 2: target_x}
    target_centers = {1: center_y, 2: center_x}

    axiss = [1, 2]
    # Interested are only y ans x axis(1 and 2 corresponding)
    for axis in axiss:
        # Append missing pixels/voxels
        if arr_mask.shape[axis] < target_sizes[axis]:
            for i in range(target_sizes[axis]-arr_mask.shape[axis]):
                if i % 2 == 0:
                    arr_mask = np.insert(arr_mask, arr_mask.shape[axis], 0, axis=axis)
                    arr_us = np.insert(arr_us, arr_us.shape[axis], 0, axis=axis)
                else:
                    arr_mask = np.insert(arr_mask, 0, 0, axis=axis)
                    arr_us = np.insert(arr_us, 0, 0, axis=axis)
        # Crop
        elif arr_mask.shape[axis] > target_sizes[axis]:
            half_axis = target_sizes[axis] / 2
            start_axis, end_axis = target_centers[axis] - half_axis, target_centers[axis] + half_axis

            # Check for bounds
            if start_axis < 0:
                start_axis, end_axis = 0, end_axis + abs(start_axis)
            if end_axis > arr_mask.shape[axis]:
                diff = end_axis - arr_mask.shape[axis]
                start_axis, end_axis = start_axis - diff, end_axis - diff
            start_axis, end_axis = int(round(start_axis)), int(round(end_axis))
            # Crop y axis
            if axis == 1:
                arr_mask = arr_mask[:, start_axis:end_axis, :]
                arr_us = arr_us[:, start_axis:end_axis, :]
            # Crop x axis
            if axis == 2:
                arr_mask = arr_mask[:, :, start_axis:end_axis]
                arr_us = arr_us[:, :, start_axis:end_axis]

    return arr_mask, arr_us


def correct_bias(inputImage, n_iter, n_fitting_levels=4):
    # # maskImage = sitk.OtsuThreshold(maskImage, 0, 1)
    # # invert mask
    # maskArray = sitk.GetArrayFromImage(maskImage)
    # maskArray = 1 - maskArray
    # maskImage = sitk.GetImageFromArray(maskArray)
    #
    # maskImage.SetOrigin(inputImage.GetOrigin())
    # maskImage.SetSpacing(inputImage.GetSpacing())
    # maskImage.SetDirection(inputImage.GetDirection())

    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter();
    corrector.SetMaximumNumberOfIterations([int(n_iter)] * n_fitting_levels)
    print("N4BiasFieldCorrection is running... "
          "It might take significant amount of time...")
    output = corrector.Execute(inputImage)

    return output


def fit_image(img, seg, target_size, shrinkFactor):

    if shrinkFactor:
        target_size = [int(item / shrinkFactor) for item in target_size]
        # Shrink image by factor
        seg = sitk.Shrink(seg, [int(shrinkFactor)] * seg.GetDimension())
        img = sitk.Shrink(img, [int(shrinkFactor)] * img.GetDimension())

    target_z, target_y, target_x = target_size

    # Normalize the intensity of MRI volume
    img = sitk.RescaleIntensity(img, 0, 1)

    # Convert to array
    seg_arr = sitk.GetArrayFromImage(seg)
    img_arr = sitk.GetArrayFromImage(img)

    # Binary the mask array
    threshold, upper, lower = 0.5, 1, 0
    seg_arr = np.where(seg_arr > threshold, upper, lower)

    center_slice_index = get_center_slice(seg_arr)
    com_x, com_y = get_center_of_mass(seg_arr, center_slice_index)
    seg_arr_d, img_arr_d = crop_slices(seg_arr, img_arr, center_slice_index, max_n_slice=target_z)
    seg_arr_norm_size, img_arr_norm_size = crop_xy(seg_arr_d, img_arr_d, com_x, com_y,
                                                   target_x=target_x, target_y=target_y)
    return img_arr_norm_size, seg_arr_norm_size

if __name__ == '__main__':
    import os
    import glob

    # read dataset
    img_file = '/data/oleksii/liver/data/test/img/d*_img.nii.gz'
    img_train_file = '/data/oleksii/liver/data/train/img/d*_img.nii.gz'
    img_files = sorted(glob.glob(img_file))
    img_train_files = sorted(glob.glob(img_train_file))

    # create output folders if not exist
    out_dir = '/data/oleksii/liver/data_corr_bias'
    imgs_train_dir = os.path.join(out_dir, 'train/img')
    imgs_test_dir = os.path.join(out_dir, 'test/img')
    [os.makedirs(i) for i in [out_dir, imgs_train_dir, imgs_test_dir] if not os.path.exists(i)]

    for i, img_path in enumerate(img_files):
        print('{}/{}, {} img processing...'.format(i, len(img_files), os.path.basename(img_path)))
        img = sitk.ReadImage(img_path)
        img = correct_bias(img, 30)
        sitk.WriteImage(img, os.path.join(imgs_test_dir, os.path.basename(img_path)))

    # for i, img_path in enumerate(img_train_files):
    #     print('{}/{}, {} img processing...'.format(i, len(img_path), os.path.basename(img_path)))
    #     img = sitk.ReadImage(img_path)
    #     img = correct_bias(img, 30)
    #     sitk.WriteImage(img, os.path.join(imgs_train_dir, os.path.basename(img_path)))