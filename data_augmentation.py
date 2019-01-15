import random as rn
import numpy as np
import SimpleITK as sitk

from conversions import correct_bias

rn.seed(53)  # seed random number generator for reproducible augmentation


def binaryThresholdImage(img, lowerThreshold):
    maxFilter = sitk.StatisticsImageFilter()
    maxFilter.Execute(img)
    maxValue = maxFilter.GetMaximum()
    thresholded = sitk.BinaryThreshold(img, lowerThreshold, maxValue, 1, 0)
    return thresholded


def similarity3D_parameter_space_regular_sampling(thetaX, thetaY, thetaZ, tx, ty, tz, scale):
    '''
    Create a list representing a regular sampling of the 3D similarity transformation parameter space. As the
    SimpleITK rotation parameterization uses the vector portion of a versor we don't have an
    intuitive way of specifying rotations. We therefor use the ZYX Euler angle parametrization and convert to
    versor.
    Args:
        thetaX, thetaY, thetaZ: numpy ndarrays with the Euler angle values to use.
        tx, ty, tz: numpy ndarrays with the translation values to use.
        scale: numpy array with the scale values to use.
    Return:
        List of lists representing the parameter space sampling (vx,vy,vz,tx,ty,tz,s).
    '''
    return [list(eul2quat(parameter_values[0], parameter_values[1], parameter_values[2])) +
            [np.asscalar(p) for p in parameter_values[3:]] for parameter_values in
            np.nditer(np.meshgrid(thetaX, thetaY, thetaZ, tx, ty, tz, scale))]


def eul2quat(ax, ay, az, atol=1e-8):
    # https: // github.com / SimpleITK / SPIE2018_COURSE / blob / master / utilities.py
    '''
    Translate between Euler angle (ZYX) order and quaternion representation of a rotation.
    Args:
        ax: X rotation angle in radians.
        ay: Y rotation angle in radians.
        az: Z rotation angle in radians.
        atol: tolerance used for stable quaternion computation (qs==0 within this tolerance).
    Return:
        Numpy array with three entries representing the vectorial component of the quaternion.

    '''
    # Create rotation matrix using ZYX Euler angles and then compute quaternion using entries.
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r = np.zeros((3, 3))
    r[0, 0] = cz * cy
    r[0, 1] = cz * sy * sx - sz * cx
    r[0, 2] = cz * sy * cx + sz * sx

    r[1, 0] = sz * cy
    r[1, 1] = sz * sy * sx + cz * cx
    r[1, 2] = sz * sy * cx - cz * sx

    r[2, 0] = -sy
    r[2, 1] = cy * sx
    r[2, 2] = cy * cx

    # Compute quaternion:
    qs = 0.5 * np.sqrt(r[0, 0] + r[1, 1] + r[2, 2] + 1)
    qv = np.zeros(3)
    # If the scalar component of the quaternion is close to zero, we
    # compute the vector part using a numerically stable approach
    if np.isclose(qs, 0.0, atol):
        i = np.argmax([r[0, 0], r[1, 1], r[2, 2]])
        j = (i + 1) % 3
        k = (j + 1) % 3
        w = np.sqrt(r[i, i] - r[j, j] - r[k, k] + 1)
        qv[i] = 0.5 * w
        qv[j] = (r[i, j] + r[j, i]) / (2 * w)
        qv[k] = (r[i, k] + r[k, i]) / (2 * w)
    else:
        denom = 4 * qs
        qv[0] = (r[2, 1] - r[1, 2]) / denom;
        qv[1] = (r[0, 2] - r[2, 0]) / denom;
        qv[2] = (r[1, 0] - r[0, 1]) / denom;
    return qv


def augment_images_spatial(original_image, reference_image, T0, T_aug, transformation_parameters, flip_hor, flip_z,
                           interpolator=sitk.sitkLinear, default_intensity_value=0.0):

    # if binary:
    #     dist_filter = sitk.SignedMaurerDistanceMapImageFilter()
    #     dist_filter.SetUseImageSpacing(True)
    #     original_image = dist_filter.Execute(gt_bg)
    #     original_image = sitk.Cast(original_image, sitk.sitkFloat32)
    #     sitk.WriteImage(original_image, 'dist_map.nrrd')
    '''
    Generate the resampled images based on the given transformations.
    Args:
        original_image (SimpleITK image): The image which we will resample and transform.
        reference_image (SimpleITK image): The image onto which we will resample.
        T0 (SimpleITK transform): Transformation which maps points from the reference image coordinate system
            to the original_image coordinate system.
        T_aug (SimpleITK transform): Map points from the reference_image coordinate system back onto itself using the
               given transformation_parameters. The reason we use this transformation as a parameter
               is to allow the user to set its center of rotation to something other than zero.
        transformation_parameters (List of lists): parameter values which we use T_aug.SetParameters().
        output_prefix (string): output file name prefix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        output_suffix (string): output file name suffix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        interpolator: One of the SimpleITK interpolators.
        default_intensity_value: The value to return if a point is mapped outside the original_image domain.
    '''
    if flip_hor:
        arr = sitk.GetArrayFromImage(original_image)
        arr = np.flip(arr, axis=2)
        flipped = sitk.GetImageFromArray(arr)
        flipped.CopyInformation(original_image)
        original_image = flipped

    if flip_z:
        arr = sitk.GetArrayFromImage(original_image)
        arr = np.flip(arr, axis=0)
        flipped = sitk.GetImageFromArray(arr)
        flipped.CopyInformation(original_image)
        original_image = flipped

    for current_parameters in transformation_parameters:
        T_aug.SetParameters(current_parameters)
        # Augmentation is done in the reference image space, so we first map the points from the reference image space
        # back onto itself T_aug (e.g. rotate the reference image) and then we map to the original image space T0.
        T_all = sitk.Transform(T0)
        T_all.AddTransform(T_aug)
        aug_image = sitk.Resample(original_image, reference_image, T_all,
                                  interpolator, default_intensity_value)
        return aug_image


def generate_and_check_if_doubles(vecs):
    '''
    prevents the same combinations
    '''
    #            delta            trans_X          transY           trans_z            scale             flip_hor          #  smooth
    rand_vec = [rn.randint(0, 4), rn.randint(1, 3), rn.randint(1, 3), rn.randint(0, 3), rn.randint(0, 4), rn.randint(0, 1), rn.randint(0, 7)]

    if rand_vec not in vecs:
        vecs.append(rand_vec)
        return rand_vec
    else:
        print("Vector duplication was prevented")
        return generate_and_check_if_doubles(vecs)


def generate_data(img, seg, n_augm=0, n_dim=3):
        """
        Creates iterable generator for augmentation.
        If augment number is 0 return unchanged.
        :param img:
        :param seg:
        :param n_augm:
        :param n_dim:
        :return:
        """

        print('\tOriginal + {} augmentation'.format(n_augm))
        yield 0, img, seg

        if n_augm:
            aug_transform = sitk.Similarity2DTransform() if n_dim == 2 else sitk.Similarity3DTransform()

            # Create the reference image with a zero origin, identity direction cosine matrix and n_dim
            reference_origin = np.zeros(n_dim)
            reference_direction = np.identity(n_dim).flatten()

            reference_size = img.GetSize()
            reference_spacing = img.GetSpacing()

            reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
            reference_image.SetOrigin(reference_origin)
            reference_image.SetSpacing(reference_spacing)
            reference_image.SetDirection(reference_direction)

            # Always use the TransformContinuousIndexToPhysicalPoint
            # to compute an indexed point's physical coordinates as
            # this takes into account size, spacing and direction cosines.
            reference_center = np.array(
                reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

            vecs = []
            for augm_index in range(0, n_augm):
                # print('\tAugmNr', augm_index+1)

                transform = sitk.AffineTransform(n_dim)
                transform.SetMatrix(img.GetDirection())
                transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)

                # Modify the transformation to align the centers of the original
                # and reference image instead of their origins.
                centering_transform = sitk.TranslationTransform(n_dim)
                img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
                centering_transform.SetOffset(
                    np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
                centered_transform = sitk.Transform(transform)
                centered_transform.AddTransform(centering_transform)

                # Set the augmenting transform's center so that rotation is around the image center.
                aug_transform.SetCenter(reference_center)

                # Vector of transformation paramaters where randomly values are picked from.
                delta_Arr = [-0.174533, -0.0872665, 0.0, 0.0872665, 0.174533]
                translationsXY_Arr = [-4, -2, 0.0, 2, 4]  # in mm
                translationsZ_Arr = [-6, -3, 0.0, 3, 6]
                scale_Arr = [0.8, 0.9, 1.0, 1.1, 1.2]
                flip_Arr = [True, False]
                smooth_Arr = [0, 0.2, 0, 0.3, 0, 0.4, 0, 0.5]

                # Randomly select parameters
                rand_vec = generate_and_check_if_doubles(vecs)

                delta_x = 0.0  # delta_Arr[rand_vec[0]]
                delta_y = 0.0  # delta_Arr[rand_vec[1]]
                delta_z = delta_Arr[rand_vec[0]]
                transl_x = translationsXY_Arr[rand_vec[1]]
                transl_y = translationsXY_Arr[rand_vec[2]]
                transl_z = translationsZ_Arr[rand_vec[3]]
                scale = scale_Arr[rand_vec[4]]

                flip_hor = flip_Arr[rand_vec[5]]
                # flip_hor = False
                # flip_z = flip_Arr[rand_vec[5]]
                flip_z = False
                smooth = smooth_Arr[rand_vec[6]]

                transformation_parameters_list = similarity3D_parameter_space_regular_sampling([delta_x], [delta_y], [delta_z],
                                                                                               [transl_x], [transl_y], [transl_z],
                                                                                               [scale])
                # spatial transformation of anatomical image
                res_img = augment_images_spatial(img, reference_image, centered_transform,
                                                 aug_transform, transformation_parameters_list, flip_hor,
                                                 flip_z=flip_z,
                                                 interpolator=sitk.sitkLinear, default_intensity_value=0.0)

                res_seg = augment_images_spatial(seg, reference_image, centered_transform,
                                                aug_transform, transformation_parameters_list, flip_hor,
                                                flip_z=flip_z,
                                                interpolator=sitk.sitkNearestNeighbor, default_intensity_value=0)

                yield augm_index+1, res_img, res_seg
    
#
# if __name__ == '__main__':
#     import os
#     import glob
#     from conversions import fit_image
#
#     # read dataset
#     img_file = 'data/train/img/d*_img.nii.gz'
#     seg_file = 'data/train/seg/d*_seg.nii.gz'
#     img_files = sorted(glob.glob(img_file))
#     seg_files = sorted(glob.glob(seg_file))
#
#     # create output folders if not exist
#     out_dir = 'data/train_augm'
#     imgs_out_dir = os.path.join(out_dir, 'img')
#     segs_out_dir = os.path.join(out_dir, 'seg')
#     [os.makedirs(i) for i in [out_dir, imgs_out_dir, segs_out_dir] if not os.path.exists(i)]
#
#     n_augm = 20
#
#     for img_path, seg_path in zip(img_files, seg_files):
#
#         # check data match
#         assert os.path.basename(img_path)[:4] == os.path.basename(seg_path)[:4], "Volumes and ground truths don't match"
#         print(os.path.basename(img_path), os.path.basename(seg_path))
#
#         # read volume and ground truth
#         img = sitk.ReadImage(img_path)
#         seg = sitk.ReadImage(seg_path)
#
#         for i_augm, res_img, res_seg in generate_data(img, seg, n_augm=n_augm):
#
#             img, seg = fit_image(res_img, res_seg, target_size=[80, 336, 336], shrinkFactor=2)
#
#             # save augmented volumes in separate files
#             # augm_i = str(i_augm).rjust(2, "0")
#             # name_img = "{}_{}_img".format(os.path.basename(img_path)[:4], augm_i)
#             # name_seg = "{}_{}_seg".format(os.path.basename(seg_path)[:4], augm_i)
#             #
#             # np.save(os.path.join(imgs_out_dir, name_img), img, allow_pickle=False)
#             # np.save(os.path.join(segs_out_dir, name_seg), seg, allow_pickle=False)