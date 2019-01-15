import tensorflow as tf
import glob
import os
import numpy as np
import SimpleITK as sitk
import argparse

from functools import reduce
from conversions import fit_image
from data_augmentation import generate_data
from sklearn.model_selection import KFold


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def split_cross_val(path, n_folds=5):
    """
    Yields new train-test pair each fold
    """
    img_files_train = sorted(glob.glob('{}/train/img/*.nii.gz'.format(path)))
    seg_files_train = sorted(glob.glob('{}/train/seg/*.nii.gz'.format(path)))
    img_files_test = sorted(glob.glob('{}/test/img/*.nii.gz'.format(path)))
    seg_files_test = sorted(glob.glob('{}/test/seg/*.nii.gz'.format(path)))

    img_files = img_files_test + img_files_train
    seg_files = seg_files_test + seg_files_train
    img_seg = list(zip(img_files, seg_files))
    print(len(img_seg))
    k_fold = KFold(n_splits=n_folds, shuffle=False)
    for i, (train_indices, test_indices) in enumerate(k_fold.split(img_seg)):
        train = [img_seg[i] for i in train_indices]
        test = [img_seg[i] for i in test_indices]
        yield i, (train, test)


def files_to_tfrecord(files, save_to, mode, nfold, n_augm=0, shrinkFactor=None):

    target_path = os.path.join(save_to, "tfrecords_{}/".format(nfold))
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Open the TFRecords file
    with tf.python_io.TFRecordWriter(os.path.join(target_path, '{}.tfrecords'.format(mode))) as writer:
        for i, (img_path, seg_path) in enumerate(files):
            # Check match
            assert os.path.basename(img_path)[:4] == os.path.basename(seg_path)[:4], "Volumes and ground truths don't match"

            print('{}/{} original volumes processed'.format(i, len(files)))
            # Read volume and ground truth
            img = sitk.ReadImage(img_path)
            seg = sitk.ReadImage(seg_path)

            for _, res_img, res_seg in generate_data(img, seg, n_augm=n_augm):
                # Normalize the image
                img, seg = fit_image(res_img, res_seg, target_size=[80, 336, 336], shrinkFactor=shrinkFactor)

                # We have to reshape array to 1-D array before saving ~ 80*336*336
                flatten_shape = reduce(lambda x, y: x * y, img.shape)
                img = np.reshape(img, [flatten_shape, ])
                seg = np.reshape(seg, [flatten_shape, ])

                # Create a feature
                feature = {'seg': _int64_feature(seg),
                           'img': _float_feature(img)}

                # feature = {'train/seg': _bytes_feature(seg.tobytes()),
                #            'train/img': _bytes_feature(img.tobytes())}
                #        'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())


def parse_seq(example_proto, dimensionality=[]):
    """
    Needed to read the stored .tfrecords data -- import this in your
    training script.
    :param example_proto: Protocol buffer of single example.
    :param dimensionality: Dimensionality of compressed single example.
    :return: Tensor containing the parsed sequence.
    """
    if len(dimensionality)==0:
        raise Warning("Dimensionality of the features "
                      "to unpack isn't defined")

    features = {"img": tf.FixedLenFeature(dimensionality, tf.float32),
                "seg": tf.FixedLenFeature(dimensionality, tf.int64)}

    parsed_features = tf.parse_single_example(example_proto, features)

    # decoded_img = tf.cast(parsed_features["img"], dtype=tf.float16)
    # decoded_seg = tf.cast(parsed_features["seg"], dtype=tf.uint8)

    return parsed_features["img"], parsed_features["seg"]
    # return decoded_img, decoded_seg


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Prepossess, augment data and store it to TFRecords")
    parser.add_argument("data",
                        help="Path to raw data"
                             " E.g. 'data")
    parser.add_argument("save_to",
                        type=str, default=None,
                        help="Path to save tfrecords")
    parser.add_argument("-a", "--augm",
                        type=int, default=0,
                        help="Number augm"
                             " Default: 0")
    parser.add_argument("-s", "--shrink",
                        type=int, default=None,
                        help="Number epochs to train the model"
                             " Default: None")

    args = parser.parse_args()

    for nfold, (train, test) in split_cross_val(args.data):
        if nfold:
            print("\t{}th fold to tfrecords...".format(nfold))
            files_to_tfrecord(train, save_to=args.save_to, mode="train", nfold=nfold, n_augm=args.augm, shrinkFactor=args.shrink)
            files_to_tfrecord(test, save_to=args.save_to, mode="eval", nfold=nfold, shrinkFactor=args.shrink)