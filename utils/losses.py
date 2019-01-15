import tensorflow as tf


from tensorflow.python.ops.losses.losses_impl import compute_weighted_loss
from tensorflow.python.ops.losses.losses_impl import Reduction

# from https://tensorlayer.readthedocs.io/en/stable/_modules/tensorlayer/cost.html


def dice_coef(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of da  ta, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background),
            dice = ```smooth/(small_value + smooth)``, then if smooth is very small,
            dice close to 0 (even the image values lower than the threshold),
            so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    # old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    # new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice, name='dice_coe')

    return dice


def dice(y_true, y_pred, axis=None, smooth=1e-07):
    """The Dice coefficient between predictions y_pred and labels y_true"""

    y_true.get_shape().assert_is_compatible_with(y_pred.get_shape())
    y_true = tf.layers.flatten(y_true)
    y_pred = tf.layers.flatten(y_pred)
    y_true = tf.to_float(y_true)
    y_pred = tf.to_float(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)


def dice_loss(y_true, y_pred, weights=1.0, scope=None,
                loss_collection=tf.GraphKeys.LOSSES,
                reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Dice loss for binary segmentation. The Dice loss is one minus the Dice
    coefficient, and therefore this loss converges towards zero.
    The Dice loss between predictions `p` and labels `g` is
    https://arxiv.org/pdf/1606.04797.pdf
    """

    losses = 1. - dice(y_true, y_pred)
    return compute_weighted_loss(
        losses=losses,
        weights=weights,
        scope=scope,
        loss_collection=loss_collection,
        reduction=reduction)


def streaming_dice(labels,
                   predictions,
                   axis=None):
    """Calculates Dice coefficient between `labels` and `features`.
    Both tensors should have the same shape and should not be one-hot encoded.
    """
    values = dice(labels, predictions, axis=axis)
    mean_dice, update_op = tf.metrics.mean(values)

    return mean_dice, update_op

