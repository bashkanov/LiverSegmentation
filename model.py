import os
import tensorflow as tf
import argparse
import numpy as np
from utils.losses import dice_loss, streaming_dice
from prepare_data import parse_seq

tf.logging.set_verbosity(tf.logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parametric_relu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                           initializer=tf.constant_initializer(0.1),
                           dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos + neg


def conv_conv_pool(input_, n_filters, name, pool=True, batch_norm=False, activation=tf.nn.relu):
    img = input_
    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            img = tf.layers.conv3d(
                img,
                F, (3, 3, 3),
                activation=None,
                padding='same',
                name="conv_{}".format(i + 1))
            if batch_norm:
                img = tf.layers.batch_normalization(img)

            img = activation(img)

        if pool is False:
            return img

        pool = tf.layers.max_pooling3d(img, (2, 2, 2), strides=(2, 2, 2), name="pool_{}".format(name))
        return img, pool


def upconv_3D(tensor, n_filter):
    return tf.layers.conv3d_transpose(
        tensor,
        filters=n_filter,
        kernel_size=(2, 2, 2),
        strides=(2, 2, 2),
        padding='same')


def upconv_concat(inputA, input_B, n_filter, name):
    up_conv = upconv_3D(inputA, n_filter)
    return tf.concat([up_conv, input_B], axis=-1)


def unet_3d_model_fn(imgs, gts, mode, params):
    # Append dimension for filters  (80, 366, 366) -> (80, 366, 366, 1)
    imgs = tf.expand_dims(imgs, 4)

    if params['leaky_relu']:
        activation = tf.nn.leaky_relu
    else:
        activation = tf.nn.relu

    levels = list()
    pool = imgs
    # Go down
    for layer_depth in range(params['depth']):
        if layer_depth < params['depth'] - 1:
            conv, pool = conv_conv_pool(pool, name=layer_depth, batch_norm=params['batch_norm'],
                                        n_filters=[params["n_base_filters"]*(2**layer_depth),
                                                   params["n_base_filters"]*(2**layer_depth)*2],
                                        activation=activation)
            levels.append([conv, pool])
        else:
            current_layer = conv_conv_pool(pool, name=layer_depth, pool=False, batch_norm=params['batch_norm'],
                                           n_filters=[params["n_base_filters"] * (2 ** layer_depth),
                                                      params["n_base_filters"] * (2 ** layer_depth) * 2],
                                           activation=activation)
            levels.append([current_layer])
    # Go up
    for i, layer_depth in enumerate(range(params['depth']-2, -1, -1)):
        concat = upconv_concat(current_layer, levels[layer_depth][0], n_filter=current_layer.shape[-1], name=params['depth']+i)
        current_layer = conv_conv_pool(concat, name=params['depth']+i, pool=False, batch_norm=params['batch_norm'],
                                       n_filters=[levels[layer_depth][0].shape[-1],
                                                  levels[layer_depth][0].shape[-1]],
                                       activation=activation)

    logits = tf.layers.conv3d(inputs=current_layer, filters=1, kernel_size=(1, 1, 1), padding='same', name='final')
    predictions = tf.nn.sigmoid(logits)
    # l2_loss = tf.losses.get_regularization_loss()
    # loss += l2_loss

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"predictions": predictions,
                       "probabilities": logits,
                       "imgs": imgs}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    gts = tf.expand_dims(gts, 4)
    # gts = tf.cast(gts, dtype=tf.float32)

    loss = dice_loss(gts, predictions)
    # loss = -tf.reduce_sum(dice(gts, predictions))
    # loss = tf.Print(loss, [loss])

    if mode == tf.estimator.ModeKeys.EVAL:
        specs = dict(mode=mode,
                     predictions={"preds": predictions,
                                  "probabilities": logits},
                     loss=loss,
                     eval_metric_ops={
                         "dice_eval": streaming_dice(labels=gts, predictions=predictions),
                         "loss_eval": tf.metrics.mean(loss)})

    else: # TRAIN
        global_step = tf.train.get_or_create_global_step()
        opt = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = opt.minimize(loss=loss, global_step=global_step)
        specs = dict(
            mode=mode,
            loss=loss,
            train_op=train_op,
        )

    return tf.estimator.EstimatorSpec(**specs)


def input_fn(path, dimensionality, train, batch_size=2, n_epochs=1):
    print("{} dataset is used...".format(path))
    data = tf.data.TFRecordDataset(path)
    if train:
        data = data.shuffle(buffer_size=802)
        data = data.repeat(n_epochs)
    data = data.map(lambda x: parse_seq(x, dimensionality=dimensionality))
    data = data.batch(batch_size=batch_size)
    return data


def call_train_and_eval(data_path, estimator, dimensionality, n_epochs=None, batch_size=None):

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(data_path + "/train.tfrecords", dimensionality, train=True,
                                  batch_size=batch_size, n_epochs=n_epochs))

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(data_path + "/eval.tfrecords", dimensionality, train=False,
                                  batch_size=2),
        steps=None,       # Evaluate until input_fn raises end-of-input.
        throttle_secs=0)  # Evaluate every epoch.

    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec)


if __name__ == '__main__':
    # python model.py predict_eval data liver_model

    parser = argparse.ArgumentParser(description="Train, evaluate, and predict the model")
    parser.add_argument("mode", help="Mode: {train, eval, predict, predict_eval}")
    parser.add_argument("data_path", help="Path to data dir with train and eval .tfrecords")
    parser.add_argument("model_dir", help="Path where to save/restore checkpoints and summaries")
    parser.add_argument("-tfr", "--tf_record_path",
                        type=str, default=None,
                        help="Path to tf.records to predict"
                             " Default: None")
    parser.add_argument("-e", "--epochs",
                        type=int, default=1,
                        help="Number of epochs"
                             " Default: 1")
    parser.add_argument("-b", "--batch",
                        type=int, default=2,
                        help="Batch size"
                             " Default: 2")
    parser.add_argument("-bn", "--batch_norm", action="store_true",
                        help="Batch normalization"
                             " Default: 2")
    parser.add_argument("-s2", "--shrink", action="store_true",
                        help="Shrink dimentions by factor 2"
                             " Default: 0")
    parser.add_argument("-lr", "--leaky_relu", action="store_true",
                        help="Leaky relu")

    args = parser.parse_args()

    if args.shrink:
        dimensionality = [40, 168, 168]
    else:
        dimensionality = [80, 336, 336]

    if args.mode == "train":
        # n_train_records = sum(1 for _ in tf.python_io.tf_record_iterator(args.data_path + "/train.tfrecords"))
        # n_steps_per_epoch = math.ceil(n_train_records / args.batch)
        # max_steps = n_steps_per_epoch * args.epochs
        # tf.logging.info("Will train for {} steps".format(max_steps))
        n_steps_per_epoch = 210

        params = {"depth": 4,
                  "n_base_filters": 8,
                  "batch_norm": args.batch_norm,
                  "leaky_relu": args.leaky_relu,
                  "learning_rate": 0.0001
                  }

        config = tf.estimator.RunConfig(model_dir=args.model_dir, save_summary_steps=n_steps_per_epoch,
                                        save_checkpoints_steps=n_steps_per_epoch,
                                        keep_checkpoint_max=10)

        estimator = tf.estimator.Estimator(
            model_fn=lambda features, labels, mode: unet_3d_model_fn(features, labels, mode, params),
            config=config)

        call_train_and_eval(args.data_path,
                            estimator,
                            dimensionality=dimensionality,
                            n_epochs=args.epochs,
                            batch_size=args.batch)

    if args.mode == "eval":
        params = {"depth": 4,
                  "n_base_filters": 8,
                  "batch_norm": args.batch_norm,
                  "leaky_relu": args.leaky_relu}
        config = tf.estimator.RunConfig(model_dir=args.model_dir)
        estimator = tf.estimator.Estimator(
            model_fn=unet_3d_model_fn,
            config=config,
            params=params)

        estimator.evaluate(input_fn=lambda: input_fn(args.data_path + "/eval.tfrecords", dimensionality,
                                                     train=False, batch_size=2))

    if args.mode == "predict_eval":
        # Rebuild the input pipeline
        data = input_fn(args.data_path + "/eval.tfrecords", dimensionality, train=False, batch_size=1)
        iterator = data.make_one_shot_iterator()
        features, labels = iterator.get_next()

        # Rebuild the model
        params = {"depth": 4,
                  "n_base_filters": 8,
                  "batch_norm": args.batch_norm,
                  "leaky_relu": args.leaky_relu}
        predictions = unet_3d_model_fn(features, labels, tf.estimator.ModeKeys.EVAL, params).predictions

        pred_eval_dir = "predict_eval_{}".format(args.model_dir)

        for i in [os.path.join(pred_eval_dir, 'img'),
                  os.path.join(pred_eval_dir, 'seg'),
                  os.path.join(pred_eval_dir, 'pred'),
                  os.path.join(pred_eval_dir, 'soft_pred')]:
            if not os.path.exists(i):
                os.makedirs(i)

        # Manually load the latest checkpoint
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(args.model_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)

            ind = 0
            while True:
                try:
                    preds, lbls, ftrs = sess.run([predictions, labels, features])
                    predict = preds["preds"]
                    soft_predict = preds["probabilities"]
                    # remove filter dimension
                    predict = np.squeeze(predict, axis=4)
                    soft_predict = np.squeeze(soft_predict, axis=4)

                    # Loop through the batches and store predictions and labels
                    print("Save next {} batches...".format(predict.shape[0]))
                    for i in range(predict.shape[0]):
                        path_pred = os.path.join(pred_eval_dir, "pred/d{}_pred.npy".format(str(ind).rjust(3, "0")))
                        path_soft_pred = os.path.join(pred_eval_dir, "soft_pred/d{}_pred.npy".format(str(ind).rjust(3, "0")))
                        path_seg = os.path.join(pred_eval_dir, "seg/d{}_seg.npy".format(str(ind).rjust(3, "0")))
                        path_img = os.path.join(pred_eval_dir, "img/d{}_img.npy".format(str(ind).rjust(3, "0")))
                        np.save(path_pred, predict[i, :, :, :], allow_pickle=False)
                        np.save(path_soft_pred, soft_predict[i, :, :, :], allow_pickle=False)
                        np.save(path_seg, lbls[i, :, :, :], allow_pickle=False)
                        np.save(path_img, ftrs[i, :, :, :], allow_pickle=False)
                        ind += 1

                except tf.errors.OutOfRangeError:
                    print("Reached the last record...")
                    break

    if args.mode == "predict":
        if args.numpy:
            features = np.load(args.numpy)
            assert features.shape[1:] == dimensionality, "Volume to predict differs from shape of the model"
        else:
            features = np.full([1] + dimensionality, 1, dtype=np.float32)

        input_fn_p = tf.estimator.inputs.numpy_input_fn(
            x=features, batch_size=1,
            num_epochs=1,
            shuffle=False)

        params = {"depth": 4,
                  "n_base_filters": 8,
                  "batch_norm": args.batch_norm,
                  "leaky_relu": args.leaky_relu}
        config = tf.estimator.RunConfig(model_dir=args.model_dir, save_summary_steps=1000,
                                        save_checkpoints_steps=1000,
                                        keep_checkpoint_max=10)

        estimator = tf.estimator.Estimator(
            model_fn=lambda features, labels, mode: unet_3d_model_fn(features, labels, mode, params),
            config=config)

        predict_dir = "predicted_{}".format(args.model_dir)
        for i in [os.path.join(predict_dir, 'img'),
                  os.path.join(predict_dir, 'pred'),
                  os.path.join(predict_dir, 'soft_pred')]:
            if not os.path.exists(i):
                os.makedirs(i)

        # Loop through the batches and store predictions, probabilities and labels
        for ind, prediction in enumerate(estimator.predict(input_fn=input_fn_p)):
            predict = np.squeeze(prediction["predictions"], axis=3)
            probab = np.squeeze(prediction["probabilities"], axis=3)
            img = np.squeeze(prediction['imgs'], axis=3)

            print("Save...")
            path_pred = os.path.join(predict_dir, "pred/d{}_pred.npy".format(str(ind).rjust(3, "0")))
            path_prob = os.path.join(predict_dir, "prob/d{}_pred.npy".format(str(ind).rjust(3, "0")))
            path_img = os.path.join(predict_dir, "img/d{}_img.npy".format(str(ind).rjust(3, "0")))
            np.save(path_pred, predict[:, :, :], allow_pickle=False)
            np.save(path_prob, probab[:, :, :], allow_pickle=False)
            np.save(path_img, img[:, :, :], allow_pickle=False)
        print("All volumes are predicted and saved to: {}".format(predict_dir))