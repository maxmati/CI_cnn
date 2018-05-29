import input
import scipy.io as sio
import numpy as np
import tensorflow as tf
import pickle
import matplotlib

import decode_model
import cv2

DOWNLOAD_URL = 'https://pydio.memleak.pl/public/9fe535/dl/IIIT5K-Word_V3.0.tar.gz'


def char_to_label(char):
    if ord('A') <= ord(char) <= ord('Z'):
        return ord(char) - ord('A') + 1

    return 26 + ord(char) - ord('0') + 1


def load_data_mat(name):
    mat = sio.loadmat('IIIT5K/' + name + '.mat')[name][0]
    count = mat.shape[0]
    # count = 100
    labels = np.zeros([count, 6], np.int32)
    images = []
    for i in range(0, count):
        word = mat[i]['GroundTruth'][0]
        image = mat[i]['ImgName'][0]
        images.append('IIIT5K/' + image)
        for j in range(0, min(len(word), 6)):
            labels[i, j] = char_to_label(word[j])
    return images, labels


def prepare_images(images):
    decoded_images = []
    for img in images:
        # with open(img, 'r') as content_file:
        decoded_images.append(cv2.resize(cv2.imread(img, 0), (1000, 230)))

    # padded = tf.image.pad_to_bounding_box(decoded_images, 0, 0, 2900, 710)

    return np.asarray(decoded_images, np.float32) / 255


# 2900x710


def model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 230, 1000, 1])

    logits = decode_model.detection_network(input_layer, 6)

    predictions = {
        # generate predictions (for predict and eval mode)
        "classes": tf.argmax(input=logits, axis=2),
        # add `softmax_tensor` to the graph. it is used for predict and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # calculate loss (for both train and eval modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # configure the training op (for train mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # add evaluation metrics (for eval mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def convert_if_needed(name):
    if input.check_if_exists('IIIT5K/' + name + '.pickle'):
        return
    images, labels = load_data_mat(name)

    with open('IIIT5K/' + name + '.pickle', 'wb') as f:
        pickle.dump((images, labels), f)


def load_data(name):
    with open('IIIT5K/' + name + '.pickle', 'rb') as f:
        return pickle.load(f)


def main():
    tf.logging.set_verbosity('DEBUG')
    input.extract_if_needed('IIIT5K', DOWNLOAD_URL)
    convert_if_needed('traindata')
    convert_if_needed('testdata')

    # with tf.device('/cpu:0'):
    images, labels = load_data('traindata')
    images = prepare_images(images)
    assert not np.any(np.isnan(images))
    assert not np.any(np.isnan(labels))

    eval_images, eval_labels = load_data('testdata')
    eval_images = prepare_images(eval_images)
    assert not np.any(np.isnan(eval_images))
    assert not np.any(np.isnan(eval_images))

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    # config.gpu_options.allow_growth = True

    # Set up logging for predictions
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": images},
        y=labels,
        batch_size=32,
        num_epochs=None,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_images},
        y=eval_labels,
        batch_size=32,
        shuffle=False)
    # num_epochs=1,
    # shuffle=True)

    classifier = tf.estimator.Estimator(
        config=tf.estimator.RunConfig(session_config=config),
        model_fn=model_fn)

    for _ in range(0, 100):
        classifier.train(
            input_fn=train_input_fn,
            steps=2000)

        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


if __name__ == '__main__':
    # images = ['IIT5K/train/6_7.png', 'IIT5K/train/13_2.png']
    # input = prepare_images(images)
    # print(input.shape)
    # print(decode_model.detection_network(input, 22))
    main()
