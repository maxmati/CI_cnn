import tensorflow as tf
import numpy as np
from network_ops import generate_sth_layer


def model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    layer = pool1
    for _ in range(0, 4):
        layer = generate_sth_layer(layer)

    pool2 = tf.layers.max_pooling2d(inputs=layer, pool_size=[2, 2], strides=2)

    # dense layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # dropout = dense
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # logits layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # generate predictions (for predict and eval mode)
        "classes": tf.argmax(input=logits, axis=1),
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


def main():
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    mnist_classifier = tf.estimator.Estimator(
        # config=tf.estimator.RunConfig(session_config=config),
        model_fn=model_fn)

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=128,
        num_epochs=None,
        shuffle=True)

    for _ in range(0, 20):
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=1000)

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity('INFO')
    main()
