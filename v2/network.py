import tensorflow as tf


def generate_resnet_layer(inputs, filters):
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=filters,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )

    return tf.add(inputs, conv2)


def model_fn(features, labels, mode):
    # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    input_layer = tf.reshape(features["x"], [-1, 200, 200, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    layer = pool1
    for _ in range(0, 3):
        layer = generate_resnet_layer(layer, 64)

    conv2 = tf.layers.conv2d(
        inputs=layer,
        filters=128,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    layer = pool2
    for _ in range(0, 3):
        layer = generate_resnet_layer(layer, 128)

    conv3 = tf.layers.conv2d(
        inputs=layer,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    layer = pool3
    for _ in range(0, 5):
        layer = generate_resnet_layer(layer, 256)

    conv4 = tf.layers.conv2d(
        inputs=layer,
        filters=512,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    layer = pool4
    for _ in range(0, 2):
        layer = generate_resnet_layer(layer, 512)

    pool5 = tf.layers.max_pooling2d(inputs=layer, pool_size=[2, 2], strides=2)


    # dense layer
    # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    pool2_flat = tf.reshape(pool5, [-1, 6 * 6 * 512])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # dropout = dense
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # logits layer
    # logits = tf.layers.dense(inputs=dropout, units=10)
    logits = tf.layers.dense(inputs=dropout, units=102)

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
