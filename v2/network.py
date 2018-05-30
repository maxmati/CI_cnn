import tensorflow as tf


def generate_resnet_layer(inputs, filters, conv_add=False):
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=(2, 2) if conv_add else (1, 1)
    )

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=filters,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )

    if conv_add:
        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=[1, 1],
            padding="same",
            activation=tf.nn.relu,
            strides=(2, 2)
        )

    return tf.add(inputs, conv2)


def model_fn(sizes):
    def model_fn_impl(features, labels, mode):
        # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
        input_layer = tf.reshape(features["x"], [-1, sizes[0], sizes[1], sizes[2]])

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=64,
            kernel_size=[7, 7],
            padding="same",
            activation=tf.nn.relu
        )

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        layer = pool1
        for _ in range(0, 3):
            layer = generate_resnet_layer(layer, 64)

        layer = generate_resnet_layer(layer, 128, True)

        for _ in range(0, 3):
            layer = generate_resnet_layer(layer, 128)

        layer = generate_resnet_layer(layer, 256, True)

        for _ in range(0, 5):
            layer = generate_resnet_layer(layer, 256)

        layer = generate_resnet_layer(layer, 512, True)

        for _ in range(0, 2):
            layer = generate_resnet_layer(layer, 512)

        pool5 = tf.layers.average_pooling2d(inputs=layer, pool_size=[2, 2], strides=2)


        # dense layer
        # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        pool2_flat = tf.reshape(pool5, [-1, int(sizes[0]/32) * int(sizes[1]/32) * 512])
        dense = tf.layers.dense(inputs=pool2_flat, units=1000, activation=tf.nn.relu)
        dropout = dense
        # dropout = tf.layers.dropout(
        #     inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

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

    return model_fn_impl
