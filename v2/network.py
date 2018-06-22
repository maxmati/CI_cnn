import tensorflow as tf


def generate_resnet_layer(inputs, filters, conv_add=False):
    with tf.name_scope('res') as scope:

        conv1 = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            strides=(2, 2) if conv_add else (1, 1)
        )

        conv1 = tf.layers.dense(inputs=conv1, units=filters, activation=tf.nn.relu)


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


def model_fn(sizes, labels_count, learning_rate, n):
    def model_fn_impl(features, labels, mode):
        # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
        input_layer = tf.reshape(features["x"], [-1, sizes[0], sizes[1], sizes[2]])
        tf.summary.image("input", input_layer)
        input_layer = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), input_layer)

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=16,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu
        )

        # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        layer = conv1
        for _ in range(0, n):
            layer = generate_resnet_layer(layer, 16)

        layer = generate_resnet_layer(layer, 32, True)

        for _ in range(0, n-1):
            layer = generate_resnet_layer(layer, 32)

        layer = generate_resnet_layer(layer, 64, True)

        for _ in range(0, n-1):
            layer = generate_resnet_layer(layer, 64)

        # layer = generate_resnet_layer(layer, 512, True)
        #
        # for _ in range(0, 2):
        #     layer = generate_resnet_layer(layer, 512)

        pool5 = tf.layers.average_pooling2d(inputs=layer, pool_size=[2, 2], strides=2)


        # dense layer
        # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        pool2_flat = tf.reshape(pool5, [-1, int(sizes[0]/8) * int(sizes[1]/8) * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=100, activation=tf.nn.relu)
        # dropout = dense"
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        # logits layer
        logits = tf.layers.dense(inputs=dropout, units=labels_count)

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
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
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
