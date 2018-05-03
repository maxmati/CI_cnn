import tensorflow as tf
from network_ops import generate_sth_layer


def get_network(data):
    num_timesteps = 1

    conv1 = tf.layers.conv2d(
        inputs=data,
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
    pool2_flat = tf.reshape(pool2, [-1, pool2.shape[1] * pool2.shape[2] * pool2.shape[3]])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dense_with_time = tf.reshape(dense, [-1, 1, 1024])
    dense_broadcasted = tf.concat([dense_with_time for _ in range(0, num_timesteps)], 1)
    num_neurons = 300
    cell = tf.nn.rnn_cell.LSTMCell(num_neurons)

    lstm_output, _ = tf.nn.dynamic_rnn(cell, dense_broadcasted, dtype=tf.float64)

    flattenet_lstm_output = tf.reshape(lstm_output, [-1, num_neurons])

    transform = tf.layers.dense(inputs=flattenet_lstm_output, units=6, activation=tf.nn.relu)

    return tf.reshape(transform, [-1, num_timesteps, 6])
