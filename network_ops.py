import tensorflow as tf

def generate_sth_layer(inputs):
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    return tf.add(inputs, conv2)