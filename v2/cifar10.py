import tensorflow as tf
import numpy as np
from v2.network import model_fn


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def main():
    train1 = unpickle("cifar-10-batches-py/data_batch_1")
    train2 = unpickle("cifar-10-batches-py/data_batch_2")
    train3 = unpickle("cifar-10-batches-py/data_batch_3")
    train4 = unpickle("cifar-10-batches-py/data_batch_4")
    train5 = unpickle("cifar-10-batches-py/data_batch_5")
    eval = unpickle("cifar-10-batches-py/test_batch")

    train_data = np.concatenate((train1[b'data'], train2[b'data'], train3[b'data'], train4[b'data'], train5[b'data']))
    train_labels = np.asarray(
        train1[b'labels'] + train2[b'labels'] + train3[b'labels'] + train4[b'labels'] + train5[b'labels'])
    train_data = np.asarray(train_data, np.float32) / 255
    eval_data = np.asarray(eval[b'data'], np.float32) / 255

    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn((32, 32, 3)),
        model_dir='/tmp/tmpeqsbe8_a'
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=32,
        num_epochs=None,
        shuffle=True)

    for _ in range(0, 150):
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=1000)

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=np.asarray(eval[b'labels']),
            num_epochs=1,
            shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

    pass


if __name__ == '__main__':
    tf.logging.set_verbosity('DEBUG')
    main()
