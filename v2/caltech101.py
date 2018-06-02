import os
import tensorflow as tf
import numpy as np
from v2.network import model_fn
import cv2


def main():
    catagories = os.listdir('101_ObjectCategories/train')

    categories_map = {}
    for i, cat in enumerate(catagories):
        categories_map[cat] = i

    data_path = '101_ObjectCategories/train'
    train_data = []
    train_labels = []

    for cat in catagories:
        image_files = os.listdir(os.path.join(data_path, cat))
        for f in image_files:
            img = cv2.resize(cv2.imread(os.path.join(data_path, cat, f), 0), (50, 50))
            train_data.append(img - np.mean(img))
            train_labels.append(categories_map[cat])

    train_data = np.asarray(train_data, np.float32) / 255
    train_labels = np.asarray(train_labels)

    data_path = '101_ObjectCategories/val'
    eval_data = []
    eval_labels = []

    for cat in catagories:
        image_files = os.listdir(os.path.join(data_path, cat))
        for f in image_files:
            img = cv2.resize(cv2.imread(os.path.join(data_path, cat, f), 0), (50, 50))
            eval_data.append(img - np.mean(img))
            eval_labels.append(categories_map[cat])

    eval_data = np.asarray(eval_data, np.float32) / 255
    eval_labels = np.asarray(eval_labels)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn((50, 50, 3), 102))

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

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
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity('INFO')
    main()
