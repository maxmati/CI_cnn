import input
import scipy.io as sio
import numpy as np
import tensorflow as tf

import localization_net

DOWNLOAD_URL = 'https://pydio.memleak.pl/public/9fe535/dl/IIIT5K-Word_V3.0.tar.gz'


def load_data():
    mat = sio.loadmat('IIIT5K/traindata.mat')['traindata'][0]
    count = mat.shape[0]
    labels = np.ones([count, 22]) * -1
    images = []
    for i in range(0, count):
        word = mat[i]['GroundTruth'][0]
        image = mat[i]['ImgName'][0]
        images.append('IIT5K/' + image)
        for j in range(0, len(word)):
            labels[i, j] = ord(word[j]) - ord('A')
    return images, labels


def prepare_images(images):
    decoded_images = []
    for img in images:
        decoded_images.append(tf.image.decode_png(img, 1))

    padded = tf.image.pad_to_bounding_box(decoded_images, 0, 0, 2900, 710)

    return tf.cast(padded, tf.float64)  # TODO: maybe scale to 0-1


def main():
    input.extract_if_needed('IIIT5K', DOWNLOAD_URL)

    images, labels = load_data()
    images = prepare_images(images)

    print(images.shape)


# 2900x710

if __name__ == '__main__':
    images = ['IIT5K/train/6_7.png', 'IIT5K/train/13_2.png']
    input = prepare_images(images)
    print(input.shape)
    print(localization_net.get_network(input, 2900, 712))
    main()
