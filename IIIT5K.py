import input
import scipy.io as sio
import numpy as np

DOWNLOAD_URL = 'https://pydio.memleak.pl/public/9fe535/dl/IIIT5K-Word_V3.0.tar.gz'


def load_data():
    mat = sio.loadmat('IIIT5K/traindata.mat')['traindata'][0]
    count = mat.shape[0]
    labels = np.ones([count, 22]) * -1
    images = []
    for i in range(0, count):
        word = mat[i]['GroundTruth'][0]
        image = mat[i]['ImgName'][0]
        images.append(image)
        for j in range(0, len(word)):
            labels[i, j] = ord(word[j]) - ord('A')
    return images, labels

def main():
    input.extract_if_needed('IIIT5K', DOWNLOAD_URL)

    images, labels = load_data()
    print("Loaded data")

if __name__ == '__main__':
    main()
