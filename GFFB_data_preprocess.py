import numpy as np
from osgeo import gdal
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def scale(raster_array):
    scaler = preprocessing.StandardScaler()
    b, h, w = raster_array.shape
    raster_array = scaler.fit_transform(raster_array.reshape((b, h * w)).T)
    return raster_array.reshape(h, w, b)


def ReadX(img_file):
    img = gdal.Open(img_file)
    data = img.ReadAsArray()
    data = scale(data)
    return data


def ReadY(gt_file):
    img = gdal.Open(gt_file)
    data = img.ReadAsArray()
    return data


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    y_tmp = np.reshape(y, [-1])
    indexes = np.where(y_tmp > 0)[0]
    patchesData = np.zeros((len(indexes), windowSize, windowSize, X.shape[2]), dtype=np.float32)
    patchesLabels = y_tmp[indexes]
    count = 0
    for index in indexes:
        i = int(index / X.shape[1])
        j = index - i * X.shape[1]
        patchesData[count] = zeroPaddedX[i:i + windowSize, j:j + windowSize, :]
        count = count + 1
    patchesLabels -= 1
    return patchesData, patchesLabels


def split_train_and_test_set(X, y, test_ratio=0.10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=345,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def main():
    ratio = 0.95
    block_size = 11
    XPath = './data/GFFB_window_{}_test_ratio_{}_x.npy'.format(block_size, ratio)
    yPath = './data/GFFB_window_{}_test_ratio_{}_y.npy'.format(block_size, ratio)
    img_file = './data/sg_quac_test2.tif'
    gt_file = './data/sg_quac_test2_gt.tif'
    X = ReadX(img_file)
    y = ReadY(gt_file)
    X_image, y_image = createImageCubes(X, y, windowSize=block_size)
    X_train, X_test, y_train, y_test = split_train_and_test_set(X_image, y_image, ratio)
    X_train, X_val, y_train, y_val = split_train_and_test_set(X_train, y_train, 0.5)
    X = np.concatenate([X_train, X_val, X_test], axis=0)
    y = np.concatenate([y_train, y_val, y_test], axis=0)
    np.save(XPath, X)
    np.save(yPath, y)


if __name__ == '__main__':
    main()
