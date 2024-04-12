import cv2
import numpy as np
from GFFB_data_loader import GFFBTreeSpecies
from GFFB_SCL_P_Net_train import Encoder, test
import torch
from common_usages import seed_everything


GPU = 0
n_classes = 12
n_test_way = n_classes
n_test_shot = 40
n_way = n_classes
n_shot = 5
temperature = 0.1


if __name__ == "__main__":
    seed_everything()
    fea = np.load('./data/GFFB_window_11_test_ratio_0.95_x.npy')
    y = np.load('./data/GFFB_window_11_test_ratio_0.95_y.npy')
    width, height, channel = 11, 11, 125
    encoder = Encoder(input_dim=channel, output_dim=64)
    encoder.cuda(GPU)
    N = 2
    model_path = './model/GFFB_SCL_P_Net_{}_{}_way_{}_shot_t_{}.pth'.format(N, n_way, n_shot, temperature)
    name = "GFFB_SCL_P_Net_classification_map"
    Xtrain = fea[:5342]
    ytrain = y[:5342]
    Xtrain = np.reshape(Xtrain, [-1, width, height, channel])
    X_dict = {}
    for i in range(n_classes):
        X_dict[i] = []
    tmp_count_index = 0
    for _, y_index in enumerate(ytrain):
        y_i = int(y_index)
        if y_i in X_dict:
            X_dict[y_i].append(Xtrain[tmp_count_index])
        else:
            X_dict[y_i] = []
            X_dict[y_i].append(Xtrain[tmp_count_index])
        tmp_count_index += 1
    for i in range(n_classes):
        arr = np.array(X_dict[i])
        if len(arr) < 10:
            arr = np.tile(arr, (20, 1, 1, 1))
        if len(arr) < 20:
            arr = np.tile(arr, (10, 1, 1, 1))
        if len(arr) < 40:
            arr = np.tile(arr, (2, 1, 1, 1))
        X_dict[i] = arr
    support_test = np.zeros([n_test_way, n_test_shot, width, height, channel], dtype=np.float32)
    epi_classes = np.arange(n_classes)
    for i, epi_cls in enumerate(epi_classes):
        selected = np.random.permutation(len(X_dict[epi_cls]))[:n_test_shot]
        support_test[i] = np.array(X_dict[epi_cls])[selected]
    support_test = support_test.transpose((0, 1, 4, 2, 3))
    support_test = np.reshape(support_test, [n_test_way * n_test_shot, channel, width, height])
    encoder.load_state_dict(torch.load(model_path))
    encoder.eval()
    del X_dict
    gffb = GFFBTreeSpecies(height)
    h, w, c = gffb.get_image_size()
    query_predict = np.zeros([w, height, width, channel], dtype=np.float32)
    predict_image = np.zeros([h, w], dtype=np.float32)
    step = 4
    for i in range(0, h, step):
        indexes = np.arange(i*w, (i + step) * w)
        query_predict = gffb.getPatches(indexes, height)
        predict_image[i:i+step] = np.reshape(test(support_test, query_predict.transpose((0, 3, 1, 2)), encoder), [step, -1])
        print(i)

    R = np.zeros([h, w], dtype='uint8')
    G = np.zeros([h, w], dtype='uint8')
    B = np.zeros([h, w], dtype='uint8')
    for i in range(h):
        for j in range(w):
            R[i, j], G[i, j], B[i, j] = gffb.get_color_by_index(predict_image[i, j])
    cv2.imwrite('./map/{}.png'.format(name), cv2.merge([B, G, R]))
    # cv2.imshow('Classifcation Map', cv2.merge([B, G, R]))
    # print('complete')
    # cv2.waitKey(0)
