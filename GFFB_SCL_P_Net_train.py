import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from sklearn.metrics import classification_report, \
    confusion_matrix, \
    cohen_kappa_score, \
    precision_score, \
    accuracy_score
from GFFB_data_aug_util import *
from CBAM import ChannelFirst
from common_usages import *
from losses import SupConLoss

GPU = 0
"""Hyper-parameters setting"""
LEARNING_RATE = 0.0005
temperature = 0.1
n_epochs = 30
n_episodes = 100
n_classes = 12
n_way = n_classes
n_shot = 5
n_query = 15
n_test_way = n_classes
n_test_shot = 40


class Projector(nn.Module):  # Projection head
    def __init__(self, input_dim, projection_dim):
        super(Projector, self).__init__()
        self.projection_dim = projection_dim

        self.projector = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=self.projection_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=self.projection_dim, out_features=self.projection_dim, bias=True),
        )

    def forward(self, x):
        out = self.projector(x)
        out = torch.nn.functional.normalize(out)
        return out


class Encoder(nn.Module):  # Image encoder
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(output_dim, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(output_dim, momentum=0.01, affine=True),
            nn.ReLU())  # 13×13, no maxpool2d
        self.layer3 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(output_dim, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(output_dim, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.CBAMLayer1 = ChannelFirst(in_planes=output_dim, reduction_ratio=4)
        self.CBAMLayer2 = ChannelFirst(in_planes=output_dim, reduction_ratio=4)
        self.CBAMLayer3 = ChannelFirst(in_planes=output_dim, reduction_ratio=4)

    def forward(self, x):
        out = self.layer1(x)
        out = self.CBAMLayer1(out)
        out = self.layer2(out)
        out = self.CBAMLayer2(out)
        out = self.layer3(out)
        out = self.CBAMLayer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        return out


def euclidean_dist(x, y):  # Squared Euclidean distance
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).mean(2)


def test(support, query, encoder):
    sample_features = encoder(Variable(torch.from_numpy(support)).cuda(GPU))
    sample_features = sample_features.view(n_test_way, n_test_shot, 64)
    sample_features = torch.mean(sample_features, 1).squeeze(1)
    test_features = encoder(Variable(torch.from_numpy(query)).cuda(GPU))
    test_features = test_features.squeeze()
    dists = euclidean_dist(test_features, sample_features)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_test_way, len(query), -1)
    _, y_hat = log_p_y.max(2)
    predict_labels = torch.argmax(-dists, dim=1)
    return predict_labels.cpu().numpy()


def main():
    seed_everything()
    fea = np.load('./data/GFFB_window_11_test_ratio_0.95_x.npy')
    y = np.load('./data/GFFB_window_11_test_ratio_0.95_y.npy')
    width, height, channel = 11, 11, 125
    # Data augmentation, optional items: 'rotate', 'flip', 'sample_pairing', 'cutout', 'mixup', 'cutmix'
    augs = ['flip', 'cutout', 'mixup', 'cutmix']
    # The quantities of data augmentation methods in RandAugment
    N = 2
    test_num = 3000
    Xtest = fea[5342:]
    Xtrain = fea[:5342]
    ytest = y[5342:]
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
    # Repeating samples of weakly classes
    for i in range(n_classes):
        arr = np.array(X_dict[i])
        if len(arr) < 10:
            arr = np.tile(arr, (20, 1, 1, 1))
        if len(arr) < 20:
            arr = np.tile(arr, (10, 1, 1, 1))
        if len(arr) < 40:
            arr = np.tile(arr, (2, 1, 1, 1))
        X_dict[i] = arr
    a = datetime.now()
    encoder = Encoder(channel, output_dim=64)
    encoder.cuda(GPU)
    projector = Projector(input_dim=64, projection_dim=64)
    projector.cuda(GPU)
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    encoder_scheduler = StepLR(encoder_optim, step_size=2000, gamma=0.5)
    projector_optim = torch.optim.Adam(projector.parameters(), lr=LEARNING_RATE)
    projector_scheduler = StepLR(projector_optim, step_size=2000, gamma=0.5)
    supConLoss = SupConLoss(temperature=temperature)
    save_path = './model/GFFB_SCL_P_Net_{}_{}_way_{}_shot_t_{}.pth'.format(N, n_way, n_shot, temperature)
    support_test = np.zeros([n_test_way, n_test_shot, width, height, channel], dtype=np.float32)
    predict_dataset = np.zeros([len(ytest)], dtype=np.int32)
    epi_classes = np.arange(n_classes)
    for i, epi_cls in enumerate(epi_classes):
        selected = np.random.permutation(len(X_dict[epi_cls]))[:n_test_shot]
        support_test[i] = np.array(X_dict[epi_cls])[selected]
    support_test = support_test.transpose((0, 1, 4, 2, 3))
    support_test = np.reshape(support_test, [n_test_way * n_test_shot, channel, width, height])
    best_acc = 0
    last_epoch_loss_avrg = 0.
    last_epoch_acc_avrg = 0.
    for ep in range(n_epochs):
        last_epoch_loss_avrg = 0.
        last_epoch_acc_avrg = 0.
        # Model training
        encoder.train()
        for epi in range(n_episodes):
            epi_classes = np.arange(n_way)
            samples = np.zeros([n_way, n_shot, width, height, channel], dtype=np.float32)
            batches = np.zeros([n_way, n_query, width, height, channel], dtype=np.float32)
            batch_labels = []
            for i, epi_cls in enumerate(epi_classes):
                selected = np.random.permutation(len(X_dict[epi_cls]))[:n_shot + n_query]
                samples[i] = np.copy(np.array(X_dict[epi_cls])[selected[:n_shot]])
                batches[i] = np.copy(np.array(X_dict[epi_cls])[selected[n_shot:]])
                for s in selected[n_shot:]:
                    batch_labels.append(epi_cls)

            # Generating positive sample pair
            ops = np.random.choice(list(augs), N, replace=False)
            batches_aug = np.copy(batches)
            for op in ops:
                batches_aug = inner_class_query_augmentation(np.copy(batches_aug), op)
                batches = inner_class_query_augmentation(np.copy(batches), op)
            samples = samples.transpose((0, 1, 4, 2, 3))
            batches = batches.transpose((0, 1, 4, 2, 3))
            batches_aug = batches_aug.transpose((0, 1, 4, 2, 3))

            samples = np.reshape(samples, [n_way * n_shot, channel, width, height])
            batches = np.reshape(batches, [n_way * n_query, channel, width, height])
            batches_aug = np.reshape(batches_aug, [n_way * n_query, channel, width, height])

            # calculate features
            sample_features = encoder(Variable(torch.from_numpy(samples)).cuda(GPU))  # 5x64
            sample_features = sample_features.view(n_way, n_shot, 64)
            sample_features = torch.mean(sample_features, 1).squeeze(1)

            test_features = encoder(Variable(torch.from_numpy(batches_aug)).cuda(GPU))  # 20x64
            test_features_proj = projector(test_features)
            test_features_proj = test_features_proj.squeeze()

            test_features_org = encoder(Variable(torch.from_numpy(batches)).cuda(GPU))  # 20x64
            test_features_org_proj = projector(test_features_org)
            test_features_org_proj = test_features_org_proj.squeeze()
            test_features_org = test_features_org.squeeze()
            features = torch.cat([test_features_proj.unsqueeze(1), test_features_org_proj.unsqueeze(1)], dim=1)
            # SCL loss
            loss_scl = supConLoss(features, torch.from_numpy(np.array(batch_labels)).cuda(GPU))

            target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
            target_inds = Variable(target_inds, requires_grad=False).cuda(GPU)
            dists = euclidean_dist(test_features_org, sample_features)
            log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
            # SCL-Proto loss = Proto loss + SCL loss
            loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean() + loss_scl
            _, y_hat = log_p_y.max(2)
            acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
            encoder_optim.zero_grad()
            projector_optim.zero_grad()
            loss_val.backward()
            projector_optim.step()
            encoder_optim.step()
            projector_scheduler.step()
            encoder_scheduler.step()
            last_epoch_loss_avrg += loss_val.data
            last_epoch_acc_avrg += acc_val.data
            if (epi + 1) % 50 == 0:
                print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(ep + 1, n_epochs, epi + 1,
                                                                                         n_episodes, loss_val.data,
                                                                                         acc_val.data))

        # Model validation
        if (ep + 1) >= 5:
            encoder.eval()
            test_count = int(len(ytest) / test_num)
            for i in range(test_count):
                query_test = np.reshape(Xtest[i * test_num:(i + 1) * test_num], [-1, width, height, channel])
                query_test = query_test.transpose((0, 3, 1, 2))
                predict_dataset[i * test_num:(i + 1) * test_num] = test(support_test, query_test, encoder)
            query_test = np.reshape(Xtest[test_count * test_num:], [-1, width, height, channel])
            query_test = query_test.transpose((0, 3, 1, 2))
            predict_dataset[test_count * test_num:] = test(support_test, query_test, encoder)
            overall_acc = accuracy_score(ytest, predict_dataset)
            if overall_acc > best_acc:
                best_acc = overall_acc
                print('best acc: {:.2f}'.format(overall_acc * 100))
                torch.save(encoder.state_dict(), save_path)
    b = datetime.now()
    durn = (b - a).seconds
    print("Training time:", durn)
    print('Last loss:{:.5f}'.format(last_epoch_loss_avrg / n_episodes))
    print('Last acc:{:.2f}'.format(last_epoch_acc_avrg / n_episodes * 100))
    print('Testing...')
    encoder.load_state_dict(torch.load(save_path))
    encoder.eval()
    del X_dict
    test_count = int(len(ytest) / test_num)
    for i in range(test_count):
        query_test = np.reshape(Xtest[i * test_num:(i + 1) * test_num], [-1, width, height, channel])
        query_test = query_test.transpose((0, 3, 1, 2))
        predict_dataset[i * test_num:(i + 1) * test_num] = test(support_test, query_test, encoder)
    query_test = np.reshape(Xtest[test_count * test_num:], [-1, width, height, channel])
    query_test = query_test.transpose((0, 3, 1, 2))
    del Xtest
    predict_dataset[test_count * test_num:] = test(support_test, query_test, encoder)
    confusion = confusion_matrix(ytest, predict_dataset)
    acc_for_each_class = precision_score(ytest, predict_dataset, average=None)
    average_accuracy = np.mean(acc_for_each_class)
    kappa = cohen_kappa_score(ytest, predict_dataset)
    overall_acc = accuracy_score(ytest, predict_dataset)
    print('OA: {:.2f}'.format(overall_acc * 100))
    print('kappa:{:.4f}'.format(kappa))
    print('PA:')
    for i in range(len(acc_for_each_class)):
        print('{:.2f}'.format(acc_for_each_class[i] * 100))
    print('AA: {:.2f}'.format(average_accuracy * 100))


if __name__ == '__main__':
    main()
