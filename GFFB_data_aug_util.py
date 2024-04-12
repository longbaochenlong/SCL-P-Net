import numpy as np


def random_rotate(imgs, is_random=True):
    start = 1
    if is_random:
        start = 0
    batch_size = imgs.shape[0]

    for i in range(batch_size):
        k_val = np.random.randint(start, 4)
        imgs[i] = np.rot90(np.copy(imgs[i]), k=k_val, axes=(0, 1))

    return imgs


def random_flip(imgs, is_random=True):
    batch_size = imgs.shape[0]
    if is_random:
        index = np.random.choice([0, 1], 1, p=[0.5, 0.5])
        if index[0] == 0:
            return imgs
    for i in range(batch_size):
        imgs[i] = np.flip(imgs[i], axis=np.random.randint(2))

    return imgs


def cutout(imgs, n_holes, length, z_length, is_random=True):
    if length == 0 or z_length == 0:
        return imgs

    if is_random:
        index = np.random.choice([0, 1], 1, p=[0.5, 0.5])
        if index[0] == 0:
            return imgs

    batch_size = imgs.shape[0]
    h = imgs.shape[1]
    w = imgs.shape[2]
    c = imgs.shape[3]
    mask = np.ones((h, w, c), np.float32)
    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1: y2, x1: x2, :] = 0.

    mask[w // 2, h//2, :] = 1.

    for i in range(batch_size):
        imgs[i] = imgs[i] * mask

    return imgs


def sample_pairing(x1, x2, is_random=True):
    if is_random:
        index = np.random.choice([0, 1], 1, p=[0.5, 0.5])
        if index[0] == 0:
            return x1
    batch_size = x1.shape[0]
    for i in range(batch_size):
        z = np.random.randint(i, batch_size)
        x1[i] = (x1[i] + x2[z]) / 2
    return x1


def mixup(x1, x2, alpha, is_random=True):
    if is_random:
        index = np.random.choice([0, 1], 1, p=[0.5, 0.5])
        if index[0] == 0:
            return x1
    n = len(x1)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, n)
    else:
        lam = np.array([1.0] * n)
    for i in range(n):
        z = np.random.randint(i, n)
        x1[i] = x1[i] * lam[i] + (1 - lam[i]) * x2[z]

    return x1


def cutmix(imgs1, imgs2, n_holes, length, is_random=True):
    if is_random:
        index = np.random.choice([0, 1], 1, p=[0.5, 0.5])
        if index[0] == 0:
            return imgs1

    batch_size = len(imgs1)
    h = imgs1.shape[1]
    w = imgs1.shape[2]
    c = imgs1.shape[3]
    tmp_imgs = np.copy(imgs1)
    for n in range(n_holes):
        y1 = np.random.randint(h - length + 1)
        x1 = np.random.randint(w - length + 1)
        y2 = np.random.randint(h - length + 1)
        x2 = np.random.randint(w - length + 1)

        for i in range(batch_size):
            z = np.random.randint(i, batch_size)
            # tmp_val = np.copy(imgs[i, y1: y1 + self.length, x1: x1 + self.length, :])
            imgs1[i, y1: y1 + length, x1: x1 + length, :] = np.copy(imgs2[z, y2: y2 + length, x2: x2 + length, :])
    imgs1[:, h//2, w//2, :] = tmp_imgs[:, h//2, w//2, :]
    return imgs1


def inner_class_query_augmentation(data, aug):
    n = len(data)
    for enum_i in range(n):
        if "rotate" in aug:
            data[enum_i] = random_rotate(np.copy(data[enum_i]), True)
        if "flip" in aug:
            data[enum_i] = random_flip(np.copy(data[enum_i]), True)
        if "sample_pairing" in aug:
            data[enum_i] = sample_pairing(np.copy(data[enum_i]), np.copy(data[enum_i]), True)
        if "cutout" in aug:
            data[enum_i] = cutout(np.copy(data[enum_i]), 1, 7, 125, True)
        if "mixup" in aug:
            data[enum_i] = mixup(np.copy(data[enum_i]), np.copy(data[enum_i]), 5, True)
        if "cutmix" in aug:
            data[enum_i] = cutmix(np.copy(data[enum_i]), np.copy(data[enum_i]), 1, 7, True)
    return data

