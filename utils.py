# -*-coding:utf-8-*-
import math
import os
import random

import numpy as np
import torch
from torchvision import datasets
from matplotlib import pyplot as plt
from skimage import io
from tqdm import tqdm


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_img(imgs, size, labels=None, title=None):
    if size.__class__.__name__ == 'int':
        m = n = math.ceil(np.sqrt(size))
    elif len(size) == 2:
        m, n = size
    else:
        raise ValueError(size)
    for i in range(len(imgs)):
        plt.subplot(m, n, i + 1)
        plt.imshow(imgs[i].transpose((1, 2, 0)) if imgs[i].shape[0] == 3 else imgs[i])
        if labels is not None:
            plt.title(labels[i])
        plt.axis('off')
    if title is not None:
        plt.suptitle(title)
    plt.show()


def plot_acc(records, tags, title=None):
    colors = ['r', 'g', 'c', 'b', 'k']
    for i, record in enumerate(records):
        plt.plot(np.arange(len(record)), record, c=colors[i], label=tags[i])
        plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()


def toImageFolder(name, src, dest):
    train_set = datasets.__dict__[name](src, train=True, download=True)
    test_set = datasets.__dict__[name](src, train=False, download=True)

    character_train = [[] for _ in range(len(train_set.classes))]
    character_test = [[] for _ in range(len(train_set.classes))]

    trainset = []
    testset = []
    for i, (X, Y) in enumerate(train_set):  # 将train_set的数据和label读入列表
        trainset.append(list((np.array(X), np.array(Y))))
    for i, (X, Y) in enumerate(test_set):  # 将test_set的数据和label读入列表
        testset.append(list((np.array(X), np.array(Y))))

    for X, Y in trainset:
        character_train[Y].append(X)  # 32*32*3

    for X, Y in testset:
        character_test[Y].append(X)  # 32*32*3

    os.mkdir(os.path.join(dest, 'train'))
    os.mkdir(os.path.join(dest, 'test'))

    filename = os.path.join(dest, 'train.txt')
    with open(filename, 'w') as file_object:
        for i, per_class in enumerate(character_train):
            character_path = os.path.join(dest, 'train', train_set.classes[i])
            os.mkdir(character_path)
            for j, img in tqdm(enumerate(per_class)):
                img_path = os.path.join(character_path, str(j) + ".jpg")
                io.imsave(img_path, img)
                file_object.write(os.path.join(train_set.classes[i], str(j) + ".jpg") + "\t" + str(i) + "\n")

    filename = os.path.join(dest, 'test.txt')
    with open(filename, 'w') as file_object:
        for i, per_class in enumerate(character_test):
            character_path = os.path.join(dest, 'test', test_set.classes[i])
            os.mkdir(character_path)
            for j, img in tqdm(enumerate(per_class)):
                img_path = os.path.join(character_path, str(j) + ".jpg")
                io.imsave(img_path, img)
                file_object.write(os.path.join(test_set.classes[i], str(j) + ".jpg") + "\t" + str(i) + '\n')


# 将cifar100转化为imagenet的目录格式
if __name__ == '__main__':
    toImageFolder('CIFAR10', 'data', 'cifar10')
