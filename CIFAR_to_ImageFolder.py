# -*-coding:utf-8-*-
import os

import numpy as np
import torchvision as tv
from skimage import io
from tqdm import tqdm


def Cifar(src, dest):
    train_set = tv.datasets.CIFAR10(src, train=True, download=True)
    test_set = tv.datasets.CIFAR10(src, train=False, download=True)

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
    Cifar('data', 'cifar10')
