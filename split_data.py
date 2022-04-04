import os
from shutil import copy
import random
# 从原始数据集划分训练集和测试集


def mk_file(file):
    if not os.path.exists(file):
        os.makedirs(file)


data_file_path = './data'
data_classes = [clazz for clazz in os.listdir(data_file_path)]

for clazz in data_classes:
    mk_file('myData/train/' + clazz)

for clazz in data_classes:
    mk_file('myData/val/' + clazz)

# 训练接划分比例 0.2
split_rate = 0.2

for clazz in data_classes:
    clazz_path = data_file_path + '/' + clazz + '/'
    images = os.listdir(clazz_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num * split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            image_path = clazz_path + image
            new_path = 'myData/val/' + clazz
            copy(image_path, new_path)
        else:
            image_path = clazz_path + image
            new_path = 'myData/train/' + clazz
            copy(image_path, new_path)
        print('\r[{}] processing [{}/{}]'.format(clazz, index + 1, num), end="")
    print()