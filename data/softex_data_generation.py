'''

在做过直方图均衡化的图像上，通过人的肉眼，基本无法准确的分出软渗出的轮廓。为了和微血管瘤和硬渗出的training数据集在形式上保持一致。固有此类。
软渗出的ahe_images文件夹中的图像是没有做过直方图均衡化处理的。

'''

import argparse
import os
from glob import glob
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='segmentation post processing')
    parser.add_argument('--root', required=True)
    return parser.parse_args()

args = parse_args()

root = args.root

images_path = os.path.join(root, 'images')
labels_path = os.path.join(root, 'labels')
ahe_images_path = os.path.join(root, 'ahe_images')

assert os.path.isdir(images_path)
assert os.path.isdir(labels_path)

os.makedirs(ahe_images_path, exist_ok=True)

assert os.path.isdir(ahe_images_path)

images_list = glob(os.path.join(images_path, '*.png'))
labels_list = glob(os.path.join(labels_path, '*.png'))

images_list = [os.path.basename(f) for f in images_list]
labels_list = [os.path.basename(f) for f in labels_list]

for index in images_list:
    index_l = index.replace('.png', '_mask.png')
    if index_l in labels_list:
        index_ahe = os.path.join(ahe_images_path, index.replace('.png', '_ahe.png'))
        index_i = os.path.join(images_path, index)
        shutil.copy(index_i, index_ahe)
        print('copy from {0} to {1}.'.format(index_i, index_ahe))

